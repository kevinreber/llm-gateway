package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenAIName is the stable identifier used in metrics + config.
const OpenAIName = "openai"

// openAIDefaultBaseURL is the production Chat Completions endpoint.
// Tests inject a httptest.Server URL via NewOpenAIWithBaseURL.
const openAIDefaultBaseURL = "https://api.openai.com"

// openAIModelPrefixes are the model-ID families this client claims.
//
// Loose prefix matching, same as Anthropic's: new model IDs ship
// constantly and routing them should not require a code change. The
// o-series entries have no trailing hyphen because the bare family names
// ("o3", "o4-mini") are themselves valid model IDs.
var openAIModelPrefixes = []string{"gpt-", "chatgpt-", "o1", "o3", "o4"}

// OpenAI implements Provider against the OpenAI Chat Completions API.
//
// Zero value is not usable; construct via NewOpenAI (production) or
// NewOpenAIWithBaseURL (tests). Safe for concurrent use — all fields are
// immutable after construction and http.Client is designed to be shared.
type OpenAI struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// NewOpenAI constructs a client against the production endpoint.
func NewOpenAI(apiKey string) *OpenAI {
	return NewOpenAIWithBaseURL(apiKey, openAIDefaultBaseURL)
}

// NewOpenAIWithBaseURL constructs a client against an arbitrary base
// URL. Used by tests, and by deployments pointing at an OpenAI-compatible
// endpoint (vLLM, LiteLLM, Together) that speaks the same wire format.
func NewOpenAIWithBaseURL(apiKey, baseURL string) *OpenAI {
	return &OpenAI{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			// Matches the Anthropic client. The resilience layer's
			// per-attempt deadline normally fires first; this stays as
			// the backstop for any call made without one.
			Timeout: 60 * time.Second,
		},
	}
}

// Name implements Provider.
func (o *OpenAI) Name() string { return OpenAIName }

// Supports implements Provider.
func (o *OpenAI) Supports(model string) bool {
	for _, p := range openAIModelPrefixes {
		if strings.HasPrefix(model, p) {
			return true
		}
	}
	return false
}

// Health implements Provider. GET /v1/models is the cheapest
// authenticated call OpenAI offers — no tokens, no completion. As with
// Anthropic, any 4xx counts as healthy: it proves we reached the API and
// it answered, which is the only thing the breaker cares about.
func (o *OpenAI) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, o.baseURL+"/v1/models", nil)
	if err != nil {
		return fmt.Errorf("build health request: %w", err)
	}
	o.setAuthHeaders(req)
	resp, err := o.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health probe: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 500 {
		return fmt.Errorf("openai health: %d", resp.StatusCode)
	}
	return nil
}

// Do implements Provider.
func (o *OpenAI) Do(ctx context.Context, req *Request) (*Response, error) {
	if err := validate(req); err != nil {
		return nil, err
	}

	body := openAIRequest{
		Model: req.Model,
		// max_completion_tokens, not max_tokens: the latter is
		// deprecated on Chat Completions and is rejected outright by the
		// reasoning models, which count their hidden reasoning tokens
		// against this budget.
		MaxCompletionTokens: req.MaxTokens,
		Messages:            openAIMessages(req),
		Temperature:         req.Temperature,
	}
	if body.MaxCompletionTokens == 0 {
		body.MaxCompletionTokens = defaultMaxTokens
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("%w: marshal: %w", ErrInvalidRequest, err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/v1/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	o.setAuthHeaders(httpReq)

	httpResp, err := o.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("call openai: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	if httpResp.StatusCode >= 400 {
		return nil, parseOpenAIError(httpResp)
	}

	var out openAIResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if len(out.Choices) == 0 {
		// A 200 with no choices is not something the API is supposed to
		// do. Treat it as an upstream fault rather than returning an
		// empty completion that looks like a legitimate answer.
		return nil, &APIError{
			Provider: OpenAIName,
			Status:   http.StatusBadGateway,
			Type:     "empty_response",
			Message:  "openai returned no choices",
		}
	}

	return &Response{
		Content:    out.Choices[0].Message.Content,
		Usage:      Usage{InputTokens: out.Usage.PromptTokens, OutputTokens: out.Usage.CompletionTokens},
		StopReason: normalizeOpenAIStopReason(out.Choices[0].FinishReason),
		Model:      out.Model,
	}, nil
}

func (o *OpenAI) setAuthHeaders(r *http.Request) {
	r.Header.Set("Authorization", "Bearer "+o.apiKey)
}

// openAIMessages folds Request.System into the message array.
//
// Anthropic takes the system prompt as a top-level field and OpenAI
// takes it as the first message, so the gateway's provider-agnostic
// Request keeps it separate and each client puts it where its API wants
// it. Doing this here rather than at the router is what lets an alias
// fail over between the two without the caller changing anything.
func openAIMessages(req *Request) []Message {
	if req.System == "" {
		return req.Messages
	}
	out := make([]Message, 0, len(req.Messages)+1)
	out = append(out, Message{Role: roleSystem, Content: req.System})
	return append(out, req.Messages...)
}

// roleSystem is OpenAI-only; Anthropic has no system role, which is why
// it isn't in the shared role constants in types.go.
const roleSystem = "system"

// openAIRequest is the on-the-wire request shape.
type openAIRequest struct {
	Model               string    `json:"model"`
	Messages            []Message `json:"messages"`
	MaxCompletionTokens int       `json:"max_completion_tokens"`
	Temperature         float64   `json:"temperature,omitempty"`
}

// openAIResponse is the on-the-wire response shape.
type openAIResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		// FinishReason: "stop" | "length" | "tool_calls" | "content_filter"
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// normalizeOpenAIStopReason translates OpenAI's finish_reason vocabulary
// into the Anthropic one the gateway exposes.
//
// The gateway's public surface is Anthropic-shaped — POST /v1/messages,
// input_tokens/output_tokens — so a client that handles "end_turn" and
// "max_tokens" should keep working when a request falls over to OpenAI.
// A fallback that changed the response vocabulary would push the
// provider difference onto every caller, which defeats the point of
// having a gateway. Unrecognized values pass through untranslated rather
// than being flattened into a wrong-but-familiar one.
func normalizeOpenAIStopReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	default:
		return reason
	}
}

// parseOpenAIError extracts the typed error body from a non-2xx
// response. OpenAI returns {"error":{"message":"...","type":"...","code":"..."}};
// a missing or broken body still yields a usable APIError with the status.
func parseOpenAIError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)
	var payload struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	_ = json.Unmarshal(body, &payload)

	// Prefer `code` over `type` when present: OpenAI's type is broad
	// ("invalid_request_error") while code is specific
	// ("context_length_exceeded"), and the specific one is what a caller
	// can actually branch on.
	kind := payload.Error.Code
	if kind == "" {
		kind = payload.Error.Type
	}
	return &APIError{
		Provider:   OpenAIName,
		Status:     resp.StatusCode,
		Type:       kind,
		Message:    payload.Error.Message,
		RetryAfter: parseRetryAfter(resp.Header.Get("Retry-After")),
	}
}
