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

// OllamaName is the stable identifier used in metrics + config.
const OllamaName = "ollama"

// ollamaDefaultBaseURL is where Ollama listens by default.
const ollamaDefaultBaseURL = "http://localhost:11434"

// OllamaPrefix is the marker a model name must carry to route here.
//
// The other clients claim model families by prefix, which works because
// each vendor ships from a known namespace. Ollama has no such
// namespace: it serves whatever the operator pulled, and that set
// overlaps every other vendor's — `ollama pull mistral` and a hosted
// Mistral endpoint are the same string. Any prefix list would be both
// incomplete for local models and liable to steal a hosted one.
//
// Requiring an explicit marker is the only rule that cannot do either.
// It is stripped before the request goes out, so `ollama/llama3` reaches
// Ollama as `llama3`.
const OllamaPrefix = "ollama/"

// Ollama implements Provider against a local Ollama server's chat API.
//
// Zero value is not usable; construct via NewOllama. Safe for
// concurrent use.
type Ollama struct {
	baseURL    string
	httpClient *http.Client
}

// NewOllama constructs a client against the default local endpoint.
func NewOllama() *Ollama { return NewOllamaWithBaseURL(ollamaDefaultBaseURL) }

// NewOllamaWithBaseURL constructs a client against an arbitrary base
// URL. There is no API key: Ollama is unauthenticated, which is another
// reason it belongs on localhost or behind a network boundary.
func NewOllamaWithBaseURL(baseURL string) *Ollama {
	return &Ollama{
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			// Longer than the hosted clients' 60s on purpose. A local
			// model on CPU is genuinely slower than a hosted one on
			// accelerators, and timing out a request that was going to
			// succeed is the failure this number exists to avoid. The
			// resilience layer's per-attempt deadline is the real bound.
			Timeout: 180 * time.Second,
		},
	}
}

// Name implements Provider.
func (o *Ollama) Name() string { return OllamaName }

// Supports implements Provider. See OllamaPrefix for why this is an
// explicit marker rather than a family list.
func (o *Ollama) Supports(model string) bool {
	return strings.HasPrefix(model, OllamaPrefix) && len(model) > len(OllamaPrefix)
}

// Health implements Provider. GET / is Ollama's liveness surface and
// costs nothing.
func (o *Ollama) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, o.baseURL+"/", nil)
	if err != nil {
		return fmt.Errorf("build health request: %w", err)
	}
	resp, err := o.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health probe: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 500 {
		return fmt.Errorf("ollama health: %d", resp.StatusCode)
	}
	return nil
}

// Do implements Provider.
func (o *Ollama) Do(ctx context.Context, req *Request) (*Response, error) {
	if err := validate(req); err != nil {
		return nil, err
	}
	if !o.Supports(req.Model) {
		return nil, fmt.Errorf("%w: model %q is missing the %q prefix",
			ErrInvalidRequest, req.Model, OllamaPrefix)
	}

	body := ollamaRequest{
		Model: strings.TrimPrefix(req.Model, OllamaPrefix),
		// Streaming off: the gateway's surface is a single non-streaming
		// completion, and Ollama's default is a newline-delimited stream
		// that would decode as garbage against the struct below.
		Stream:   false,
		Messages: ollamaMessages(req),
		Options: ollamaOptions{
			Temperature: req.Temperature,
			NumPredict:  req.MaxTokens,
		},
	}
	if body.Options.NumPredict == 0 {
		body.Options.NumPredict = defaultMaxTokens
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("%w: marshal: %w", ErrInvalidRequest, err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/chat", bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := o.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("call ollama: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	if httpResp.StatusCode >= 400 {
		return nil, parseOllamaError(httpResp)
	}

	var out ollamaResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &Response{
		Content:    out.Message.Content,
		Usage:      Usage{InputTokens: out.PromptEvalCount, OutputTokens: out.EvalCount},
		StopReason: normalizeOllamaStopReason(out.DoneReason),
		// The prefixed name, not the bare one Ollama echoes. Cost
		// attribution and metrics key on this, and `llama3` on its own
		// says nothing about where it ran — the same weights served
		// locally and by a hosted vendor must not share a label.
		Model: req.Model,
	}, nil
}

// ollamaMessages folds Request.System into the message array, the same
// way the OpenAI client does. Ollama accepts a system role.
func ollamaMessages(req *Request) []Message {
	if req.System == "" {
		return req.Messages
	}
	out := make([]Message, 0, len(req.Messages)+1)
	out = append(out, Message{Role: roleSystem, Content: req.System})
	return append(out, req.Messages...)
}

type ollamaOptions struct {
	Temperature float64 `json:"temperature,omitempty"`
	// NumPredict is Ollama's max-output-tokens knob.
	NumPredict int `json:"num_predict,omitempty"`
}

type ollamaRequest struct {
	Model    string        `json:"model"`
	Messages []Message     `json:"messages"`
	Stream   bool          `json:"stream"`
	Options  ollamaOptions `json:"options"`
}

type ollamaResponse struct {
	Model   string `json:"model"`
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Done       bool   `json:"done"`
	DoneReason string `json:"done_reason"`
	// Token counts. Absent on some model backends, in which case the
	// cost row records zero — which is correct here, since local
	// inference has no per-token price anyway.
	PromptEvalCount int `json:"prompt_eval_count"`
	EvalCount       int `json:"eval_count"`
}

// normalizeOllamaStopReason maps Ollama's done_reason onto the
// Anthropic-shaped vocabulary the gateway exposes, so a client handling
// "end_turn" keeps working when an alias falls over to a local model.
func normalizeOllamaStopReason(reason string) string {
	switch reason {
	case "stop", "":
		return "end_turn"
	case "length":
		return "max_tokens"
	default:
		return reason
	}
}

// parseOllamaError builds a typed error from a non-2xx response. Ollama
// returns {"error":"..."} with no type or code, so the status carries
// the classification the retry and breaker layers need.
func parseOllamaError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)
	var payload struct {
		Error string `json:"error"`
	}
	_ = json.Unmarshal(body, &payload)

	msg := payload.Error
	if msg == "" {
		msg = strings.TrimSpace(string(body))
	}
	return &APIError{
		Provider:   OllamaName,
		Status:     resp.StatusCode,
		Type:       "ollama_error",
		Message:    msg,
		RetryAfter: parseRetryAfter(resp.Header.Get("Retry-After")),
	}
}
