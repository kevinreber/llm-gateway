package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// GeminiName is the stable identifier used in metrics + config.
const GeminiName = "gemini"

// geminiDefaultBaseURL is the production Generative Language endpoint.
const geminiDefaultBaseURL = "https://generativelanguage.googleapis.com"

// geminiModelPrefixes are the model-ID families this client claims.
var geminiModelPrefixes = []string{"gemini-"}

// geminiRoleModel is Gemini's name for the assistant turn.
//
// Every other provider the gateway speaks to calls it "assistant".
// Sending that string to Gemini is a 400, so the translation has to
// happen here rather than being pushed onto callers — an alias failing
// over from Anthropic to Gemini must not require the client to rewrite
// its message history.
const geminiRoleModel = "model"

// Gemini implements Provider against Google's generateContent API.
//
// Zero value is not usable; construct via NewGemini. Safe for
// concurrent use.
type Gemini struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// NewGemini constructs a client against the production endpoint.
func NewGemini(apiKey string) *Gemini {
	return NewGeminiWithBaseURL(apiKey, geminiDefaultBaseURL)
}

// NewGeminiWithBaseURL constructs a client against an arbitrary base
// URL. Used by tests and by Vertex-compatible endpoints.
func NewGeminiWithBaseURL(apiKey, baseURL string) *Gemini {
	return &Gemini{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// Name implements Provider.
func (g *Gemini) Name() string { return GeminiName }

// Supports implements Provider.
func (g *Gemini) Supports(model string) bool {
	for _, p := range geminiModelPrefixes {
		if strings.HasPrefix(model, p) {
			return true
		}
	}
	return false
}

// Health implements Provider. Listing models is the cheapest
// authenticated call; as with the other clients any 4xx counts as
// healthy, because it proves the API was reached and answered.
func (g *Gemini) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, g.baseURL+"/v1beta/models", nil)
	if err != nil {
		return fmt.Errorf("build health request: %w", err)
	}
	g.setAuthHeaders(req)
	resp, err := g.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health probe: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 500 {
		return fmt.Errorf("gemini health: %d", resp.StatusCode)
	}
	return nil
}

// Do implements Provider.
func (g *Gemini) Do(ctx context.Context, req *Request) (*Response, error) {
	if err := validate(req); err != nil {
		return nil, err
	}

	body := geminiRequest{
		Contents: geminiContents(req.Messages),
		GenerationConfig: geminiGenerationConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxTokens,
		},
	}
	if body.GenerationConfig.MaxOutputTokens == 0 {
		body.GenerationConfig.MaxOutputTokens = defaultMaxTokens
	}
	if req.System != "" {
		// A dedicated field, like Anthropic's — not a first message,
		// like OpenAI's. Gemini rejects a "system" role outright.
		body.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: req.System}},
		}
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("%w: marshal: %w", ErrInvalidRequest, err)
	}

	// The model goes in the path, so it has to be escaped: an
	// unescaped name containing a slash would silently retarget the
	// request at a different endpoint.
	endpoint := g.baseURL + "/v1beta/models/" + url.PathEscape(req.Model) + ":generateContent"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	g.setAuthHeaders(httpReq)

	httpResp, err := g.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("call gemini: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	if httpResp.StatusCode >= 400 {
		return nil, parseGeminiError(httpResp)
	}

	var out geminiResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if len(out.Candidates) == 0 {
		// Gemini returns zero candidates when a safety filter blocked
		// the prompt outright. That is a refusal, not a transport
		// failure, so it must not read as the upstream being broken —
		// promptFeedback carries the reason when there is one.
		return nil, &APIError{
			Provider: GeminiName,
			Status:   http.StatusBadRequest,
			Type:     "no_candidates",
			Message:  geminiBlockMessage(&out),
		}
	}

	return &Response{
		Content:    geminiText(out.Candidates[0].Content.Parts),
		Usage:      Usage{InputTokens: out.UsageMetadata.PromptTokenCount, OutputTokens: out.UsageMetadata.CandidatesTokenCount},
		StopReason: normalizeGeminiStopReason(out.Candidates[0].FinishReason),
		Model:      req.Model,
	}, nil
}

// setAuthHeaders sends the key as a header rather than the ?key= query
// parameter Google's docs lead with. A query string lands in proxy
// access logs and browser history; a header does not.
func (g *Gemini) setAuthHeaders(r *http.Request) {
	r.Header.Set("x-goog-api-key", g.apiKey)
}

// geminiContents translates the gateway's messages into Gemini's shape,
// mapping the assistant role onto Gemini's "model".
func geminiContents(messages []Message) []geminiContent {
	out := make([]geminiContent, 0, len(messages))
	for _, m := range messages {
		role := m.Role
		if role == RoleAssistant {
			role = geminiRoleModel
		}
		out = append(out, geminiContent{
			Role:  role,
			Parts: []geminiPart{{Text: m.Content}},
		})
	}
	return out
}

// geminiText concatenates a candidate's parts, matching how the
// Anthropic client flattens content blocks.
func geminiText(parts []geminiPart) string {
	if len(parts) == 1 {
		return parts[0].Text
	}
	var b strings.Builder
	for _, p := range parts {
		b.WriteString(p.Text)
	}
	return b.String()
}

// geminiBlockMessage explains a zero-candidate response when Gemini
// said why, and says so plainly when it did not.
func geminiBlockMessage(out *geminiResponse) string {
	if r := out.PromptFeedback.BlockReason; r != "" {
		return "gemini blocked the prompt: " + r
	}
	return "gemini returned no candidates"
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiGenerationConfig struct {
	Temperature     float64 `json:"temperature,omitempty"`
	MaxOutputTokens int     `json:"maxOutputTokens,omitempty"`
}

type geminiRequest struct {
	Contents          []geminiContent        `json:"contents"`
	SystemInstruction *geminiContent         `json:"systemInstruction,omitempty"`
	GenerationConfig  geminiGenerationConfig `json:"generationConfig"`
}

type geminiResponse struct {
	Candidates []struct {
		Content      geminiContent `json:"content"`
		FinishReason string        `json:"finishReason"`
	} `json:"candidates"`
	PromptFeedback struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

// normalizeGeminiStopReason maps Gemini's SCREAMING_CASE finish reasons
// onto the Anthropic-shaped vocabulary the gateway exposes.
func normalizeGeminiStopReason(reason string) string {
	switch reason {
	case "STOP":
		return "end_turn"
	case "MAX_TOKENS":
		return "max_tokens"
	default:
		// SAFETY, RECITATION, and anything Google adds later pass
		// through untranslated. Flattening an unrecognized reason into
		// a familiar one would tell the caller the model stopped
		// normally when it did not.
		return reason
	}
}

// parseGeminiError extracts the typed error body from a non-2xx
// response. Gemini returns {"error":{"code":N,"message":"...","status":"..."}}.
func parseGeminiError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)
	var payload struct {
		Error struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
			Status  string `json:"status"`
		} `json:"error"`
	}
	_ = json.Unmarshal(body, &payload)

	kind := payload.Error.Status
	if kind == "" {
		kind = "gemini_error"
	}
	return &APIError{
		Provider:   GeminiName,
		Status:     resp.StatusCode,
		Type:       kind,
		Message:    payload.Error.Message,
		RetryAfter: parseRetryAfter(resp.Header.Get("Retry-After")),
	}
}
