package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// AnthropicName is the stable identifier used in metrics + config.
const AnthropicName = "anthropic"

// anthropicDefaultBaseURL is the production Messages API endpoint.
// Tests inject a httptest.Server URL via NewAnthropicWithBaseURL.
const anthropicDefaultBaseURL = "https://api.anthropic.com"

// anthropicAPIVersion pins the Messages API version. Anthropic evolves
// the API by requiring an explicit version header; upgrading it is a
// deliberate act, not something to leave floating.
const anthropicAPIVersion = "2023-06-01"

// defaultMaxTokens is applied when a caller omits max_tokens. Anthropic
// requires the field, so we can't just forward the zero value.
const defaultMaxTokens = 1024

// Anthropic implements Provider against the Anthropic Messages API.
//
// Zero value is not usable; construct via NewAnthropic (production) or
// NewAnthropicWithBaseURL (tests). The struct is safe for concurrent
// use — all fields are immutable after construction and the underlying
// http.Client is designed to be shared.
type Anthropic struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// NewAnthropic constructs a client against the production endpoint.
func NewAnthropic(apiKey string) *Anthropic {
	return NewAnthropicWithBaseURL(apiKey, anthropicDefaultBaseURL)
}

// NewAnthropicWithBaseURL constructs a client against an arbitrary base
// URL. Used by tests to point at a httptest.Server; production callers
// use NewAnthropic.
func NewAnthropicWithBaseURL(apiKey, baseURL string) *Anthropic {
	return &Anthropic{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// Name implements Provider.
func (a *Anthropic) Name() string { return AnthropicName }

// Supports implements Provider. Claude model IDs start with "claude-".
// This is loose on purpose — new model IDs land regularly and the
// gateway should not need a code change to route them.
func (a *Anthropic) Supports(model string) bool {
	return strings.HasPrefix(model, "claude-")
}

// Health implements Provider. We deliberately do NOT hit /v1/messages
// with a real completion (would burn tokens); instead we send an
// intentionally malformed request and treat any 4xx as "reachable and
// authenticated", which is what the breaker actually cares about.
func (a *Anthropic) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.baseURL+"/v1/messages", nil)
	if err != nil {
		return fmt.Errorf("build health request: %w", err)
	}
	a.setAuthHeaders(req)
	resp, err := a.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health probe: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 500 {
		return fmt.Errorf("anthropic health: %d", resp.StatusCode)
	}
	return nil
}

// Do implements Provider.
func (a *Anthropic) Do(ctx context.Context, req *Request) (*Response, error) {
	if req == nil {
		return nil, errors.New("provider: nil request")
	}
	if req.Model == "" {
		return nil, errors.New("provider: model is required")
	}
	if len(req.Messages) == 0 {
		return nil, errors.New("provider: at least one message is required")
	}

	body := anthropicRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		System:      req.System,
		Messages:    req.Messages,
		Temperature: req.Temperature,
	}
	if body.MaxTokens == 0 {
		body.MaxTokens = defaultMaxTokens
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/messages", bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	a.setAuthHeaders(httpReq)

	httpResp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("call anthropic: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	if httpResp.StatusCode >= 400 {
		return nil, parseAnthropicError(httpResp)
	}

	var out anthropicResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &Response{
		Content:    concatTextBlocks(out.Content),
		Usage:      out.Usage,
		StopReason: out.StopReason,
		Model:      out.Model,
	}, nil
}

func (a *Anthropic) setAuthHeaders(r *http.Request) {
	r.Header.Set("x-api-key", a.apiKey)
	r.Header.Set("anthropic-version", anthropicAPIVersion)
}

// anthropicRequest is the on-the-wire request shape. Kept separate from
// Request so the public type stays provider-agnostic.
type anthropicRequest struct {
	Model       string    `json:"model"`
	MaxTokens   int       `json:"max_tokens"`
	System      string    `json:"system,omitempty"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature,omitempty"`
}

// anthropicResponse is the on-the-wire response shape. Content is a
// list of typed blocks; we flatten text blocks into Response.Content.
type anthropicResponse struct {
	ID      string             `json:"id"`
	Type    string             `json:"type"`
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
	Model   string             `json:"model"`
	// StopReason: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"
	StopReason string `json:"stop_reason"`
	Usage      Usage  `json:"usage"`
}

type anthropicContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// APIError is a typed error returned when Anthropic responds with a
// non-2xx status. Callers can errors.As it to inspect Status + Type
// (for retry/breaker decisions later in Phase 3).
type APIError struct {
	Provider string
	Status   int
	Type     string
	Message  string
}

func (e *APIError) Error() string {
	if e.Type != "" {
		return fmt.Sprintf("%s %d %s: %s", e.Provider, e.Status, e.Type, e.Message)
	}
	return fmt.Sprintf("%s %d: %s", e.Provider, e.Status, e.Message)
}

// parseAnthropicError extracts the typed error body from a non-2xx
// response. Anthropic returns {"type":"error","error":{"type":"...","message":"..."}};
// we tolerate a missing/broken body by falling through to a generic message.
func parseAnthropicError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)
	var payload struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	_ = json.Unmarshal(body, &payload)
	return &APIError{
		Provider: AnthropicName,
		Status:   resp.StatusCode,
		Type:     payload.Error.Type,
		Message:  payload.Error.Message,
	}
}

func concatTextBlocks(blocks []anthropicContent) string {
	if len(blocks) == 1 && blocks[0].Type == "text" {
		return blocks[0].Text
	}
	var sb strings.Builder
	for _, b := range blocks {
		if b.Type == "text" {
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}
