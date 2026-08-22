package provider_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/kevinreber/llm-gateway/internal/provider"
)

func geminiServer(t *testing.T, h http.HandlerFunc) *provider.Gemini {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return provider.NewGeminiWithBaseURL("test-key", srv.URL)
}

const geminiOK = `{"candidates":[{"content":{"parts":[{"text":"hello"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":2}}`

func TestGemini_Supports(t *testing.T) {
	g := provider.NewGemini("k")
	for model, want := range map[string]bool{
		"gemini-2.5-pro":   true,
		"gemini-1.5-flash": true,
		"gpt-4o":           false,
		"claude-opus-5":    false,
		"ollama/llama3":    false,
	} {
		if got := g.Supports(model); got != want {
			t.Errorf("Supports(%q) = %v, want %v", model, got, want)
		}
	}
}

func TestGemini_SendsTheKeyAsAHeaderNotAQueryParam(t *testing.T) {
	// A query string lands in proxy access logs and browser history.
	var sawHeader, sawQuery bool
	g := geminiServer(t, func(w http.ResponseWriter, r *http.Request) {
		sawHeader = r.Header.Get("x-goog-api-key") == "test-key"
		sawQuery = r.URL.Query().Get("key") != ""
		_, _ = io.WriteString(w, geminiOK)
	})

	if _, err := g.Do(context.Background(), &provider.Request{
		Model:    "gemini-2.5-pro",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	}); err != nil {
		t.Fatalf("Do: %v", err)
	}
	if !sawHeader {
		t.Error("api key was not sent as a header")
	}
	if sawQuery {
		t.Error("api key leaked into the query string")
	}
}

func TestGemini_TranslatesTheAssistantRole(t *testing.T) {
	// Gemini calls it "model" and 400s on "assistant". An alias failing
	// over from Anthropic must not require the client to rewrite its
	// message history.
	var roles []string
	g := geminiServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Contents []struct {
				Role string `json:"role"`
			} `json:"contents"`
		}
		_ = json.Unmarshal(body, &req)
		for _, c := range req.Contents {
			roles = append(roles, c.Role)
		}
		_, _ = io.WriteString(w, geminiOK)
	})

	if _, err := g.Do(context.Background(), &provider.Request{
		Model: "gemini-2.5-pro",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: "hi"},
			{Role: provider.RoleAssistant, Content: "hello"},
			{Role: provider.RoleUser, Content: "again"},
		},
	}); err != nil {
		t.Fatalf("Do: %v", err)
	}
	want := []string{"user", "model", "user"}
	if len(roles) != len(want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Errorf("roles[%d] = %q, want %q", i, roles[i], want[i])
		}
	}
}

func TestGemini_SystemPromptUsesTheDedicatedField(t *testing.T) {
	// Not a first message, the way OpenAI wants it: Gemini rejects a
	// "system" role outright.
	var systemText string
	var roles []string
	g := geminiServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			SystemInstruction *struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"systemInstruction"`
			Contents []struct {
				Role string `json:"role"`
			} `json:"contents"`
		}
		_ = json.Unmarshal(body, &req)
		if req.SystemInstruction != nil && len(req.SystemInstruction.Parts) > 0 {
			systemText = req.SystemInstruction.Parts[0].Text
		}
		for _, c := range req.Contents {
			roles = append(roles, c.Role)
		}
		_, _ = io.WriteString(w, geminiOK)
	})

	if _, err := g.Do(context.Background(), &provider.Request{
		Model:    "gemini-2.5-pro",
		System:   "be terse",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	}); err != nil {
		t.Fatalf("Do: %v", err)
	}
	if systemText != "be terse" {
		t.Errorf("systemInstruction = %q", systemText)
	}
	for _, r := range roles {
		if r == "system" {
			t.Error("system leaked into contents as a role; Gemini rejects that")
		}
	}
}

func TestGemini_NormalizesTheResponse(t *testing.T) {
	g := geminiServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, geminiOK)
	})

	resp, err := g.Do(context.Background(), &provider.Request{
		Model:    "gemini-2.5-pro",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("content = %q", resp.Content)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop reason = %q, want end_turn (from STOP)", resp.StopReason)
	}
	if resp.Usage.InputTokens != 7 || resp.Usage.OutputTokens != 2 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestGemini_UnrecognizedFinishReasonPassesThrough(t *testing.T) {
	// Flattening SAFETY into end_turn would tell the caller the model
	// stopped normally when it was cut off.
	g := geminiServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"SAFETY"}],"usageMetadata":{}}`)
	})

	resp, err := g.Do(context.Background(), &provider.Request{
		Model:    "gemini-2.5-pro",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if resp.StopReason != "SAFETY" {
		t.Errorf("stop reason = %q, want SAFETY unchanged", resp.StopReason)
	}
}

func TestGemini_BlockedPromptIsA400NotAnUpstreamFault(t *testing.T) {
	// A safety block is a refusal of this request. Reporting it as 5xx
	// would count it against the provider's circuit breaker and take a
	// healthy vendor out of rotation because one caller sent something
	// it would not answer.
	g := geminiServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"candidates":[],"promptFeedback":{"blockReason":"SAFETY"}}`)
	})

	_, err := g.Do(context.Background(), &provider.Request{
		Model:    "gemini-2.5-pro",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})

	var apiErr *provider.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v, want *APIError", err)
	}
	if apiErr.Status != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", apiErr.Status)
	}
	if !strings.Contains(apiErr.Message, "SAFETY") {
		t.Errorf("message = %q, want the block reason", apiErr.Message)
	}
}

func TestGemini_ErrorBodyIsTyped(t *testing.T) {
	g := geminiServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Header().Set("Retry-After", "30")
		_, _ = io.WriteString(w, `{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`)
	})

	_, err := g.Do(context.Background(), &provider.Request{
		Model:    "gemini-2.5-pro",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})

	var apiErr *provider.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v, want *APIError", err)
	}
	if apiErr.Status != http.StatusTooManyRequests {
		t.Errorf("status = %d", apiErr.Status)
	}
	if apiErr.Type != "RESOURCE_EXHAUSTED" {
		t.Errorf("type = %q, want the upstream status", apiErr.Type)
	}
}

func TestGemini_ModelNameIsPathEscaped(t *testing.T) {
	// An unescaped name containing a slash would silently retarget the
	// request at a different endpoint.
	var path string
	g := geminiServer(t, func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.EscapedPath()
		_, _ = io.WriteString(w, geminiOK)
	})

	_, _ = g.Do(context.Background(), &provider.Request{
		Model:    "gemini-2.5-pro/../../evil",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})

	if strings.Contains(path, "/../") {
		t.Errorf("path = %q, want the model escaped", path)
	}
}
