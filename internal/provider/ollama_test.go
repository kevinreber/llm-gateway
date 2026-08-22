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

	"github.com/kevinreber/llm-gateway/internal/cost"
	"github.com/kevinreber/llm-gateway/internal/provider"
)

func ollamaServer(t *testing.T, h http.HandlerFunc) *provider.Ollama {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return provider.NewOllamaWithBaseURL(srv.URL)
}

func TestOllama_Supports(t *testing.T) {
	o := provider.NewOllama()
	cases := map[string]bool{
		"ollama/llama3":  true,
		"ollama/mistral": true,
		// Bare names must not be claimed. `ollama pull mistral` and a
		// hosted Mistral endpoint are the same string, so a client that
		// claimed it would silently steal hosted traffic.
		"llama3":        false,
		"mistral":       false,
		"gpt-4o":        false,
		"claude-opus-5": false,
		"ollama/":       false,
		"":              false,
	}
	for model, want := range cases {
		if got := o.Supports(model); got != want {
			t.Errorf("Supports(%q) = %v, want %v", model, got, want)
		}
	}
}

func TestOllama_StripsThePrefixOnTheWire(t *testing.T) {
	var got string
	o := ollamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		_ = json.Unmarshal(body, &req)
		got = req.Model
		if req.Stream {
			t.Error("stream must be false; the gateway serves single completions")
		}
		_, _ = io.WriteString(w, `{"model":"llama3","message":{"role":"assistant","content":"hi"},"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":3}`)
	})

	resp, err := o.Do(context.Background(), &provider.Request{
		Model:    "ollama/llama3",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if got != "llama3" {
		t.Errorf("model sent upstream = %q, want the prefix stripped", got)
	}
	// The response keeps the prefixed name: the same weights served
	// locally and by a hosted vendor must not share a cost label.
	if resp.Model != "ollama/llama3" {
		t.Errorf("Response.Model = %q, want the prefixed name", resp.Model)
	}
	if resp.Content != "hi" {
		t.Errorf("content = %q", resp.Content)
	}
	if resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 3 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop reason = %q, want the normalized end_turn", resp.StopReason)
	}
}

func TestOllama_RejectsAnUnprefixedModel(t *testing.T) {
	o := ollamaServer(t, func(http.ResponseWriter, *http.Request) {
		t.Error("upstream must not be called for a model this client does not claim")
	})

	_, err := o.Do(context.Background(), &provider.Request{
		Model:    "llama3",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})
	if !errors.Is(err, provider.ErrInvalidRequest) {
		t.Errorf("err = %v, want ErrInvalidRequest", err)
	}
}

func TestOllama_SystemPromptBecomesTheFirstMessage(t *testing.T) {
	var roles []string
	o := ollamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Messages []provider.Message `json:"messages"`
		}
		_ = json.Unmarshal(body, &req)
		for _, m := range req.Messages {
			roles = append(roles, m.Role)
		}
		_, _ = io.WriteString(w, `{"message":{"content":"ok"},"done":true,"done_reason":"stop"}`)
	})

	if _, err := o.Do(context.Background(), &provider.Request{
		Model:    "ollama/llama3",
		System:   "be terse",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	}); err != nil {
		t.Fatalf("Do: %v", err)
	}
	if len(roles) != 2 || roles[0] != "system" {
		t.Errorf("roles = %v, want the system prompt first", roles)
	}
}

func TestOllama_ErrorsCarryTheStatus(t *testing.T) {
	// The status is what the retry and breaker layers classify on;
	// Ollama gives no type or code of its own.
	o := ollamaServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":"model is loading"}`)
	})

	_, err := o.Do(context.Background(), &provider.Request{
		Model:    "ollama/llama3",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})

	var apiErr *provider.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v, want *APIError", err)
	}
	if apiErr.Status != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", apiErr.Status)
	}
	if !strings.Contains(apiErr.Message, "model is loading") {
		t.Errorf("message = %q, want the upstream text", apiErr.Message)
	}
}

func TestOllama_PrefixAgreesWithThePricingTable(t *testing.T) {
	// internal/cost keeps its own copy of this prefix so the pricing
	// table does not depend on the client packages. If they drift, every
	// local completion starts logging "no price for model".
	if _, known := cost.Cents(provider.OllamaPrefix+"llama3", 10, 10); !known {
		t.Errorf("cost does not recognize %q as local; the two prefixes have drifted",
			provider.OllamaPrefix)
	}
}
