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

func TestAnthropic_Do_HappyPath(t *testing.T) {
	var capturedReq map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path = %s, want /v1/messages", r.URL.Path)
		}
		if got := r.Header.Get("x-api-key"); got != "test-key" {
			t.Errorf("x-api-key = %q, want %q", got, "test-key")
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Errorf("missing anthropic-version header")
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedReq)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id": "msg_01ABC",
			"type": "message",
			"role": "assistant",
			"content": [{"type":"text","text":"Hello there"}],
			"model": "claude-sonnet-4-6",
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 12, "output_tokens": 3}
		}`)
	}))
	defer srv.Close()

	client := provider.NewAnthropicWithBaseURL("test-key", srv.URL)
	resp, err := client.Do(context.Background(), &provider.Request{
		Model:    "claude-sonnet-4-6",
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if resp.Content != "Hello there" {
		t.Errorf("Content = %q, want %q", resp.Content, "Hello there")
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 3 {
		t.Errorf("Usage = %+v, want {12,3}", resp.Usage)
	}
	if resp.Model != "claude-sonnet-4-6" {
		t.Errorf("Model = %q, want claude-sonnet-4-6", resp.Model)
	}

	// max_tokens must have been set (Anthropic requires it, our default is 1024).
	if got, ok := capturedReq["max_tokens"].(float64); !ok || got != 1024 {
		t.Errorf("max_tokens sent = %v, want 1024", capturedReq["max_tokens"])
	}
}

func TestAnthropic_Do_4xxError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"model not found"}}`)
	}))
	defer srv.Close()

	client := provider.NewAnthropicWithBaseURL("test-key", srv.URL)
	_, err := client.Do(context.Background(), &provider.Request{
		Model:    "claude-nonexistent",
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("Do: want error, got nil")
	}
	var apiErr *provider.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error type = %T, want *provider.APIError", err)
	}
	if apiErr.Status != 400 {
		t.Errorf("Status = %d, want 400", apiErr.Status)
	}
	if apiErr.Type != "invalid_request_error" {
		t.Errorf("Type = %q, want invalid_request_error", apiErr.Type)
	}
	if !strings.Contains(apiErr.Message, "model not found") {
		t.Errorf("Message = %q, want to contain 'model not found'", apiErr.Message)
	}
}

func TestAnthropic_Do_5xxError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"type":"overloaded_error","message":"try again later"}}`)
	}))
	defer srv.Close()

	client := provider.NewAnthropicWithBaseURL("test-key", srv.URL)
	_, err := client.Do(context.Background(), &provider.Request{
		Model:    "claude-sonnet-4-6",
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	var apiErr *provider.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error type = %T, want *provider.APIError", err)
	}
	if apiErr.Status != 503 {
		t.Errorf("Status = %d, want 503", apiErr.Status)
	}
}

func TestAnthropic_Do_ContextCancellation(t *testing.T) {
	// A server that never responds — we should abort via ctx cancel.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer srv.Close()

	client := provider.NewAnthropicWithBaseURL("test-key", srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel

	_, err := client.Do(ctx, &provider.Request{
		Model:    "claude-sonnet-4-6",
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("Do: want error from cancelled ctx, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want to wrap context.Canceled", err)
	}
}

func TestAnthropic_Do_Validation(t *testing.T) {
	client := provider.NewAnthropicWithBaseURL("test-key", "http://unreachable.invalid")
	cases := []struct {
		name string
		req  *provider.Request
	}{
		{"nil request", nil},
		{"empty model", &provider.Request{Messages: []provider.Message{{Role: "user", Content: "hi"}}}},
		{"no messages", &provider.Request{Model: "claude-sonnet-4-6"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := client.Do(context.Background(), tc.req)
			if err == nil {
				t.Fatal("want validation error, got nil")
			}
		})
	}
}

func TestAnthropic_Supports(t *testing.T) {
	client := provider.NewAnthropic("test-key")
	cases := []struct {
		model string
		want  bool
	}{
		{"claude-sonnet-4-6", true},
		{"claude-opus-4-6", true},
		{"claude-haiku-4-5", true},
		{"gpt-4o", false},
		{"gemini-1.5-pro", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := client.Supports(tc.model); got != tc.want {
			t.Errorf("Supports(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestAnthropic_Name(t *testing.T) {
	client := provider.NewAnthropic("test-key")
	if client.Name() != "anthropic" {
		t.Errorf("Name() = %q, want anthropic", client.Name())
	}
}
