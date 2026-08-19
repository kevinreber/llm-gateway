package provider_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/provider"
)

func TestOpenAI_Do_HappyPath(t *testing.T) {
	var capturedReq map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %s, want /v1/chat/completions", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer test-key")
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedReq)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id": "chatcmpl-123",
			"object": "chat.completion",
			"model": "gpt-4o-2024-08-06",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Hello there"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
		}`)
	}))
	defer srv.Close()

	client := provider.NewOpenAIWithBaseURL("test-key", srv.URL)
	resp, err := client.Do(context.Background(), &provider.Request{
		Model:     "gpt-4o",
		MaxTokens: 256,
		Messages:  []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}

	if resp.Content != "Hello there" {
		t.Errorf("Content = %q, want %q", resp.Content, "Hello there")
	}
	// OpenAI's prompt/completion naming is translated into the
	// gateway's input/output vocabulary, which is what lets the cost
	// tracker stay provider-agnostic.
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 3 {
		t.Errorf("Usage = %+v, want {12 3}", resp.Usage)
	}
	// The canonical dated ID is echoed back, and that is what billing
	// keys on.
	if resp.Model != "gpt-4o-2024-08-06" {
		t.Errorf("Model = %q, want gpt-4o-2024-08-06", resp.Model)
	}

	// max_completion_tokens, not the deprecated max_tokens: the
	// reasoning models reject the latter outright.
	if got, ok := capturedReq["max_completion_tokens"]; !ok || got != float64(256) {
		t.Errorf("max_completion_tokens = %v (present=%v), want 256", got, ok)
	}
	if _, ok := capturedReq["max_tokens"]; ok {
		t.Error("request sent max_tokens, which is deprecated on Chat Completions")
	}
}

func TestOpenAI_Do_NormalizesStopReason(t *testing.T) {
	// A client that handles Anthropic's vocabulary must keep working
	// when a request falls over to OpenAI. A fallback that changed the
	// response shape would push the provider difference onto every
	// caller, which is what a gateway exists to prevent.
	tests := []struct {
		wire string
		want string
	}{
		{"stop", "end_turn"},
		{"length", "max_tokens"},
		{"tool_calls", "tool_use"},
		{"content_filter", "content_filter"},
	}

	for _, tc := range tests {
		t.Run(tc.wire, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(w, `{
					"model": "gpt-4o",
					"choices": [{"index":0,"message":{"role":"assistant","content":"x"},"finish_reason":"`+tc.wire+`"}],
					"usage": {"prompt_tokens": 1, "completion_tokens": 1}
				}`)
			}))
			defer srv.Close()

			resp, err := provider.NewOpenAIWithBaseURL("k", srv.URL).Do(context.Background(), &provider.Request{
				Model:    "gpt-4o",
				Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
			})
			if err != nil {
				t.Fatalf("Do: %v", err)
			}
			if resp.StopReason != tc.want {
				t.Errorf("StopReason = %q, want %q", resp.StopReason, tc.want)
			}
		})
	}
}

func TestOpenAI_Do_SystemBecomesFirstMessage(t *testing.T) {
	// Anthropic takes the system prompt as a top-level field, OpenAI as
	// the first message. Folding it in here is what lets one alias fail
	// over to the other without the caller changing anything.
	var captured struct {
		Messages []provider.Message `json:"messages"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		_, _ = io.WriteString(w, `{"model":"gpt-4o","choices":[{"message":{"content":"x"},"finish_reason":"stop"}],"usage":{}}`)
	}))
	defer srv.Close()

	_, err := provider.NewOpenAIWithBaseURL("k", srv.URL).Do(context.Background(), &provider.Request{
		Model:    "gpt-4o",
		System:   "you are terse",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}

	if len(captured.Messages) != 2 {
		t.Fatalf("messages = %+v, want 2", captured.Messages)
	}
	if captured.Messages[0].Role != "system" || captured.Messages[0].Content != "you are terse" {
		t.Errorf("first message = %+v, want the system prompt", captured.Messages[0])
	}
	if captured.Messages[1].Role != provider.RoleUser {
		t.Errorf("second message role = %q, want user", captured.Messages[1].Role)
	}
}

func TestOpenAI_Do_ErrorsAreTyped(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		retryAfter string
		body       string
		wantType   string
		wantWait   time.Duration
	}{
		{
			name:     "auth",
			status:   http.StatusUnauthorized,
			body:     `{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key"}}`,
			wantType: "invalid_api_key",
		},
		{
			// code beats type: "invalid_request_error" is too broad for
			// a caller to branch on, "context_length_exceeded" is not.
			name:     "context length",
			status:   http.StatusBadRequest,
			body:     `{"error":{"message":"too long","type":"invalid_request_error","code":"context_length_exceeded"}}`,
			wantType: "context_length_exceeded",
		},
		{
			name:     "type when code is absent",
			status:   http.StatusBadRequest,
			body:     `{"error":{"message":"nope","type":"invalid_request_error"}}`,
			wantType: "invalid_request_error",
		},
		{
			name:       "rate limited",
			status:     http.StatusTooManyRequests,
			retryAfter: "7",
			body:       `{"error":{"message":"slow down","type":"rate_limit_error","code":"rate_limit_exceeded"}}`,
			wantType:   "rate_limit_exceeded",
			wantWait:   7 * time.Second,
		},
		{
			// A broken body must still yield a usable status rather than
			// a decode error that loses it.
			name:     "unparseable body",
			status:   http.StatusBadGateway,
			body:     `<html>502 Bad Gateway</html>`,
			wantType: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if tc.retryAfter != "" {
					w.Header().Set("Retry-After", tc.retryAfter)
				}
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer srv.Close()

			_, err := provider.NewOpenAIWithBaseURL("k", srv.URL).Do(context.Background(), &provider.Request{
				Model:    "gpt-4o",
				Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
			})
			if err == nil {
				t.Fatal("Do = nil, want an error")
			}

			var apiErr *provider.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %v, want *provider.APIError", err)
			}
			if apiErr.Provider != provider.OpenAIName {
				t.Errorf("Provider = %q, want openai", apiErr.Provider)
			}
			if apiErr.Status != tc.status {
				t.Errorf("Status = %d, want %d", apiErr.Status, tc.status)
			}
			if apiErr.Type != tc.wantType {
				t.Errorf("Type = %q, want %q", apiErr.Type, tc.wantType)
			}
			if apiErr.RetryAfter != tc.wantWait {
				t.Errorf("RetryAfter = %v, want %v", apiErr.RetryAfter, tc.wantWait)
			}
		})
	}
}

func TestOpenAI_Do_EmptyChoicesIsAnUpstreamFault(t *testing.T) {
	// A 200 with no choices is not something the API is supposed to do.
	// Returning an empty completion would look like a legitimate answer
	// and be billed as one.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"model":"gpt-4o","choices":[],"usage":{}}`)
	}))
	defer srv.Close()

	_, err := provider.NewOpenAIWithBaseURL("k", srv.URL).Do(context.Background(), &provider.Request{
		Model:    "gpt-4o",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("Do = nil, want an error")
	}
	var apiErr *provider.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v, want *provider.APIError", err)
	}
	if apiErr.Status < 500 {
		t.Errorf("Status = %d, want 5xx so the retry layer treats it as an upstream fault", apiErr.Status)
	}
}

func TestOpenAI_Do_InvalidRequest(t *testing.T) {
	// Never reaches the wire, and carries the sentinel so the retry
	// layer does not replay it or blame the provider for it.
	client := provider.NewOpenAIWithBaseURL("k", "http://127.0.0.1:1")

	for _, tc := range []struct {
		name string
		req  *provider.Request
	}{
		{"nil", nil},
		{"no model", &provider.Request{Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}}}},
		{"no messages", &provider.Request{Model: "gpt-4o"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := client.Do(context.Background(), tc.req)
			if !errors.Is(err, provider.ErrInvalidRequest) {
				t.Errorf("err = %v, want ErrInvalidRequest", err)
			}
		})
	}
}

func TestOpenAI_Supports(t *testing.T) {
	client := provider.NewOpenAI("k")
	for _, model := range []string{"gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-4.1-nano", "o3", "o4-mini", "chatgpt-4o-latest"} {
		if !client.Supports(model) {
			t.Errorf("Supports(%q) = false, want true", model)
		}
	}
	for _, model := range []string{"claude-sonnet-5", "llama3.2", "gemini-2.0-flash", ""} {
		if client.Supports(model) {
			t.Errorf("Supports(%q) = true, want false", model)
		}
	}
}

func TestOpenAI_Do_RespectsContextCancellation(t *testing.T) {
	// The gateway abandons abandoned work: an in-flight upstream call
	// has to die with the request that started it.
	//
	// release, rather than the handler waiting on its own request
	// context: an HTTP/1.1 server does not notice a client walking away
	// mid-response, so the handler would still be parked when
	// srv.Close() started waiting for it and the test would hang instead
	// of failing.
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := provider.NewOpenAIWithBaseURL("k", srv.URL).Do(ctx, &provider.Request{
		Model:    "gpt-4o",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("Do = nil, want a context error")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want it to wrap context.DeadlineExceeded", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("Do took %v — the call is not bound to the request context", elapsed)
	}
}

func TestOpenAI_Health(t *testing.T) {
	tests := []struct {
		name    string
		status  int
		wantErr bool
	}{
		// Any 4xx proves we reached the API and it answered, which is
		// the only thing the breaker cares about.
		{"ok", http.StatusOK, false},
		{"unauthorized is still reachable", http.StatusUnauthorized, false},
		{"server error", http.StatusInternalServerError, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/models" {
					t.Errorf("health path = %s, want /v1/models", r.URL.Path)
				}
				w.WriteHeader(tc.status)
			}))
			defer srv.Close()

			err := provider.NewOpenAIWithBaseURL("k", srv.URL).Health(context.Background())
			if (err != nil) != tc.wantErr {
				t.Errorf("Health = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}
