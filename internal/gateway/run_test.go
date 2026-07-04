package gateway_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/gateway"
	"github.com/kevinreber/llm-gateway/internal/provider"
)

// TestGateway_EndToEnd exercises the full path: HTTP inbound → gateway
// handler → Anthropic provider → mock upstream → response back to
// caller. Uses Run() with a real net.Listener so we cover the wiring
// in run.go, not just the handler.
func TestGateway_EndToEnd(t *testing.T) {
	// 1. Mock upstream Anthropic.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("upstream path = %s, want /v1/messages", r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("upstream x-api-key missing")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id": "msg_e2e",
			"type": "message",
			"role": "assistant",
			"content": [{"type":"text","text":"pong"}],
			"model": "claude-sonnet-4-6",
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 4, "output_tokens": 1}
		}`)
	}))
	defer upstream.Close()

	// 2. Bind an ephemeral port for the gateway to avoid conflicts.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close() // release; Run will re-bind. Small race, acceptable in tests.

	// 3. Start the gateway.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	runErr := make(chan error, 1)
	go func() {
		runErr <- gateway.Run(ctx, gateway.Config{
			Addr:             addr,
			ShutdownTimeout:  2 * time.Second,
			AnthropicAPIKey:  "test-key",
			AnthropicBaseURL: upstream.URL,
		})
	}()

	// 4. Wait for the gateway to be reachable.
	waitReady(t, "http://"+addr+"/healthz", 2*time.Second)

	// 5. Hit /v1/messages.
	reqBody, _ := json.Marshal(provider.Request{
		Model:    "claude-sonnet-4-6",
		Messages: []provider.Message{{Role: "user", Content: "ping"}},
	})
	resp, err := http.Post("http://"+addr+"/v1/messages", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	var got provider.Response
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Content != "pong" {
		t.Errorf("Content = %q, want %q", got.Content, "pong")
	}

	// 6. Graceful shutdown.
	cancel()
	select {
	case err := <-runErr:
		if err != nil {
			t.Errorf("Run returned error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Run did not shut down within 3s")
	}
}

func TestGateway_UnsupportedModel(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("upstream should NOT be called for unsupported model")
	}))
	defer upstream.Close()

	addr := reservePort(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		_ = gateway.Run(ctx, gateway.Config{
			Addr:             addr,
			ShutdownTimeout:  time.Second,
			AnthropicAPIKey:  "test-key",
			AnthropicBaseURL: upstream.URL,
		})
	}()
	waitReady(t, "http://"+addr+"/healthz", 2*time.Second)

	resp, err := http.Post("http://"+addr+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for unsupported model", resp.StatusCode)
	}
}

func TestGateway_UpstreamError_Surfaced(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"type":"rate_limit_error","message":"rate limited"}}`)
	}))
	defer upstream.Close()

	addr := reservePort(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		_ = gateway.Run(ctx, gateway.Config{
			Addr:             addr,
			ShutdownTimeout:  time.Second,
			AnthropicAPIKey:  "test-key",
			AnthropicBaseURL: upstream.URL,
		})
	}()
	waitReady(t, "http://"+addr+"/healthz", 2*time.Second)

	resp, err := http.Post("http://"+addr+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// 429 from upstream should surface as 429 to the caller.
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429", resp.StatusCode)
	}
}

func TestGateway_ForwardsRetryAfter(t *testing.T) {
	// Upstream 429 with Retry-After: gateway must forward the header
	// to the client so they know how long to back off.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "30")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"type":"rate_limit_error","message":"slow down"}}`)
	}))
	defer upstream.Close()

	addr := reservePort(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		_ = gateway.Run(ctx, gateway.Config{
			Addr:             addr,
			ShutdownTimeout:  time.Second,
			AnthropicAPIKey:  "test-key",
			AnthropicBaseURL: upstream.URL,
		})
	}()
	waitReady(t, "http://"+addr+"/healthz", 2*time.Second)

	resp, err := http.Post("http://"+addr+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429", resp.StatusCode)
	}
	if got := resp.Header.Get("Retry-After"); got != "30" {
		t.Errorf("Retry-After = %q, want %q", got, "30")
	}
}

func reservePort(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()
	return addr
}

func waitReady(t *testing.T, url string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("gateway not ready at %s within %s", url, timeout)
}
