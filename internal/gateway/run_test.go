package gateway_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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

// TestGateway_AliasEndToEnd covers the wiring Run() does that the
// handler tests mock out: reading gateway.yaml off disk, translating the
// alias before the upstream call, and running the cost writer against
// the no-database sink without breaking the request path.
func TestGateway_AliasEndToEnd(t *testing.T) {
	gotModel := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotModel <- body.Model

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id": "msg_alias",
			"type": "message",
			"role": "assistant",
			"content": [{"type":"text","text":"routed"}],
			"model": "claude-sonnet-5",
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 1000, "output_tokens": 500}
		}`)
	}))
	defer upstream.Close()

	configPath := filepath.Join(t.TempDir(), "gateway.yaml")
	if err := os.WriteFile(configPath, []byte(
		"aliases:\n  smart: { provider: anthropic, model: claude-sonnet-5 }\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	addr := reservePort(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	runErr := make(chan error, 1)
	go func() {
		runErr <- gateway.Run(ctx, gateway.Config{
			Addr:             addr,
			ShutdownTimeout:  2 * time.Second,
			AnthropicAPIKey:  "test-key",
			AnthropicBaseURL: upstream.URL,
			ConfigPath:       configPath,
			// No BUCKETD_ADDRS and no DATABASE_URL: the gateway must
			// serve normally with both dependencies absent.
		})
	}()
	waitReady(t, "http://"+addr+"/healthz", 2*time.Second)

	resp, err := http.Post("http://"+addr+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"smart","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	if got := resp.Header.Get("X-Gateway-Alias"); got != "smart" {
		t.Errorf("X-Gateway-Alias = %q, want smart", got)
	}
	if got := resp.Header.Get("X-Gateway-Model"); got != "claude-sonnet-5" {
		t.Errorf("X-Gateway-Model = %q, want claude-sonnet-5", got)
	}

	select {
	case m := <-gotModel:
		if m != "claude-sonnet-5" {
			t.Errorf("upstream received model %q, want claude-sonnet-5", m)
		}
	default:
		t.Fatal("upstream was never called")
	}

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

// TestGateway_MissingExplicitConfigFails guards the operator-intent
// rule: an explicitly configured path that doesn't exist must abort
// startup rather than silently serving with no rate limits.
func TestGateway_MissingExplicitConfigFails(t *testing.T) {
	err := gateway.Run(context.Background(), gateway.Config{
		Addr:            reservePort(t),
		ShutdownTimeout: time.Second,
		AnthropicAPIKey: "test-key",
		ConfigPath:      filepath.Join(t.TempDir(), "does-not-exist.yaml"),
	})
	if err == nil {
		t.Fatal("Run succeeded with a missing explicit config path, want an error")
	}
	if !strings.Contains(err.Error(), "does-not-exist.yaml") {
		t.Errorf("err = %q, want it to name the missing path", err)
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

func TestGateway_AdminListenerServesOnItsOwnPort(t *testing.T) {
	// The separation is the security boundary: /admin/reload repoints
	// live traffic and carries no authentication, so it must not be
	// reachable on the port that serves clients.
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "gateway.yaml")
	if err := os.WriteFile(cfgPath, []byte(
		"aliases:\n  smart: { provider: anthropic, model: claude-sonnet-5 }\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	addr, adminAddr := reservePort(t), reservePort(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		_ = gateway.Run(ctx, gateway.Config{
			Addr:            addr,
			AdminAddr:       adminAddr,
			ConfigPath:      cfgPath,
			AnthropicAPIKey: "test-key",
			ShutdownTimeout: time.Second,
		})
	}()
	waitReady(t, "http://"+addr+"/healthz", 3*time.Second)
	waitReady(t, "http://"+adminAddr+"/admin/aliases", 3*time.Second)

	// The operational surface answers on the admin port...
	for _, path := range []string{"/admin/aliases", "/admin/stats", "/metrics"} {
		resp, err := http.Get("http://" + adminAddr + path)
		if err != nil {
			t.Fatalf("GET admin %s: %v", path, err)
		}
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("admin %s = %d, want 200", path, resp.StatusCode)
		}
	}

	// ...and not on the request port.
	for _, path := range []string{"/admin/aliases", "/admin/stats", "/metrics"} {
		resp, err := http.Get("http://" + addr + path)
		if err != nil {
			t.Fatalf("GET request-port %s: %v", path, err)
		}
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("request port served %s with %d; it must not be there", path, resp.StatusCode)
		}
	}
}

func TestGateway_ReloadThroughAdminRedirectsLiveTraffic(t *testing.T) {
	// The whole point of the atomic store: editing the file and posting
	// a reload changes where requests go, with no restart and no
	// dropped connection.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		model := "claude-sonnet-5"
		if strings.Contains(string(body), "claude-opus-5") {
			model = "claude-opus-5"
		}
		_, _ = io.WriteString(w, `{"content":[{"type":"text","text":"hi"}],"model":"`+model+
			`","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer upstream.Close()

	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "gateway.yaml")
	if err := os.WriteFile(cfgPath, []byte(
		"aliases:\n  smart: { provider: anthropic, model: claude-sonnet-5 }\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	addr, adminAddr := reservePort(t), reservePort(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		_ = gateway.Run(ctx, gateway.Config{
			Addr:             addr,
			AdminAddr:        adminAddr,
			ConfigPath:       cfgPath,
			AnthropicAPIKey:  "test-key",
			AnthropicBaseURL: upstream.URL,
			ShutdownTimeout:  time.Second,
		})
	}()
	waitReady(t, "http://"+addr+"/healthz", 3*time.Second)

	post := func() string {
		t.Helper()
		resp, err := http.Post("http://"+addr+"/v1/messages", "application/json",
			strings.NewReader(`{"model":"smart","messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatalf("post: %v", err)
		}
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(resp.Body)
			t.Fatalf("post = %d: %s", resp.StatusCode, b)
		}
		return resp.Header.Get("X-Gateway-Model")
	}

	if got := post(); got != "claude-sonnet-5" {
		t.Fatalf("before reload, served model = %q", got)
	}

	if err := os.WriteFile(cfgPath, []byte(
		"aliases:\n  smart: { provider: anthropic, model: claude-opus-5 }\n"), 0o600); err != nil {
		t.Fatalf("rewrite config: %v", err)
	}
	resp, err := http.Post("http://"+adminAddr+"/admin/reload", "application/json", nil)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("reload = %d, want 200", resp.StatusCode)
	}

	if got := post(); got != "claude-opus-5" {
		t.Errorf("after reload, served model = %q, want claude-opus-5", got)
	}
}
