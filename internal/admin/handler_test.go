package admin

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/config"
	"github.com/kevinreber/llm-gateway/internal/observe"
	"github.com/kevinreber/llm-gateway/internal/provider"
	"github.com/kevinreber/llm-gateway/internal/resilience"
)

const adminYAML = `
aliases:
  smart: { provider: anthropic, model: claude-sonnet-5 }
  fast:  { provider: anthropic, model: claude-haiku-4-5 }
ratelimits:
  smart: { capacity: 10, refill_rate: 5 }
fallback:
  smart: [fast]
`

// stubProvider serves a fixed model prefix and always succeeds.
type stubProvider struct {
	name   string
	prefix string
}

func (s *stubProvider) Name() string                 { return s.name }
func (s *stubProvider) Supports(m string) bool       { return strings.HasPrefix(m, s.prefix) }
func (s *stubProvider) Health(context.Context) error { return nil }
func (s *stubProvider) Do(context.Context, *provider.Request) (*provider.Response, error) {
	return &provider.Response{Content: "ok"}, nil
}

type stubDrops struct{ n atomic.Int64 }

func (s *stubDrops) Dropped() int64 { return s.n.Load() }

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "gateway.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	return path
}

// newHandler builds an admin handler over a real resilience-wrapped
// provider, so breaker assertions exercise the real state machine.
func newHandler(t *testing.T, path string) (*Handler, *resilience.Breaker) {
	t.Helper()

	var cfg *config.Config
	var store *config.Store
	if path == "" {
		parsed, err := config.Parse(strings.NewReader(adminYAML))
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		cfg = parsed
		store = config.Static(cfg)
	} else {
		loaded, err := config.Load(path)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		store = config.NewStore(path, loaded)
	}

	wrapped := resilience.Wrap(&stubProvider{name: "anthropic", prefix: "claude-"},
		resilience.Options{FailureThreshold: 1, RecoveryTimeout: time.Minute})

	return &Handler{
		Store:     store,
		Providers: map[string]provider.Provider{"anthropic": wrapped},
		Order:     []string{"anthropic"},
		Logger:    slog.New(slog.NewTextHandler(io.Discard, nil)),
	}, wrapped.Breaker()
}

func do(t *testing.T, h *Handler, method, path string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	h.Routes().ServeHTTP(rec, httptest.NewRequest(method, path, nil))
	return rec
}

func TestAliases_ReportsTheLiveConfigAndItsSource(t *testing.T) {
	// The first question when routing is not what you expected is
	// whether the file you have been editing is the one in use.
	path := writeConfig(t, adminYAML)
	h, _ := newHandler(t, path)

	rec := do(t, h, http.MethodGet, "/admin/aliases")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	var got aliasesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Path != path {
		t.Errorf("path = %q, want %q", got.Path, path)
	}
	if !got.Reloadable {
		t.Error("a file-backed store should report reloadable")
	}
	if got.Aliases["smart"].Model != "claude-sonnet-5" {
		t.Errorf("smart = %+v, want the configured model", got.Aliases["smart"])
	}
	if len(got.Fallback["smart"]) != 1 {
		t.Errorf("fallback = %v, want one entry", got.Fallback["smart"])
	}
}

func TestAliases_StaticStoreReportsNotReloadable(t *testing.T) {
	h, _ := newHandler(t, "")

	var got aliasesResponse
	if err := json.Unmarshal(do(t, h, http.MethodGet, "/admin/aliases").Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Reloadable || got.Path != "" {
		t.Errorf("static store reported path=%q reloadable=%v", got.Path, got.Reloadable)
	}
}

func TestStats_ReportsBreakerState(t *testing.T) {
	h, brk := newHandler(t, "")

	var before statsResponse
	if err := json.Unmarshal(do(t, h, http.MethodGet, "/admin/stats").Body.Bytes(), &before); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(before.Providers) != 1 {
		t.Fatalf("providers = %d, want 1", len(before.Providers))
	}
	if before.Providers[0].Breaker != "closed" || !before.Providers[0].Healthy {
		t.Errorf("initial = %+v, want closed and healthy", before.Providers[0])
	}

	brk.RecordFailure() // threshold is 1

	var after statsResponse
	if err := json.Unmarshal(do(t, h, http.MethodGet, "/admin/stats").Body.Bytes(), &after); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if after.Providers[0].Breaker != "open" {
		t.Errorf("breaker = %q, want open", after.Providers[0].Breaker)
	}
	if after.Providers[0].Healthy {
		t.Error("an open breaker must not report healthy")
	}
}

func TestStats_RequestCountsComeFromTheMetricRegistry(t *testing.T) {
	// Reading them back from the registry rather than keeping a second
	// tally is what stops /admin/stats and /metrics from disagreeing.
	h, _ := newHandler(t, "")
	h.Order = []string{"stats-registry-provider"}
	h.Providers = map[string]provider.Provider{
		"stats-registry-provider": &stubProvider{name: "stats-registry-provider", prefix: "x-"},
	}

	observe.RecordRequest("smart", "stats-registry-provider", observe.ResultOK)
	observe.RecordRequest("smart", "stats-registry-provider", observe.ResultOK)
	observe.RecordRequest("smart", "stats-registry-provider", observe.ResultProviderError)

	var got statsResponse
	if err := json.Unmarshal(do(t, h, http.MethodGet, "/admin/stats").Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	reqs := got.Providers[0].Requests
	if reqs[observe.ResultOK] != 2 {
		t.Errorf("ok = %v, want 2", reqs[observe.ResultOK])
	}
	if reqs[observe.ResultProviderError] != 1 {
		t.Errorf("provider_error = %v, want 1", reqs[observe.ResultProviderError])
	}
}

func TestStats_ReportsDroppedCostEvents(t *testing.T) {
	h, _ := newHandler(t, "")
	drops := &stubDrops{}
	drops.n.Store(7)
	h.Costs = drops

	var got statsResponse
	if err := json.Unmarshal(do(t, h, http.MethodGet, "/admin/stats").Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.CostEventsDropped != 7 {
		t.Errorf("dropped = %d, want 7", got.CostEventsDropped)
	}
}

func TestReload_PicksUpEdits(t *testing.T) {
	path := writeConfig(t, adminYAML)
	h, _ := newHandler(t, path)

	if err := os.WriteFile(path, []byte(`
aliases:
  smart: { provider: anthropic, model: claude-opus-5 }
`), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}

	rec := do(t, h, http.MethodPost, "/admin/reload")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body)
	}

	var got reloadResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Status != "reloaded" || got.Aliases != 1 {
		t.Errorf("response = %+v, want reloaded with 1 alias", got)
	}
	if m := h.Store.Load().Aliases["smart"].Model; m != "claude-opus-5" {
		t.Errorf("live config smart = %q, want the edited value", m)
	}
}

func TestReload_InvalidConfigIsRejectedAndChangesNothing(t *testing.T) {
	path := writeConfig(t, adminYAML)
	h, _ := newHandler(t, path)

	if err := os.WriteFile(path, []byte("aliases:\n  broken: { provider: anthropic }\n"), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}

	rec := do(t, h, http.MethodPost, "/admin/reload")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	if m := h.Store.Load().Aliases["smart"].Model; m != "claude-sonnet-5" {
		t.Errorf("live config changed to %q after a rejected reload", m)
	}
}

func TestReload_StaticStoreReportsConflict(t *testing.T) {
	h, _ := newHandler(t, "")

	rec := do(t, h, http.MethodPost, "/admin/reload")
	if rec.Code != http.StatusConflict {
		t.Errorf("status = %d, want 409", rec.Code)
	}
}

func TestReload_NamesAliasesThisBinaryCannotServe(t *testing.T) {
	// Telling the operator who just made the edit beats a request
	// discovering it at 3am. Not an error: the same config is meant to
	// deploy where a provider key is set and where it is not.
	path := writeConfig(t, adminYAML)
	h, _ := newHandler(t, path)

	if err := os.WriteFile(path, []byte(`
aliases:
  smart:  { provider: anthropic, model: claude-sonnet-5 }
  absent: { provider: gemini,    model: gemini-2.0-flash }
  wrong:  { provider: anthropic, model: gpt-4o }
`), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}

	rec := do(t, h, http.MethodPost, "/admin/reload")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body)
	}

	var got reloadResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	want := []string{"absent", "wrong"}
	if len(got.Unroutable) != len(want) {
		t.Fatalf("unroutable = %v, want %v", got.Unroutable, want)
	}
	for i, w := range want {
		if got.Unroutable[i] != w {
			t.Errorf("unroutable[%d] = %q, want %q (sorted)", i, got.Unroutable[i], w)
		}
	}
}

func TestRoutes_MetricsIsServedHere(t *testing.T) {
	h, _ := newHandler(t, "")

	rec := do(t, h, http.MethodGet, "/metrics")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "llm_gateway_") {
		t.Error("exposition did not contain gateway metrics")
	}
}

func TestRoutes_WrongMethodIsRejected(t *testing.T) {
	// Method-scoped routing means a GET to reload must not mutate
	// anything, which matters for an endpoint a browser could reach.
	h, _ := newHandler(t, "")

	if rec := do(t, h, http.MethodGet, "/admin/reload"); rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET /admin/reload = %d, want 405", rec.Code)
	}
}
