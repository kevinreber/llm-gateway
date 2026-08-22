package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/config"
	"github.com/kevinreber/llm-gateway/internal/cost"
	"github.com/kevinreber/llm-gateway/internal/provider"
	"github.com/kevinreber/llm-gateway/internal/ratelimit"
)

// fakeProvider records what it was asked for and returns a canned reply.
//
// The mutex is load-bearing from Phase 3 on: one provider value is now
// shared across the goroutines of a chaos run, and the counters these
// tests assert on would otherwise be read while being written.
type fakeProvider struct {
	mu       sync.Mutex
	name     string
	supports func(string) bool
	gotModel string
	calls    int
	resp     *provider.Response
	err      error
	// errFor overrides err for specific models, so one provider can be
	// broken for the alias a chain starts at and healthy for the one it
	// ends at — the shape a real partial outage takes.
	errFor map[string]error
	// block makes Do hang until its context expires, which is the
	// failure mode a per-attempt deadline exists to survive.
	block time.Duration
	// doFunc, when set, replaces the canned behavior entirely.
	doFunc func(*provider.Request) (*provider.Response, error)
}

func (f *fakeProvider) Name() string { return f.name }

func (f *fakeProvider) Supports(model string) bool {
	if f.supports != nil {
		return f.supports(model)
	}
	return strings.HasPrefix(model, "claude-")
}

func (f *fakeProvider) Health(context.Context) error { return nil }

func (f *fakeProvider) Do(ctx context.Context, req *provider.Request) (*provider.Response, error) {
	f.mu.Lock()
	f.calls++
	f.gotModel = req.Model
	err := f.err
	if e, ok := f.errFor[req.Model]; ok {
		err = e
	}
	resp, block, doFunc := f.resp, f.block, f.doFunc
	f.mu.Unlock()

	if block > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(block):
		}
	}
	if doFunc != nil {
		return doFunc(req)
	}
	if err != nil {
		return nil, err
	}
	if resp != nil {
		return resp, nil
	}
	return &provider.Response{
		Content:    "ok",
		Model:      req.Model,
		StopReason: "end_turn",
		Usage:      provider.Usage{InputTokens: 1000, OutputTokens: 500},
	}, nil
}

// fakeLimiter returns a fixed verdict (or error) for every key.
type fakeLimiter struct {
	verdict ratelimit.Verdict
	err     error
	keys    []string
	limits  []ratelimit.Limit
}

func (f *fakeLimiter) Allow(_ context.Context, key string, l ratelimit.Limit) (ratelimit.Verdict, error) {
	f.keys = append(f.keys, key)
	f.limits = append(f.limits, l)
	return f.verdict, f.err
}

func (f *fakeLimiter) Close() error { return nil }

// recordingTracker captures cost events without a database.
type recordingTracker struct {
	mu     sync.Mutex
	events []cost.Event
}

func (r *recordingTracker) Track(e cost.Event) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
}

func (r *recordingTracker) all() []cost.Event {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]cost.Event, len(r.events))
	copy(out, r.events)
	return out
}

const aliasYAML = `
aliases:
  smart:    { provider: anthropic, model: claude-sonnet-5 }
  unmetered: { provider: anthropic, model: claude-haiku-4-5 }
  broken:   { provider: openai, model: gpt-4o }
  mismatch: { provider: anthropic, model: gpt-4o }

ratelimits:
  smart: { capacity: 10, refill_rate: 5 }
`

type harness struct {
	h        *handler
	provider *fakeProvider
	limiter  *fakeLimiter
	costs    *recordingTracker
}

func newHarness(t *testing.T, yaml string) *harness {
	t.Helper()

	cfg, err := config.Parse(strings.NewReader(yaml))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}

	fp := &fakeProvider{name: provider.AnthropicName}
	fl := &fakeLimiter{verdict: ratelimit.Verdict{Allowed: true}}
	rt := &recordingTracker{}

	return &harness{
		h: &handler{
			providers:     map[string]provider.Provider{provider.AnthropicName: fp},
			providerOrder: []string{provider.AnthropicName},
			cfg:           config.Static(cfg),
			limiter:       fl,
			costs:         rt,
			logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
		},
		provider: fp,
		limiter:  fl,
		costs:    rt,
	}
}

func TestResolve_DirectModelIsDeterministic(t *testing.T) {
	// Two providers both claiming the same model is the situation that
	// arrives with Ollama and a hosted mirror. Ranging the provider map
	// would pick a winner per request at random; the explicit order has
	// to win every time, or cost attribution and latency become
	// irreproducible in a way no test would catch.
	first := &fakeProvider{name: "first", supports: func(string) bool { return true }}
	second := &fakeProvider{name: "second", supports: func(string) bool { return true }}

	h := &handler{
		providers: map[string]provider.Provider{
			"first":  first,
			"second": second,
		},
		providerOrder: []string{"first", "second"},
		cfg:           config.Static(&config.Config{}),
		limiter:       ratelimit.AllowAll{},
		costs:         &recordingTracker{},
		logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	}

	for i := 0; i < 100; i++ {
		rt, err := h.resolve(h.cfg.Load(), "claude-sonnet-5")
		if err != nil {
			t.Fatalf("iteration %d: resolve: %v", i, err)
		}
		if rt.provider.Name() != "first" {
			t.Fatalf("iteration %d: resolved to %q, want the first provider in order",
				i, rt.provider.Name())
		}
	}
}

func (hn *harness) post(t *testing.T, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	rec := httptest.NewRecorder()
	hn.h.routes().ServeHTTP(rec, req)
	return rec
}

const userTurn = `"messages":[{"role":"user","content":"hi"}]`

func TestMessages_AliasResolvesToConcreteModel(t *testing.T) {
	hn := newHarness(t, aliasYAML)

	rec := hn.post(t, `{"model":"smart",`+userTurn+`}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	// The alias must be translated before the upstream call — sending
	// "smart" to Anthropic would be a 404 from their API.
	if hn.provider.gotModel != "claude-sonnet-5" {
		t.Errorf("upstream model = %q, want claude-sonnet-5", hn.provider.gotModel)
	}
	if got := rec.Header().Get("X-Gateway-Alias"); got != "smart" {
		t.Errorf("X-Gateway-Alias = %q, want smart", got)
	}
	if got := rec.Header().Get("X-Gateway-Model"); got != "claude-sonnet-5" {
		t.Errorf("X-Gateway-Model = %q, want claude-sonnet-5", got)
	}
	if got := rec.Header().Get("X-Gateway-Provider"); got != "anthropic" {
		t.Errorf("X-Gateway-Provider = %q, want anthropic", got)
	}
}

func TestMessages_DirectModelStillWorks(t *testing.T) {
	// Phase 1 behavior: a caller naming a concrete model bypasses the
	// alias table entirely and carries no alias label.
	hn := newHarness(t, aliasYAML)

	rec := hn.post(t, `{"model":"claude-haiku-4-5",`+userTurn+`}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	if hn.provider.gotModel != "claude-haiku-4-5" {
		t.Errorf("upstream model = %q, want claude-haiku-4-5", hn.provider.gotModel)
	}
	if got := rec.Header().Get("X-Gateway-Alias"); got != "" {
		t.Errorf("X-Gateway-Alias = %q, want empty for a direct model", got)
	}
	if n := len(hn.limiter.keys); n != 0 {
		t.Errorf("limiter consulted %d times for a direct model, want 0", n)
	}
}

func TestMessages_AliasToUnwiredProvider(t *testing.T) {
	// gateway.yaml names openai, which this build has no client for.
	hn := newHarness(t, aliasYAML)

	rec := hn.post(t, `{"model":"broken",`+userTurn+`}`)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body = %s", rec.Code, rec.Body)
	}
	if hn.provider.calls != 0 {
		t.Error("provider was called for an unresolvable alias")
	}
}

func TestMessages_AliasToUnsupportedModel(t *testing.T) {
	// A YAML edit pointing anthropic at gpt-4o must fail loudly here,
	// not become a confusing 404 from the upstream.
	hn := newHarness(t, aliasYAML)

	rec := hn.post(t, `{"model":"mismatch",`+userTurn+`}`)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body = %s", rec.Code, rec.Body)
	}
	if hn.provider.calls != 0 {
		t.Error("provider was called for a mismatched alias")
	}
}

func TestMessages_RateLimitDenied(t *testing.T) {
	hn := newHarness(t, aliasYAML)
	hn.limiter.verdict = ratelimit.Verdict{
		Allowed:    false,
		RetryAfter: 2500 * time.Millisecond,
	}

	rec := hn.post(t, `{"model":"smart",`+userTurn+`}`)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429; body = %s", rec.Code, rec.Body)
	}
	if hn.provider.calls != 0 {
		t.Error("provider was called despite a denied rate limit")
	}
	// 2.5s must round up: advertising 0 tells the client to hammer.
	if got := rec.Header().Get("Retry-After"); got != "3" {
		t.Errorf("Retry-After = %q, want 3", got)
	}
	if len(hn.costs.all()) != 0 {
		t.Error("a denied request produced a cost event")
	}
}

func TestMessages_RateLimitKeyAndPolicy(t *testing.T) {
	hn := newHarness(t, aliasYAML)

	hn.post(t, `{"model":"smart",`+userTurn+`}`)

	if len(hn.limiter.keys) != 1 {
		t.Fatalf("limiter consulted %d times, want 1", len(hn.limiter.keys))
	}
	// The key is namespaced so a future per-provider limit can share the
	// same bucketd cluster without colliding with an identically named
	// alias.
	if hn.limiter.keys[0] != "alias:smart" {
		t.Errorf("limit key = %q, want alias:smart", hn.limiter.keys[0])
	}
	want := ratelimit.Limit{Capacity: 10, RefillRate: 5}
	if hn.limiter.limits[0] != want {
		t.Errorf("limit policy = %+v, want %+v", hn.limiter.limits[0], want)
	}
}

func TestMessages_AliasWithoutLimitIsUnlimited(t *testing.T) {
	// An alias with no ratelimits entry must serve traffic, not fail
	// closed on a config omission.
	hn := newHarness(t, aliasYAML)
	hn.limiter.verdict = ratelimit.Verdict{Allowed: false}

	rec := hn.post(t, `{"model":"unmetered",`+userTurn+`}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body)
	}
	if len(hn.limiter.keys) != 0 {
		t.Errorf("limiter consulted for an alias with no configured limit")
	}
}

// blockingLimiter models a bucketd node that is slow rather than down.
type blockingLimiter struct{}

func (blockingLimiter) Allow(ctx context.Context, _ string, _ ratelimit.Limit) (ratelimit.Verdict, error) {
	<-ctx.Done()
	return ratelimit.Verdict{}, ctx.Err()
}

func (blockingLimiter) Close() error { return nil }

func TestMessages_SlowLimiterDoesNotStallTheRequest(t *testing.T) {
	// A degraded limiter must stop mattering, not become the slowest
	// thing in the request path. Without a bound on the Allow call this
	// request would hang for as long as the client was willing to wait.
	hn := newHarness(t, aliasYAML)
	hn.h.limiter = blockingLimiter{}

	start := time.Now()
	rec := hn.post(t, `{"model":"smart",`+userTurn+`}`)
	elapsed := time.Since(start)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (fail open); body = %s", rec.Code, rec.Body)
	}
	if hn.provider.calls != 1 {
		t.Errorf("provider called %d times, want 1", hn.provider.calls)
	}
	if elapsed > 3*time.Second {
		t.Errorf("request took %s — the limiter call is unbounded", elapsed)
	}
}

func TestMessages_LimiterErrorFailsOpen(t *testing.T) {
	// A bucketd outage degrades enforcement; it must not take the
	// gateway's traffic down with it.
	hn := newHarness(t, aliasYAML)
	hn.limiter.err = errors.New("bucketd unreachable")

	rec := hn.post(t, `{"model":"smart",`+userTurn+`}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body)
	}
	if hn.provider.calls != 1 {
		t.Errorf("provider called %d times, want 1", hn.provider.calls)
	}
}

func TestMessages_TracksCost(t *testing.T) {
	hn := newHarness(t, aliasYAML)

	rec := hn.post(t, `{"model":"smart",`+userTurn+`}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}

	events := hn.costs.all()
	if len(events) != 1 {
		t.Fatalf("recorded %d cost events, want 1", len(events))
	}
	e := events[0]
	if e.Provider != "anthropic" {
		t.Errorf("Provider = %q, want anthropic", e.Provider)
	}
	if e.Model != "claude-sonnet-5" {
		t.Errorf("Model = %q, want claude-sonnet-5", e.Model)
	}
	if e.Alias != "smart" {
		t.Errorf("Alias = %q, want smart", e.Alias)
	}
	if e.InputTokens != 1000 || e.OutputTokens != 500 {
		t.Errorf("tokens = %d/%d, want 1000/500", e.InputTokens, e.OutputTokens)
	}
	// Sonnet 5: 1000 in at $3/MTok + 500 out at $15/MTok
	// = $0.003 + $0.0075 = $0.0105 = 1.05 cents.
	if math.Abs(e.CostCents-1.05) > 1e-9 {
		t.Errorf("CostCents = %v, want 1.05", e.CostCents)
	}
	if e.TS.IsZero() {
		t.Error("TS is zero")
	}
}

func TestMessages_BillsTheModelThatServed(t *testing.T) {
	// Anthropic echoes back the model that actually handled the
	// request. Billing follows that, not what the client asked for.
	hn := newHarness(t, aliasYAML)
	hn.provider.resp = &provider.Response{
		Content: "ok",
		Model:   "claude-haiku-4-5",
		Usage:   provider.Usage{InputTokens: 1000, OutputTokens: 500},
	}

	hn.post(t, `{"model":"smart",`+userTurn+`}`)

	events := hn.costs.all()
	if len(events) != 1 {
		t.Fatalf("recorded %d cost events, want 1", len(events))
	}
	if events[0].Model != "claude-haiku-4-5" {
		t.Errorf("Model = %q, want the model the provider reported", events[0].Model)
	}
	// Haiku rates, not Sonnet's: 1000 in at $1 + 500 out at $5 = 0.35c.
	if math.Abs(events[0].CostCents-0.35) > 1e-9 {
		t.Errorf("CostCents = %v, want 0.35 (Haiku rates)", events[0].CostCents)
	}
}

func TestMessages_UpstreamRetryAfterRoundsUp(t *testing.T) {
	// A sub-second upstream Retry-After must not truncate to 0 — that
	// tells an already-throttled client to retry immediately, which is
	// the opposite of what the upstream asked for.
	hn := newHarness(t, aliasYAML)
	hn.provider.err = &provider.APIError{
		Provider: "anthropic", Status: http.StatusTooManyRequests,
		Type: "rate_limit_error", Message: "slow down",
		RetryAfter: 400 * time.Millisecond,
	}

	rec := hn.post(t, `{"model":"smart",`+userTurn+`}`)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want 1 (rounded up from 400ms)", got)
	}
}

func TestMessages_ProviderErrorIsNotBilled(t *testing.T) {
	hn := newHarness(t, aliasYAML)
	hn.provider.err = &provider.APIError{
		Provider: "anthropic", Status: http.StatusServiceUnavailable,
		Type: "overloaded_error", Message: "overloaded",
	}

	rec := hn.post(t, `{"model":"smart",`+userTurn+`}`)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
	if n := len(hn.costs.all()); n != 0 {
		t.Errorf("recorded %d cost events for a failed request, want 0", n)
	}
}

func TestMessages_UnpricedModelStillRecorded(t *testing.T) {
	// A model missing from the pricing table must still produce a usage
	// row — losing the token counts too would hide the gap entirely.
	hn := newHarness(t, "")
	hn.provider.supports = func(string) bool { return true }
	hn.provider.resp = &provider.Response{
		Content: "ok",
		Model:   "claude-future-9",
		Usage:   provider.Usage{InputTokens: 42, OutputTokens: 7},
	}

	rec := hn.post(t, `{"model":"claude-future-9",`+userTurn+`}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}

	events := hn.costs.all()
	if len(events) != 1 {
		t.Fatalf("recorded %d cost events, want 1", len(events))
	}
	if events[0].CostCents != 0 {
		t.Errorf("CostCents = %v, want 0 for an unpriced model", events[0].CostCents)
	}
	if events[0].InputTokens != 42 || events[0].OutputTokens != 7 {
		t.Errorf("tokens = %d/%d, want 42/7", events[0].InputTokens, events[0].OutputTokens)
	}
}

func TestMessages_RejectsMalformedRequests(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"not json", `{`},
		{"no model", `{` + userTurn + `}`},
		{"no messages", `{"model":"smart"}`},
		{"unknown model", `{"model":"gpt-4o",` + userTurn + `}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hn := newHarness(t, aliasYAML)
			rec := hn.post(t, tt.body)
			if rec.Code != http.StatusBadRequest {
				t.Errorf("status = %d, want 400; body = %s", rec.Code, rec.Body)
			}
			if hn.provider.calls != 0 {
				t.Error("provider was called for a malformed request")
			}
			var body map[string]any
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
				t.Errorf("error body is not JSON: %v", err)
			}
			if _, ok := body["error"]; !ok {
				t.Errorf("error body missing 'error' key: %s", rec.Body)
			}
		})
	}
}
