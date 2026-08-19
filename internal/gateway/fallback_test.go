package gateway

import (
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/config"
	"github.com/kevinreber/llm-gateway/internal/provider"
	"github.com/kevinreber/llm-gateway/internal/ratelimit"
	"github.com/kevinreber/llm-gateway/internal/resilience"
)

// fallbackYAML wires two vendors so a chain has somewhere real to go.
// `smart` prefers Anthropic and falls back to OpenAI then to a cheaper
// Anthropic model; `lonely` has no chain at all.
const fallbackYAML = `
aliases:
  smart:     { provider: anthropic, model: claude-sonnet-5 }
  fast:      { provider: anthropic, model: claude-haiku-4-5 }
  smart-alt: { provider: openai,    model: gpt-4o }
  lonely:    { provider: anthropic, model: claude-opus-5 }
  orphan:    { provider: gemini,    model: gemini-2.0-flash }

ratelimits:
  smart: { capacity: 10, refill_rate: 5 }

fallback:
  smart:  [smart-alt, fast]
  lonely: []
  fast:   [orphan]
`

// fallbackHarness gives each provider its own fake and wraps both in the
// real resilience layer, so these tests exercise the same composition
// production runs rather than a hand-rolled stand-in.
type fallbackHarness struct {
	h      *handler
	anth   *fakeProvider
	openai *fakeProvider
	costs  *recordingTracker
	brk    map[string]*resilience.Breaker
}

func newFallbackHarness(t *testing.T, opts resilience.Options) *fallbackHarness {
	t.Helper()

	cfg, err := config.Parse(strings.NewReader(fallbackYAML))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}

	anth := &fakeProvider{name: provider.AnthropicName}
	oai := &fakeProvider{
		name:     provider.OpenAIName,
		supports: func(m string) bool { return strings.HasPrefix(m, "gpt-") },
	}

	wrappedAnth := resilience.Wrap(anth, opts)
	wrappedOAI := resilience.Wrap(oai, opts)
	costs := &recordingTracker{}

	return &fallbackHarness{
		h: &handler{
			providers: map[string]provider.Provider{
				provider.AnthropicName: wrappedAnth,
				provider.OpenAIName:    wrappedOAI,
			},
			providerOrder: []string{provider.AnthropicName, provider.OpenAIName},
			cfg:           cfg,
			limiter:       ratelimit.AllowAll{},
			costs:         costs,
			logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
			callBudget:    5 * time.Second,
		},
		anth:   anth,
		openai: oai,
		costs:  costs,
		brk: map[string]*resilience.Breaker{
			provider.AnthropicName: wrappedAnth.Breaker(),
			provider.OpenAIName:    wrappedOAI.Breaker(),
		},
	}
}

func (hn *fallbackHarness) post(t *testing.T, model string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"`+model+`",`+userTurn+`}`))
	rec := httptest.NewRecorder()
	hn.h.routes().ServeHTTP(rec, req)
	return rec
}

// noRetry keeps the retry loop out of tests that are about the chain.
func noRetry() resilience.Options {
	return resilience.Options{
		MaxAttempts:      1,
		FailureThreshold: 100,
		AttemptTimeout:   time.Second,
		Budget:           2 * time.Second,
	}
}

func TestFallback_SecondProviderServesWhenTheFirstIs503(t *testing.T) {
	// The Week 3 milestone: Anthropic goes down, the request lands on
	// OpenAI, and the client gets a 200.
	hn := newFallbackHarness(t, noRetry())
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}
	hn.openai.resp = &provider.Response{
		Content: "from openai",
		Model:   "gpt-4o-2024-08-06",
		Usage:   provider.Usage{InputTokens: 400_000, OutputTokens: 100_000},
	}

	rec := hn.post(t, "smart")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body)
	}
	if hn.anth.calls != 1 {
		t.Errorf("anthropic calls = %d, want 1", hn.anth.calls)
	}
	if hn.openai.calls != 1 {
		t.Errorf("openai calls = %d, want 1", hn.openai.calls)
	}
	// The fallback alias's model, not the primary's: sending
	// claude-sonnet-5 to OpenAI would be a 404 from their API.
	if hn.openai.gotModel != "gpt-4o" {
		t.Errorf("openai got model %q, want gpt-4o", hn.openai.gotModel)
	}
}

func TestFallback_HeadersReportWhatActuallyServed(t *testing.T) {
	// Without accurate headers an alias is a black box: the body of a
	// fallback response looks exactly like the body of a normal one.
	hn := newFallbackHarness(t, noRetry())
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}
	hn.openai.resp = &provider.Response{Content: "x", Model: "gpt-4o-2024-08-06"}

	rec := hn.post(t, "smart")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body)
	}

	if got := rec.Header().Get("X-Gateway-Provider"); got != "openai" {
		t.Errorf("X-Gateway-Provider = %q, want openai", got)
	}
	if got := rec.Header().Get("X-Gateway-Model"); got != "gpt-4o" {
		t.Errorf("X-Gateway-Model = %q, want gpt-4o", got)
	}
	// The alias stays what the client asked for; the fallback header is
	// what says the request did not go where that alias points.
	if got := rec.Header().Get("X-Gateway-Alias"); got != "smart" {
		t.Errorf("X-Gateway-Alias = %q, want smart", got)
	}
	if got := rec.Header().Get("X-Gateway-Fallback"); got != "smart-alt" {
		t.Errorf("X-Gateway-Fallback = %q, want smart-alt", got)
	}
}

func TestFallback_NoFallbackHeaderOnTheHappyPath(t *testing.T) {
	hn := newFallbackHarness(t, noRetry())

	rec := hn.post(t, "smart")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body)
	}
	if got := rec.Header().Get("X-Gateway-Fallback"); got != "" {
		t.Errorf("X-Gateway-Fallback = %q, want empty when the primary served", got)
	}
	if hn.openai.calls != 0 {
		t.Errorf("openai called %d times on the happy path, want 0", hn.openai.calls)
	}
}

func TestFallback_CostIsAttributedToTheProviderThatServed(t *testing.T) {
	// The bug this prevents is silent and expensive: billing a
	// fallback's tokens to the provider that refused the request.
	hn := newFallbackHarness(t, noRetry())
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}
	hn.openai.resp = &provider.Response{
		Content: "x",
		Model:   "gpt-4o-2024-08-06",
		Usage:   provider.Usage{InputTokens: 400_000, OutputTokens: 100_000},
	}

	if rec := hn.post(t, "smart"); rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}

	events := hn.costs.all()
	if len(events) != 1 {
		t.Fatalf("recorded %d cost events, want 1", len(events))
	}
	e := events[0]
	if e.Provider != "openai" {
		t.Errorf("Provider = %q, want openai — cost must follow the provider that served", e.Provider)
	}
	if e.Model != "gpt-4o-2024-08-06" {
		t.Errorf("Model = %q, want the model the response reported", e.Model)
	}
	// The alias is the unit of accounting and does not move: `smart`
	// traffic is `smart` traffic wherever it landed.
	if e.Alias != "smart" {
		t.Errorf("Alias = %q, want smart", e.Alias)
	}
	// gpt-4o at $2.50/$10: 400k in + 100k out = $1.00 + $1.00 = 200c.
	// Billing this at Sonnet's rate would be 220c, and at zero if the
	// pricing table had been left Anthropic-only.
	if e.CostCents < 199.9 || e.CostCents > 200.1 {
		t.Errorf("CostCents = %v, want ~200 (gpt-4o rates)", e.CostCents)
	}
}

func TestFallback_ThirdHopWhenTwoFail(t *testing.T) {
	// The chain is ordered, and being ordered has to mean something:
	// smart -> smart-alt -> fast, with fast on Anthropic again.
	hn := newFallbackHarness(t, noRetry())
	hn.anth.errFor = map[string]error{
		"claude-sonnet-5": &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"},
	}
	hn.openai.err = &provider.APIError{Provider: "openai", Status: 500, Message: "boom"}

	rec := hn.post(t, "smart")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body)
	}
	if got := rec.Header().Get("X-Gateway-Fallback"); got != "fast" {
		t.Errorf("X-Gateway-Fallback = %q, want fast", got)
	}
	if got := rec.Header().Get("X-Gateway-Model"); got != "claude-haiku-4-5" {
		t.Errorf("X-Gateway-Model = %q, want claude-haiku-4-5", got)
	}
	if hn.anth.calls != 2 {
		t.Errorf("anthropic calls = %d, want 2 (sonnet then haiku)", hn.anth.calls)
	}
}

func TestFallback_ClientErrorsDoNotWalkTheChain(t *testing.T) {
	// A malformed prompt fails the same way at every vendor. Falling
	// back would turn one honest 400 into a tour of the whole chain
	// before returning the same answer, slower.
	hn := newFallbackHarness(t, noRetry())
	hn.anth.err = &provider.APIError{
		Provider: "anthropic",
		Status:   400,
		Type:     "invalid_request_error",
		Message:  "messages: unexpected role",
	}

	rec := hn.post(t, "smart")

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body = %s", rec.Code, rec.Body)
	}
	if hn.openai.calls != 0 {
		t.Errorf("openai called %d times after a 400, want 0", hn.openai.calls)
	}
}

func TestFallback_ExhaustedChainReportsThePrimaryFailure(t *testing.T) {
	// The client asked for `smart`. Why *smart* could not be served is
	// the answer to their question; the other hops are in the log.
	hn := newFallbackHarness(t, noRetry())
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Type: "overloaded_error", Message: "primary is down"}
	hn.openai.err = &provider.APIError{Provider: "openai", Status: 500, Type: "server_error", Message: "secondary is down"}

	rec := hn.post(t, "smart")

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body = %s", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "primary is down") {
		t.Errorf("body = %s, want the primary's error", rec.Body)
	}
	if hn.anth.calls != 2 || hn.openai.calls != 1 {
		t.Errorf("calls = anthropic %d / openai %d, want 2 / 1 (the whole chain was tried)",
			hn.anth.calls, hn.openai.calls)
	}
}

func TestFallback_AliasWithNoChainFailsDirectly(t *testing.T) {
	hn := newFallbackHarness(t, noRetry())
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}

	rec := hn.post(t, "lonely")

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body = %s", rec.Code, rec.Body)
	}
	if hn.openai.calls != 0 {
		t.Errorf("openai called %d times for an alias with no chain, want 0", hn.openai.calls)
	}
}

func TestFallback_DirectModelNamesGetNoFallback(t *testing.T) {
	// A caller naming claude-sonnet-5 outright asked for that model.
	// Answering with a different one from a different vendor would be
	// the gateway lying about what it did.
	hn := newFallbackHarness(t, noRetry())
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}

	rec := hn.post(t, "claude-sonnet-5")

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body = %s", rec.Code, rec.Body)
	}
	if hn.openai.calls != 0 {
		t.Errorf("openai called %d times for a direct model name, want 0", hn.openai.calls)
	}
}

func TestFallback_UnroutableEntriesAreSkippedNotFatal(t *testing.T) {
	// `fast` falls back to `orphan`, which names a provider this build
	// has no client for. A chain with a dead entry must still serve from
	// the entries that are alive — here there are none after it, so the
	// primary's own error is what comes back, but the request must not
	// blow up on the dead name.
	hn := newFallbackHarness(t, noRetry())
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}

	rec := hn.post(t, "fast")

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body = %s", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "overloaded") {
		t.Errorf("body = %s, want the primary's error", rec.Body)
	}
}

func TestFallback_OpenBreakerSkipsStraightToTheNextProvider(t *testing.T) {
	// Once the breaker is open the primary is not called at all — that
	// is the whole point of having one. The request still succeeds,
	// because the chain is what turns "we refuse to call Anthropic" into
	// an answer rather than an outage.
	opts := noRetry()
	opts.FailureThreshold = 2
	opts.RecoveryTimeout = time.Hour
	hn := newFallbackHarness(t, opts)
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}
	hn.openai.resp = &provider.Response{Content: "x", Model: "gpt-4o"}

	// Two requests trip it: each tries sonnet, fails, then falls over to
	// OpenAI and succeeds.
	for i := range 2 {
		if rec := hn.post(t, "smart"); rec.Code != http.StatusOK {
			t.Fatalf("request %d: status = %d, want 200", i, rec.Code)
		}
	}
	if got := hn.brk[provider.AnthropicName].State(); got != resilience.StateOpen {
		t.Fatalf("anthropic breaker = %v, want open", got)
	}

	before := hn.anth.calls
	rec := hn.post(t, "smart")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body)
	}
	if hn.anth.calls != before {
		t.Errorf("anthropic called %d more times while its breaker was open, want 0", hn.anth.calls-before)
	}
	if got := rec.Header().Get("X-Gateway-Provider"); got != "openai" {
		t.Errorf("X-Gateway-Provider = %q, want openai", got)
	}
}

func TestFallback_EveryProviderOpenReturns503WithRetryAfter(t *testing.T) {
	// Telling a client to come back at an unspecified time is barely
	// better than telling it nothing, and the breaker knows exactly when
	// it will next admit a probe.
	opts := noRetry()
	opts.FailureThreshold = 1
	opts.RecoveryTimeout = 45 * time.Second
	hn := newFallbackHarness(t, opts)
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}
	hn.openai.err = &provider.APIError{Provider: "openai", Status: 500, Message: "boom"}

	// First request trips both breakers walking the chain.
	if rec := hn.post(t, "smart"); rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("priming request: status = %d, want 503", rec.Code)
	}
	if got := hn.brk[provider.AnthropicName].State(); got != resilience.StateOpen {
		t.Fatalf("anthropic breaker = %v, want open", got)
	}

	callsBefore := hn.anth.calls + hn.openai.calls
	rec := hn.post(t, "smart")

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body = %s", rec.Code, rec.Body)
	}
	if got := hn.anth.calls + hn.openai.calls; got != callsBefore {
		t.Errorf("%d upstream calls made with every breaker open, want 0", got-callsBefore)
	}
	if got := rec.Header().Get("Retry-After"); got != "45" {
		t.Errorf("Retry-After = %q, want 45", got)
	}
	if !strings.Contains(rec.Body.String(), "circuit_open") {
		t.Errorf("body = %s, want a circuit_open error type", rec.Body)
	}
}

func TestFallback_RateLimitIsChargedOnceToTheRequestedAlias(t *testing.T) {
	// One request, one token, from the name the caller used. Charging
	// the fallback alias too would bill a single request twice and add a
	// limiter round-trip to the path that is already the degraded one.
	hn := newFallbackHarness(t, noRetry())
	limiter := &fakeLimiter{verdict: ratelimit.Verdict{Allowed: true}}
	hn.h.limiter = limiter
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}
	hn.openai.resp = &provider.Response{Content: "x", Model: "gpt-4o"}

	if rec := hn.post(t, "smart"); rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body)
	}

	if len(limiter.keys) != 1 {
		t.Fatalf("limiter consulted %d times, want 1: %v", len(limiter.keys), limiter.keys)
	}
	if limiter.keys[0] != "alias:smart" {
		t.Errorf("limit key = %q, want alias:smart", limiter.keys[0])
	}
}

func TestFallback_RetryHappensBeforeFallback(t *testing.T) {
	// The two mechanisms compose in one order and not the other: a
	// provider gets its retries before the chain gives up on it.
	// Otherwise a single blip would move traffic to a second vendor that
	// costs different money.
	opts := resilience.Options{
		MaxAttempts:      3,
		FailureThreshold: 100,
		BaseBackoff:      time.Millisecond,
		MaxBackoff:       2 * time.Millisecond,
		AttemptTimeout:   time.Second,
		Budget:           2 * time.Second,
	}
	hn := newFallbackHarness(t, opts)
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}
	hn.openai.resp = &provider.Response{Content: "x", Model: "gpt-4o"}

	rec := hn.post(t, "smart")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body)
	}
	// 3 attempts at sonnet, then 3 at haiku on the third hop is not
	// reached because OpenAI serves the second hop first.
	if hn.anth.calls != 3 {
		t.Errorf("anthropic calls = %d, want 3 (all retries before falling back)", hn.anth.calls)
	}
	if hn.openai.calls != 1 {
		t.Errorf("openai calls = %d, want 1", hn.openai.calls)
	}
}

func TestFallback_TotalTimeIsBoundedAcrossTheWholeChain(t *testing.T) {
	// Per-hop bounds multiplied by chain length is not a bound. This is
	// the one number that makes the whole request finite.
	hn := newFallbackHarness(t, resilience.Options{
		MaxAttempts:      3,
		FailureThreshold: 100,
		BaseBackoff:      time.Millisecond,
		AttemptTimeout:   500 * time.Millisecond,
		Budget:           500 * time.Millisecond,
	})
	hn.h.callBudget = 300 * time.Millisecond
	hn.anth.block = time.Hour
	hn.openai.block = time.Hour

	start := time.Now()
	rec := hn.post(t, "smart")
	elapsed := time.Since(start)

	if rec.Code == http.StatusOK {
		t.Fatal("status = 200, want a failure")
	}
	if elapsed > 3*time.Second {
		t.Errorf("request took %v — the call budget does not bound the chain", elapsed)
	}
}

func TestChain_IsOneLevelDeep(t *testing.T) {
	// Chains are not walked transitively, which is what makes cycles
	// structurally impossible rather than something to detect. `smart`
	// falls back to `fast`, and `fast` has its own chain, which must not
	// be spliced in.
	hn := newFallbackHarness(t, noRetry())

	rt, err := hn.h.resolve("smart")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	chain := hn.h.chain(rt)

	if len(chain) != 3 {
		t.Fatalf("chain length = %d, want 3 (smart, smart-alt, fast)", len(chain))
	}
	wantVia := []string{"", "smart-alt", "fast"}
	for i, want := range wantVia {
		if chain[i].via != want {
			t.Errorf("chain[%d].via = %q, want %q", i, chain[i].via, want)
		}
		if chain[i].alias != "smart" {
			t.Errorf("chain[%d].alias = %q, want smart on every hop", i, chain[i].alias)
		}
	}
}

func TestWriteProviderError_OpenBreakerNamesTheAlias(t *testing.T) {
	// The client asked in the alias vocabulary and should be answered in
	// it, not told about a provider name it never used.
	h := &handler{logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	rec := httptest.NewRecorder()

	h.writeProviderError(rec, route{alias: "smart", model: "claude-sonnet-5"},
		&resilience.OpenError{Provider: "anthropic", RetryAfter: 2500 * time.Millisecond})

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "alias smart") {
		t.Errorf("body = %s, want it to name the alias", rec.Body)
	}
	// Rounded up: advertising 2 for a 2.5s wait sends the client back
	// while the breaker is still open.
	if got := rec.Header().Get("Retry-After"); got != "3" {
		t.Errorf("Retry-After = %q, want 3", got)
	}
}

func TestWriteProviderError_WrappedOpenErrorIsStillRecognized(t *testing.T) {
	// errors.As walks the chain; a single Unwrap would silently stop
	// working the moment another layer of context was added.
	h := &handler{logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	rec := httptest.NewRecorder()

	wrapped := errors.Join(errors.New("context"),
		&resilience.OpenError{Provider: "openai", RetryAfter: time.Second})
	h.writeProviderError(rec, route{alias: "smart"}, wrapped)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
}
