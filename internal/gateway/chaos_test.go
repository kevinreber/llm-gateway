package gateway

import (
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/config"
	"github.com/kevinreber/llm-gateway/internal/cost"
	"github.com/kevinreber/llm-gateway/internal/provider"
	"github.com/kevinreber/llm-gateway/internal/ratelimit"
	"github.com/kevinreber/llm-gateway/internal/resilience"
)

// chaosYAML gives `smart` a chain and `lonely` none, so one run can
// measure what the chain is actually worth.
const chaosYAML = `
aliases:
  smart:     { provider: anthropic, model: claude-sonnet-5 }
  smart-alt: { provider: openai,    model: gpt-4o }
  lonely:    { provider: anthropic, model: claude-sonnet-5 }

fallback:
  smart: [smart-alt]
`

// flakyUpstream is an HTTP server that fails a fixed fraction of
// requests with 503.
//
// Seeded deterministically: a chaos test that passes or fails on the
// weather is a test that gets marked flaky and then ignored, which is
// worse than not having one. The seed is fixed, the arrival order is
// not, so this still exercises concurrent breaker transitions under
// -race while producing a stable failure count.
type flakyUpstream struct {
	mu       sync.Mutex
	rng      *rand.Rand
	failRate float64
	requests atomic.Int64
	failures atomic.Int64
}

func (f *flakyUpstream) shouldFail() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.rng.Float64() < f.failRate
}

// anthropicChaosServer speaks the real Anthropic wire format, so the
// chaos runs through the production client rather than a stand-in: its
// error parsing, its Retry-After handling, its response decoding.
func anthropicChaosServer(t *testing.T, failRate float64) (*httptest.Server, *flakyUpstream) {
	t.Helper()
	up := &flakyUpstream{rng: rand.New(rand.NewPCG(0x5eed, 0xf00d)), failRate: failRate}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		up.requests.Add(1)
		if up.shouldFail() {
			up.failures.Add(1)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"msg_chaos","type":"message","role":"assistant",
			"content":[{"type":"text","text":"ok"}],
			"model":"claude-sonnet-5","stop_reason":"end_turn",
			"usage":{"input_tokens":10,"output_tokens":5}
		}`)
	}))
	t.Cleanup(srv.Close)
	return srv, up
}

func openAIChaosServer(t *testing.T, failRate float64) (*httptest.Server, *flakyUpstream) {
	t.Helper()
	up := &flakyUpstream{rng: rand.New(rand.NewPCG(0xbeef, 0xcafe)), failRate: failRate}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		up.requests.Add(1)
		if up.shouldFail() {
			up.failures.Add(1)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = io.WriteString(w, `{"error":{"message":"server had an error","type":"server_error"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-chaos","object":"chat.completion","model":"gpt-4o-2024-08-06",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
		}`)
	}))
	t.Cleanup(srv.Close)
	return srv, up
}

func newChaosHandler(t *testing.T, anthURL, oaiURL string) *handler {
	t.Helper()

	cfg, err := config.Parse(strings.NewReader(chaosYAML))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}

	opts := resilience.Options{
		MaxAttempts:      3,
		FailureThreshold: 5,
		// Short enough that the breaker cycles several times over a
		// 200-request run rather than opening once and staying open,
		// which is what makes half-open contention part of the test.
		RecoveryTimeout: 20 * time.Millisecond,
		BaseBackoff:     time.Millisecond,
		MaxBackoff:      4 * time.Millisecond,
		AttemptTimeout:  2 * time.Second,
		Budget:          3 * time.Second,
	}

	anth := resilience.Wrap(provider.NewAnthropicWithBaseURL("k", anthURL), opts)
	oai := resilience.Wrap(provider.NewOpenAIWithBaseURL("k", oaiURL), opts)

	return &handler{
		providers: map[string]provider.Provider{
			provider.AnthropicName: anth,
			provider.OpenAIName:    oai,
		},
		providerOrder: []string{provider.AnthropicName, provider.OpenAIName},
		cfg:           cfg,
		limiter:       ratelimit.AllowAll{},
		costs:         cost.Discard{},
		logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
		callBudget:    10 * time.Second,
	}
}

// chaosResult is what one run of drive() measured.
type chaosResult struct {
	total     int
	succeeded int
	fellBack  int
}

func (r chaosResult) successRate() float64 {
	if r.total == 0 {
		return 0
	}
	return float64(r.succeeded) / float64(r.total)
}

// drive fires n requests at an alias through concurrent workers and
// reports how many were answered.
func drive(t *testing.T, h *handler, alias string, n, workers int) chaosResult {
	t.Helper()

	var succeeded, fellBack atomic.Int64
	routes := h.routes()

	var wg sync.WaitGroup
	work := make(chan int, n)
	for i := range n {
		work <- i
	}
	close(work)

	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range work {
				req := httptest.NewRequest(http.MethodPost, "/v1/messages",
					strings.NewReader(`{"model":"`+alias+`","messages":[{"role":"user","content":"hi"}]}`))
				rec := httptest.NewRecorder()
				routes.ServeHTTP(rec, req)
				if rec.Code == http.StatusOK {
					succeeded.Add(1)
					if rec.Header().Get("X-Gateway-Fallback") != "" {
						fellBack.Add(1)
					}
				}
			}
		}()
	}
	wg.Wait()

	return chaosResult{
		total:     n,
		succeeded: int(succeeded.Load()),
		fellBack:  int(fellBack.Load()),
	}
}

func TestChaos_FallbackKeepsSuccessRateAbove95Percent(t *testing.T) {
	// The Week 3 milestone as an assertion: with a primary failing half
	// its requests, the gateway still answers better than 95% of the
	// time. Retry absorbs the isolated blips, the breaker stops us
	// hammering a provider that is clearly unwell, and the chain routes
	// around it while it recovers.
	anthSrv, anthUp := anthropicChaosServer(t, 0.5)
	oaiSrv, _ := openAIChaosServer(t, 0)
	h := newChaosHandler(t, anthSrv.URL, oaiSrv.URL)

	got := drive(t, h, "smart", 200, 8)

	if rate := got.successRate(); rate < 0.95 {
		t.Errorf("success rate = %.1f%% (%d/%d), want >= 95%%",
			rate*100, got.succeeded, got.total)
	}
	// Guard against a vacuous pass: if the injected chaos never actually
	// fired, a 100% success rate would prove nothing about resilience.
	if anthUp.failures.Load() == 0 {
		t.Error("no failures were injected — the chaos harness did nothing")
	}
	if got.fellBack == 0 {
		t.Error("no request was served by the fallback chain, so the chain was not what kept the rate up")
	}
	t.Logf("success %d/%d (%.1f%%), fell back %d, upstream 503s %d/%d",
		got.succeeded, got.total, got.successRate()*100,
		got.fellBack, anthUp.failures.Load(), anthUp.requests.Load())
}

func TestChaos_WithoutAChainTheSameFailuresGetThrough(t *testing.T) {
	// The control for the test above. `lonely` points at the same flaky
	// provider with the same retry and the same breaker, and differs
	// only in having nowhere to go. If this also cleared 95% then the
	// previous test would be measuring retry, not fallback.
	anthSrv, anthUp := anthropicChaosServer(t, 0.5)
	oaiSrv, _ := openAIChaosServer(t, 0)
	h := newChaosHandler(t, anthSrv.URL, oaiSrv.URL)

	got := drive(t, h, "lonely", 200, 8)

	if anthUp.failures.Load() == 0 {
		t.Fatal("no failures were injected — the chaos harness did nothing")
	}
	if rate := got.successRate(); rate >= 0.95 {
		t.Errorf("success rate = %.1f%% without a fallback chain; the chained test is not measuring fallback",
			rate*100)
	}
	t.Logf("success %d/%d (%.1f%%) with no fallback chain",
		got.succeeded, got.total, got.successRate()*100)
}

func TestChaos_BreakerCapsUpstreamCallsWhenAProviderIsDown(t *testing.T) {
	// A provider that fails everything must stop receiving traffic.
	// Without a breaker, 200 requests at 3 attempts each would be 600
	// calls into a hole; with one, the open periods keep it to a
	// fraction of that, and the chain still answers every request.
	anthSrv, anthUp := anthropicChaosServer(t, 1.0)
	oaiSrv, _ := openAIChaosServer(t, 0)
	h := newChaosHandler(t, anthSrv.URL, oaiSrv.URL)

	const requests = 200
	got := drive(t, h, "smart", requests, 8)

	if got.succeeded != requests {
		t.Errorf("succeeded %d/%d, want all — the healthy fallback should serve every request",
			got.succeeded, requests)
	}
	calls := anthUp.requests.Load()
	if calls >= requests {
		t.Errorf("dead provider received %d calls for %d requests; the breaker is not shedding load",
			calls, requests)
	}
	t.Logf("dead provider received %d calls for %d requests", calls, requests)
}
