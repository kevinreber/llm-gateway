package gateway

import (
	"bytes"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/provider"
	"github.com/kevinreber/llm-gateway/internal/ratelimit"
	"github.com/kevinreber/llm-gateway/internal/resilience"
)

// scrape drives GET /metrics through the real route, so these tests
// exercise the scrape-time gauge refresh rather than reaching into the
// observe package and asserting on what was pushed.
func scrape(t *testing.T, h *handler) string {
	t.Helper()
	rec := httptest.NewRecorder()
	h.routes().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /metrics = %d, want 200", rec.Code)
	}
	return rec.Body.String()
}

// seriesValue returns the value of one exposition line, treating an
// absent series as zero.
//
// The metrics are process-wide singletons, so every assertion below is a
// delta across one request rather than an absolute count — an absolute
// one would pass or fail depending on which other tests in this package
// happened to run first.
func seriesValue(t *testing.T, body, series string) float64 {
	t.Helper()
	for line := range strings.SplitSeq(body, "\n") {
		name, val, ok := strings.Cut(line, " ")
		if !ok || name != series {
			continue
		}
		f, err := strconv.ParseFloat(strings.TrimSpace(val), 64)
		if err != nil {
			t.Fatalf("parse %q: %v", line, err)
		}
		return f
	}
	return 0
}

// tripsAfterOneFailure opens a breaker on the first failure and keeps it
// open for the whole test, so an assertion about an open circuit is not
// racing the recovery timer.
func tripsAfterOneFailure() resilience.Options {
	return resilience.Options{
		MaxAttempts:      1,
		FailureThreshold: 1,
		RecoveryTimeout:  time.Minute,
		AttemptTimeout:   time.Second,
		Budget:           2 * time.Second,
	}
}

func TestMetrics_CountsASuccessfulRequest(t *testing.T) {
	hn := newFallbackHarness(t, noRetry())
	const series = `llm_gateway_requests_total{alias="smart",provider="anthropic",result="ok"}`

	before := seriesValue(t, scrape(t, hn.h), series)
	if rec := hn.post(t, "smart"); rec.Code != http.StatusOK {
		t.Fatalf("POST = %d, want 200", rec.Code)
	}
	after := seriesValue(t, scrape(t, hn.h), series)

	if after-before != 1 {
		t.Errorf("ok count delta = %v, want 1", after-before)
	}
}

func TestMetrics_AttributesAFallbackToTheProviderThatServed(t *testing.T) {
	// The primary's error rate is the number this counter exists to
	// expose. Billing a fallback's success to the primary would let the
	// replacement quietly cover for it.
	hn := newFallbackHarness(t, noRetry())
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}
	hn.openai.resp = &provider.Response{
		Content: "from openai",
		Model:   "gpt-4o",
		Usage:   provider.Usage{InputTokens: 100, OutputTokens: 50},
	}

	const servedByOpenAI = `llm_gateway_requests_total{alias="smart",provider="openai",result="ok"}`
	const servedByAnthropic = `llm_gateway_requests_total{alias="smart",provider="anthropic",result="ok"}`

	body := scrape(t, hn.h)
	beforeOAI := seriesValue(t, body, servedByOpenAI)
	beforeAnth := seriesValue(t, body, servedByAnthropic)

	if rec := hn.post(t, "smart"); rec.Code != http.StatusOK {
		t.Fatalf("POST = %d, want 200", rec.Code)
	}

	body = scrape(t, hn.h)
	if got := seriesValue(t, body, servedByOpenAI) - beforeOAI; got != 1 {
		t.Errorf("openai ok delta = %v, want 1", got)
	}
	if got := seriesValue(t, body, servedByAnthropic) - beforeAnth; got != 0 {
		t.Errorf("anthropic ok delta = %v, want 0 — the primary did not serve this", got)
	}
}

func TestMetrics_CountsShedTraffic(t *testing.T) {
	// A rate-limited request keeps the provider label it was routed to.
	// "Which upstream is this shedding protect" is the question the
	// label answers, and dropping it to none would erase it.
	hn := newFallbackHarness(t, noRetry())
	hn.h.limiter = &fakeLimiter{
		verdict: ratelimit.Verdict{Allowed: false, RetryAfter: time.Second},
	}

	const series = `llm_gateway_requests_total{alias="smart",provider="anthropic",result="rate_limited"}`
	before := seriesValue(t, scrape(t, hn.h), series)

	if rec := hn.post(t, "smart"); rec.Code != http.StatusTooManyRequests {
		t.Fatalf("POST = %d, want 429", rec.Code)
	}

	if got := seriesValue(t, scrape(t, hn.h), series) - before; got != 1 {
		t.Errorf("rate_limited delta = %v, want 1", got)
	}
}

func TestMetrics_CountsRequestsRejectedBeforeRouting(t *testing.T) {
	hn := newFallbackHarness(t, noRetry())
	const series = `llm_gateway_requests_total{alias="none",provider="none",result="bad_request"}`

	before := seriesValue(t, scrape(t, hn.h), series)

	rec := httptest.NewRecorder()
	hn.h.routes().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST = %d, want 400", rec.Code)
	}

	if got := seriesValue(t, scrape(t, hn.h), series) - before; got != 1 {
		t.Errorf("bad_request delta = %v, want 1", got)
	}
}

func TestMetrics_DoorRejectionsStayOutOfTheLatencyHistogram(t *testing.T) {
	// Mixing microsecond-scale refusals into a histogram measuring
	// provider latency produces percentiles that describe neither
	// population. The refusals are still counted — by requests_total.
	hn := newFallbackHarness(t, noRetry())
	const series = `llm_gateway_request_duration_seconds_count{alias="none",provider="none"}`

	before := seriesValue(t, scrape(t, hn.h), series)

	rec := httptest.NewRecorder()
	hn.h.routes().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"nope",`+userTurn+`}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST = %d, want 400", rec.Code)
	}

	if got := seriesValue(t, scrape(t, hn.h), series) - before; got != 0 {
		t.Errorf("histogram count delta = %v, want 0", got)
	}
}

func TestMetrics_ObservesLatencyForServedRequests(t *testing.T) {
	hn := newFallbackHarness(t, noRetry())
	const series = `llm_gateway_request_duration_seconds_count{alias="smart",provider="anthropic"}`

	before := seriesValue(t, scrape(t, hn.h), series)
	hn.post(t, "smart")

	if got := seriesValue(t, scrape(t, hn.h), series) - before; got != 1 {
		t.Errorf("histogram count delta = %v, want 1", got)
	}
}

func TestMetrics_ReflectsAnOpenBreaker(t *testing.T) {
	// `lonely` has no fallback chain, so the failure lands on Anthropic
	// and stays there instead of being absorbed by OpenAI.
	hn := newFallbackHarness(t, tripsAfterOneFailure())

	const stateSeries = `llm_gateway_breaker_state{provider="anthropic"}`
	const healthSeries = `llm_gateway_provider_health{provider="anthropic"}`

	body := scrape(t, hn.h)
	if got := seriesValue(t, body, stateSeries); got != 0 {
		t.Fatalf("breaker_state before failure = %v, want 0 (closed)", got)
	}
	if got := seriesValue(t, body, healthSeries); got != 1 {
		t.Fatalf("provider_health before failure = %v, want 1", got)
	}

	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}
	hn.post(t, "lonely")

	body = scrape(t, hn.h)
	if got := seriesValue(t, body, stateSeries); got != 2 {
		t.Errorf("breaker_state after failure = %v, want 2 (open)", got)
	}
	if got := seriesValue(t, body, healthSeries); got != 0 {
		t.Errorf("provider_health after failure = %v, want 0", got)
	}
}

func TestMetrics_CountsCircuitOpenSeparatelyFromProviderErrors(t *testing.T) {
	// A refusal the gateway made on its own is a different operational
	// event from an upstream that answered badly, and an operator
	// looking at error rate needs to tell them apart.
	hn := newFallbackHarness(t, tripsAfterOneFailure())
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "overloaded"}

	const series = `llm_gateway_requests_total{alias="lonely",provider="anthropic",result="circuit_open"}`
	before := seriesValue(t, scrape(t, hn.h), series)

	hn.post(t, "lonely") // trips the breaker
	rec := hn.post(t, "lonely")
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("second POST = %d, want 503", rec.Code)
	}

	if got := seriesValue(t, scrape(t, hn.h), series) - before; got != 1 {
		t.Errorf("circuit_open delta = %v, want 1", got)
	}
}

func TestMetrics_CostFollowsTheModelThatBilled(t *testing.T) {
	hn := newFallbackHarness(t, noRetry())
	const series = `llm_gateway_cost_cents_total{model="claude-sonnet-5",provider="anthropic"}`

	before := seriesValue(t, scrape(t, hn.h), series)
	hn.post(t, "smart")

	if got := seriesValue(t, scrape(t, hn.h), series) - before; got <= 0 {
		t.Errorf("cost delta = %v, want > 0", got)
	}
}

func TestHealthz_ServingByDefault(t *testing.T) {
	// A handler with no serving flag is what a unit test builds; it must
	// not read as "shutting down".
	hn := newFallbackHarness(t, noRetry())

	rec := httptest.NewRecorder()
	hn.h.routes().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("GET /healthz = %d, want 200", rec.Code)
	}
}

func TestHealthz_ReportsNotServingDuringShutdown(t *testing.T) {
	hn := newFallbackHarness(t, noRetry())
	var serving atomic.Int32
	serving.Store(1)
	hn.h.serving = &serving

	rec := httptest.NewRecorder()
	hn.h.routes().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("while serving: %d, want 200", rec.Code)
	}

	serving.Store(0)

	rec = httptest.NewRecorder()
	hn.h.routes().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("while draining: %d, want 503", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "shutting_down") {
		t.Errorf("body = %s, want it to say shutting_down", rec.Body)
	}
}

func TestRequestID_GeneratedWhenAbsent(t *testing.T) {
	hn := newFallbackHarness(t, noRetry())

	rec := hn.post(t, "smart")

	id := rec.Header().Get(requestIDHeader)
	if len(id) != 32 {
		t.Fatalf("generated id = %q, want 32 hex chars", id)
	}
	if strings.Trim(id, "0123456789abcdef") != "" {
		t.Errorf("generated id = %q, want hex only", id)
	}
}

func TestRequestID_IsUniquePerRequest(t *testing.T) {
	hn := newFallbackHarness(t, noRetry())

	first := hn.post(t, "smart").Header().Get(requestIDHeader)
	second := hn.post(t, "smart").Header().Get(requestIDHeader)

	if first == second {
		t.Errorf("two requests shared id %q", first)
	}
}

func TestRequestID_ReusesAnUpstreamValue(t *testing.T) {
	// One identifier spanning the whole hop chain is the entire point of
	// accepting the header at all.
	hn := newFallbackHarness(t, noRetry())

	req := httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"smart",`+userTurn+`}`))
	req.Header.Set(requestIDHeader, "edge-proxy-abc123")
	rec := httptest.NewRecorder()
	hn.h.routes().ServeHTTP(rec, req)

	if got := rec.Header().Get(requestIDHeader); got != "edge-proxy-abc123" {
		t.Errorf("id = %q, want the supplied value", got)
	}
}

func TestRequestID_RejectsUntrustworthyValues(t *testing.T) {
	// A control character in a client-controlled field that reaches both
	// a log line and a response header is log forging and response
	// splitting respectively. Go's own server rejects these on the wire;
	// this guard is what holds when the value arrives another way.
	cases := []struct {
		name string
		id   string
	}{
		{"newline", "abc\ndef"},
		{"carriage return", "abc\rdef"},
		{"null byte", "abc\x00def"},
		{"non-ascii", "abc\u00e9def"},
		{"oversized", strings.Repeat("a", maxRequestIDBytes+1)},
	}

	hn := newFallbackHarness(t, noRetry())
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/messages",
				strings.NewReader(`{"model":"smart",`+userTurn+`}`))
			req.Header.Set(requestIDHeader, tc.id)
			rec := httptest.NewRecorder()
			hn.h.routes().ServeHTTP(rec, req)

			got := rec.Header().Get(requestIDHeader)
			if got == tc.id {
				t.Fatalf("echoed the untrusted value %q back", got)
			}
			if len(got) != 32 {
				t.Errorf("replacement id = %q, want a generated 32-char id", got)
			}
		})
	}
}

func TestRequestID_ReachesTheLogs(t *testing.T) {
	// The whole point of the identifier: a client holding a failed
	// response can quote it and find every line that request produced.
	var buf bytes.Buffer
	hn := newFallbackHarness(t, noRetry())
	hn.h.logger = slog.New(slog.NewJSONHandler(&buf, nil))
	hn.anth.err = errors.New("upstream exploded")

	rec := hn.post(t, "lonely")
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("POST = %d, want 502", rec.Code)
	}

	id := rec.Header().Get(requestIDHeader)
	if id == "" {
		t.Fatal("no request id on the response")
	}
	if !strings.Contains(buf.String(), `"request_id":"`+id+`"`) {
		t.Errorf("logs did not carry request_id %q:\n%s", id, buf.String())
	}
}

func TestMetrics_CollapsesDatedSnapshotsOntoTheFamily(t *testing.T) {
	// The model string on a cost row is whatever the upstream echoed
	// back. Using it as a label directly would let a provider mint a
	// permanent time series per value it returns, and counter series are
	// never retired. Collapsing onto the pricing table also gives the
	// dashboard the number it actually wants: what the family costs.
	hn := newFallbackHarness(t, noRetry())
	hn.anth.resp = &provider.Response{
		Content: "ok",
		Model:   "claude-sonnet-5-20251001",
		Usage:   provider.Usage{InputTokens: 1000, OutputTokens: 500},
	}

	const family = `llm_gateway_cost_cents_total{model="claude-sonnet-5",provider="anthropic"}`
	const snapshot = `llm_gateway_cost_cents_total{model="claude-sonnet-5-20251001",provider="anthropic"}`

	before := seriesValue(t, scrape(t, hn.h), family)
	hn.post(t, "smart")

	body := scrape(t, hn.h)
	if got := seriesValue(t, body, family) - before; got <= 0 {
		t.Errorf("family series delta = %v, want > 0", got)
	}
	if strings.Contains(body, snapshot) {
		t.Error("dated snapshot got its own series; the label is not bounded by the pricing table")
	}
}

func TestMetrics_UnpricedModelsShareOneSeries(t *testing.T) {
	// An unrecognized model name is exactly the input that cannot be
	// trusted to be one of finitely many, so it goes in a shared bucket.
	// The specific name survives on the cost row and in the warning log.
	hn := newFallbackHarness(t, noRetry())
	hn.anth.resp = &provider.Response{
		Content: "ok",
		Model:   "claude-experimental-does-not-exist",
		Usage:   provider.Usage{InputTokens: 1000, OutputTokens: 500},
	}

	hn.post(t, "smart")

	body := scrape(t, hn.h)
	if !strings.Contains(body, `llm_gateway_cost_cents_total{model="unpriced",provider="anthropic"}`) {
		t.Error("expected the unpriced bucket series")
	}
	if strings.Contains(body, `model="claude-experimental-does-not-exist"`) {
		t.Error("unpriced model name leaked into a label")
	}
}
