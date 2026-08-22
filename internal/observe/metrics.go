// Package observe holds the gateway's Prometheus instrumentation.
//
// The surface:
//
//   - llm_gateway_requests_total{alias, provider, result}    counter
//   - llm_gateway_request_duration_seconds{alias, provider}  histogram
//   - llm_gateway_cost_cents_total{provider, model}          counter
//   - llm_gateway_breaker_state{provider}                    gauge
//   - llm_gateway_provider_health{provider}                  gauge
//
// Metrics are package-level singletons registered against
// prometheus.DefaultRegisterer, the same shape bucketd's observe package
// uses, so promhttp.Handler() exposes them with no extra wiring.
//
// Registering at package scope rather than inside Run is also what keeps
// Run callable more than once per process: promauto panics on a
// duplicate registration, and the gateway's own tests start several
// servers in a single test binary.
//
// The cache counters sketched for this phase are deliberately absent.
// There is no cache until Phase 5, and a metric that is always zero is
// worse than a missing one — it reads as "the cache is never hitting"
// rather than "there is no cache".
package observe

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Result values for the result dimension of llm_gateway_requests_total.
//
// The set is closed on purpose. result is a metric label, so every
// distinct value that reaches it becomes a permanent time series in
// whatever scrapes this process; deriving one from an upstream error
// string would hand cardinality control to the provider.
const (
	// ResultOK is a request a provider served.
	ResultOK = "ok"
	// ResultBadRequest is a request rejected before routing: malformed
	// JSON, a missing field, a model no provider serves.
	ResultBadRequest = "bad_request"
	// ResultRateLimited is a request denied by its alias's rate limit.
	ResultRateLimited = "rate_limited"
	// ResultCircuitOpen is a request refused because every provider in
	// its fallback chain had an open breaker.
	ResultCircuitOpen = "circuit_open"
	// ResultUpstreamRejected is a request the provider refused on its
	// own terms: a malformed prompt, a context-length overflow, a 429.
	//
	// Separate from ResultProviderError because these are not evidence
	// that the provider is unhealthy, and folding them together would
	// make the obvious alert — rate of provider_error — page somebody
	// for a caller sending bad prompts. internal/resilience already
	// draws this line for the circuit breaker, which is why a 400 does
	// not count toward opening it; the metric drawing it in a different
	// place than the breaker would be the inconsistency.
	ResultUpstreamRejected = "upstream_rejected"
	// ResultProviderError is a request where the provider itself failed:
	// a 5xx, a timeout, a transport error.
	ResultProviderError = "provider_error"
)

// LabelNone fills the alias or provider label when a request carries
// neither.
//
// A direct model name genuinely has no alias, and a request rejected
// before routing genuinely has no provider. Both are ordinary outcomes
// rather than missing data, and an empty label value renders as
// `alias=""` in a dashboard legend, which reads like an instrumentation
// bug every time somebody new looks at it.
const LabelNone = "none"

// ModelUnpriced is the model label for a completion whose model is not
// in the pricing table.
//
// One shared bucket rather than the model's own name, because the name
// in that case is an unrecognized string from the upstream and is
// exactly the input that cannot be trusted to be one of finitely many.
// The specific model is not lost — it is on the cost row and in the
// warning this path logs.
const ModelUnpriced = "unpriced"

// Breaker state gauge values. Ordered by severity so that a dashboard
// can alert on `max_over_time(llm_gateway_breaker_state[5m]) >= 2`
// without a mapping table.
const (
	BreakerClosed   = 0
	BreakerHalfOpen = 1
	BreakerOpen     = 2
)

var (
	requestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_gateway_requests_total",
			Help: "Total completion requests, partitioned by alias, serving provider, and outcome.",
		},
		[]string{"alias", "provider", "result"},
	)

	requestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "llm_gateway_request_duration_seconds",
			Help: "End-to-end latency of requests that reached the provider phase, " +
				"including retries and fallback hops.",
			// Tuned for completions, not for the sub-millisecond range
			// bucketd cares about: a short completion is hundreds of
			// milliseconds and a long one is tens of seconds. The top
			// bucket is the handler's call budget, so the count above it
			// is exactly "requests that exhausted the budget" — the
			// number worth alerting on.
			Buckets: []float64{0.01, 0.05, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 150},
		},
		[]string{"alias", "provider"},
	)

	costCentsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_gateway_cost_cents_total",
			Help: "Cumulative spend in cents, partitioned by the provider and model that billed it.",
		},
		[]string{"provider", "model"},
	)

	breakerState = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_gateway_breaker_state",
			Help: "Circuit breaker state per provider: 0 closed, 1 half-open, 2 open.",
		},
		[]string{"provider"},
	)

	cacheLookups = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_gateway_cache_lookups_total",
			Help: "Cache lookups by layer and outcome: hit, miss, or error.",
		},
		[]string{"layer", "result"},
	)

	costEventsDropped = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "llm_gateway_cost_events_dropped_total",
			Help: "Cost events discarded because the writer's buffer was full. " +
				"Non-zero means recorded spend is an undercount.",
		},
	)

	providerHealth = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_gateway_provider_health",
			Help: "1 when the gateway would currently admit a call to this provider, 0 when its " +
				"breaker is open. Derived from breaker state, not from an upstream probe.",
		},
		[]string{"provider"},
	)
)

// RecordRequest counts one completed request. Call exactly once per
// inbound request, with the route that actually served it.
//
// Pass LabelNone for alias or provider rather than an empty string.
func RecordRequest(alias, provider, result string) {
	requestsTotal.WithLabelValues(alias, provider, result).Inc()
}

// ObserveRequestDuration records how long a request spent in the
// gateway.
//
// Only requests that reached the provider phase belong here. Folding in
// the ones rejected at the door — bad JSON, a denied rate limit — would
// mix a population measured in microseconds into a histogram whose
// point is provider latency, and the percentiles that result describe
// neither group. Refusals are still counted; they are counted by
// RecordRequest, which is the metric that answers "how much traffic was
// shed".
//
// A request refused by an open breaker does belong here, and lands in
// the bottom bucket. That is the signal, not noise: converting a slow
// failure into an immediate one is what the breaker is for, and the
// histogram is where you can see it happen.
func ObserveRequestDuration(alias, provider string, d time.Duration) {
	requestDuration.WithLabelValues(alias, provider).Observe(d.Seconds())
}

// AddCostCents adds one request's spend to the running total.
//
// Labelled by the model that actually billed, which after a fallback is
// not the model the client asked for. The alias is deliberately not a
// label here: alias-level attribution lives in the costs table, where it
// can be joined and re-aggregated, and adding a third dimension to a
// counter that already crosses provider and model would multiply series
// for a question SQL answers better.
func AddCostCents(provider, model string, cents float64) {
	costCentsTotal.WithLabelValues(provider, model).Add(cents)
}

// Cache lookup outcomes.
const (
	CacheHit   = "hit"
	CacheMiss  = "miss"
	CacheError = "error"
)

// RecordCacheLookup counts one cache consultation.
//
// Hits and misses share a counter rather than the hits-only counter the
// design sketch called for, because the number anyone actually wants is
// the hit *rate*, and a hits-only series cannot produce one — there is
// no denominator. Errors are a third outcome rather than folded into
// misses: both mean the request goes to the provider, but only one of
// them means the cache is broken.
func RecordCacheLookup(layer, result string) {
	cacheLookups.WithLabelValues(layer, result).Inc()
}

// AddDroppedCostEvents records n cost events lost to a full buffer.
//
// The writer drops rather than blocks, which is the right trade — a
// slow Postgres must not become the gateway's latency — but it means
// the one place this system knowingly loses data had no signal on
// /metrics, only a line in a log nobody reads until afterwards. Any
// non-zero rate here says recorded spend is an undercount, which is the
// kind of thing you want to learn from a dashboard and not from a
// finance question three weeks later.
func AddDroppedCostEvents(n float64) {
	costEventsDropped.Add(n)
}

// SetBreakerState publishes a provider's breaker state and the health
// gauge derived from it.
//
// The two move together by construction because they are the same fact
// asked at two altitudes: breaker_state is for a human reading a
// dashboard, provider_health is for an alert rule that wants a single
// `min(llm_gateway_provider_health) == 0` across the fleet without
// encoding what 2 means.
//
// Half-open counts as healthy. The breaker is admitting a probe, so a
// request can get through, and paging on a provider that is in the
// middle of recovering on its own is how an alert teaches people to
// ignore it.
func SetBreakerState(provider string, state float64) {
	breakerState.WithLabelValues(provider).Set(state)
	healthy := 1.0
	if state == BreakerOpen {
		healthy = 0
	}
	providerHealth.WithLabelValues(provider).Set(healthy)
}
