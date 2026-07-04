package provider

import "context"

// Provider is the abstraction the proxy layer routes over.
//
// Implementations must be safe for concurrent use — the gateway serves
// per-request goroutines and reuses one Provider value across all of
// them, primarily so the underlying *http.Client's connection pool is
// shared.
type Provider interface {
	// Name returns a short, stable identifier used in metrics labels,
	// config lookups, and log fields. Never localize; never change once
	// deployed (breaks Prometheus label continuity).
	Name() string

	// Do executes a single completion. Callers pass ctx through; on ctx
	// cancellation the implementation must abort the outbound HTTP call
	// so we don't do wasted work on abandoned inbound requests.
	Do(ctx context.Context, req *Request) (*Response, error)

	// Supports reports whether this provider can serve the given model
	// name. Used by the router to sanity-check alias resolution — a
	// misconfigured YAML that maps `smart` → gemini/claude-sonnet-4-6
	// should fail loudly at request time, not silently pass through.
	Supports(model string) bool

	// Health issues a lightweight probe. Return nil when healthy. The
	// admin API surfaces the result; the circuit breaker uses it to
	// decide whether to attempt a recovery probe in half-open state.
	Health(ctx context.Context) error
}
