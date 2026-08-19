package provider

import (
	"context"
	"errors"
	"fmt"
)

// ErrInvalidRequest marks a request the provider rejected before it ever
// went on the wire — a missing model, no messages, an unencodable body.
//
// It exists so the retry layer can tell "this call will fail again
// identically" from "this call might work next time" without inspecting
// error strings. Without it, a malformed request would look exactly like
// a dial failure: retried three times and counted against the provider's
// circuit breaker, which is how a bug in one caller takes a healthy
// provider out of rotation for everyone.
var ErrInvalidRequest = errors.New("provider: invalid request")

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

// validate applies the checks every provider shares, so each Do starts
// from the same guarantees and every rejection carries ErrInvalidRequest.
func validate(req *Request) error {
	switch {
	case req == nil:
		return fmt.Errorf("%w: nil request", ErrInvalidRequest)
	case req.Model == "":
		return fmt.Errorf("%w: model is required", ErrInvalidRequest)
	case len(req.Messages) == 0:
		return fmt.Errorf("%w: at least one message is required", ErrInvalidRequest)
	}
	return nil
}
