package resilience

import (
	"context"
	"errors"
	"math/rand/v2"
	"net/http"
	"time"

	"github.com/kevinreber/llm-gateway/internal/provider"
)

// Defaults applied when an Options field is left at its zero value.
//
// AttemptTimeout matches the provider clients' own http.Client timeout
// on purpose. Phase 3 must not quietly shorten how long a legitimately
// slow completion is allowed to take — a long Opus generation that
// worked before this package existed still has to work — so the retry
// layer bounds the number of attempts and the gaps between them, not the
// length of an attempt.
const (
	DefaultFailureThreshold = 5
	DefaultRecoveryTimeout  = 30 * time.Second
	DefaultMaxAttempts      = 3
	DefaultBaseBackoff      = 200 * time.Millisecond
	DefaultMaxBackoff       = 2 * time.Second
	DefaultAttemptTimeout   = 60 * time.Second
	// DefaultBudget bounds one provider's total share of a request:
	// every attempt, plus every backoff between them. It sits just above
	// AttemptTimeout so a single slow attempt is never cut short, while
	// a provider that keeps failing slowly still surrenders the request
	// to the fallback chain instead of consuming all of it.
	DefaultBudget = 65 * time.Second
)

// minAttemptTime is the smallest slice of budget worth starting an
// attempt with. Below this the request would be cancelled mid-flight
// anyway, so we skip it and let the caller fall back — burning an
// upstream's rate limit on a call we have already decided to abandon is
// the worst of both outcomes.
const minAttemptTime = 250 * time.Millisecond

// maxRetryAfterWait caps how long a Retry-After header can park a
// request inside the gateway.
//
// Honoring a short wait is worth it: the upstream knows its own quota
// better than our backoff curve does, and a second of patience turns a
// failure into a success. Honoring a long one is not. A provider saying
// "come back in 30 seconds" is a provider telling us to route
// elsewhere, and sleeping on it inside the request would hold the
// client's connection for half a minute to produce the same 429 we
// could have returned immediately — with an accurate Retry-After the
// client can act on, and after giving the fallback chain a turn.
//
// So a wait longer than this is not shortened, it is declined: the
// attempt loop stops and the error goes back up. Truncating the sleep
// instead would be the worst option, sending a retry at an upstream
// that just told us its quota is not free yet.
const maxRetryAfterWait = 3 * time.Second

// classify decides what an error means for retry and for provider health.
//
// The two answers are separate because they are separate questions. A
// 429 is worth retrying but says nothing bad about the provider — it is
// a healthy upstream correctly telling us to slow down, and tripping a
// breaker on it would take a working provider out of rotation for
// everyone because one alias was noisy. A 500 is worth retrying and does
// mean the provider is unwell. A 400 is neither.
//
// Everything reads provider.APIError's typed Status field rather than
// matching on message text, so a reworded upstream error message can
// never silently change our retry behavior.
func classify(err error) (retry, unhealthy bool) {
	if err == nil {
		return false, false
	}

	// Malformed input never improves on a second attempt and is our bug,
	// not the provider's.
	if errors.Is(err, provider.ErrInvalidRequest) {
		return false, false
	}

	var apiErr *provider.APIError
	if errors.As(err, &apiErr) {
		switch {
		case apiErr.Status >= 500:
			return true, true
		case apiErr.Status == http.StatusTooManyRequests:
			return true, false
		case apiErr.Status == http.StatusRequestTimeout:
			return true, true
		default:
			// Every other 4xx is a request the provider understood and
			// refused: bad key, bad model, context too long. Retrying
			// replays the same refusal.
			return false, false
		}
	}

	// Anything that is not a typed API error never reached a response:
	// dial failure, TLS error, connection reset, or an attempt deadline
	// firing. All of those are transient by nature and all of them are
	// the provider (or the path to it) failing to serve us.
	return true, true
}

// ShouldFallback reports whether a different provider is worth trying
// after err.
//
// It is the same question classify already answers for retry — "could
// this have gone differently?" — asked about a different provider rather
// than a second attempt at the same one, and the answers coincide. A 500
// or a dial failure is worth escaping; a 400 is a refusal any provider
// would repeat, so falling back would turn one honest error into a tour
// of the whole chain before returning the same thing.
//
// An open breaker is the case the fallback chain exists for, and it is
// called out explicitly here rather than left to fall through classify's
// default branch, because "the reason we did not call this provider is
// exactly why we should call another one" is the point, not a detail.
func ShouldFallback(err error) bool {
	if errors.Is(err, ErrOpen) {
		return true
	}
	retry, _ := classify(err)
	return retry
}

// backoff returns how long to wait before the next attempt, and whether
// waiting is worth it at all.
//
// The wait is exponential with half jitter: half the computed delay is
// fixed and half is random. Full jitter (uniform over the whole
// interval) can return a near-zero wait, which under a synchronized
// failure — exactly when backoff matters — puts the retry back on the
// wire immediately. Half jitter keeps a floor while still breaking up
// the thundering herd.
//
// An upstream Retry-After overrides our own number when it is longer,
// because the upstream knows something we don't — but only up to
// maxRetryAfterWait, and only within the remaining budget. Failing
// either check returns false, and the caller stops trying this provider
// and moves down the fallback chain rather than waiting.
func (p *Provider) backoff(ctx context.Context, attempt int, err error) (time.Duration, bool) {
	wait := p.baseBackoff << (attempt - 1)
	if wait > p.maxBackoff || wait <= 0 { // <=0 catches shift overflow
		wait = p.maxBackoff
	}
	wait = wait/2 + rand.N(wait/2+1)

	var apiErr *provider.APIError
	if errors.As(err, &apiErr) && apiErr.RetryAfter > wait {
		if apiErr.RetryAfter > maxRetryAfterWait {
			return 0, false
		}
		wait = apiErr.RetryAfter
	}

	if deadline, ok := ctx.Deadline(); ok {
		if remaining := time.Until(deadline); wait+minAttemptTime > remaining {
			return 0, false
		}
	}
	return wait, true
}

// sleepCtx waits for d or until ctx is done, whichever comes first.
// A bare time.Sleep here would be the one unbounded wait in the retry
// path: a client that hung up, or a request that has spent its budget,
// would still be held for the full backoff before anyone noticed.
func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
