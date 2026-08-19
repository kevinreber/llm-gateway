package resilience

import (
	"context"
	"time"

	"github.com/kevinreber/llm-gateway/internal/provider"
)

// Options configures a wrapped provider. Every zero-valued field takes
// the corresponding Default constant, so Options{} is a usable policy.
type Options struct {
	// FailureThreshold is consecutive failures before the breaker opens.
	FailureThreshold int
	// RecoveryTimeout is how long the breaker stays open before probing.
	RecoveryTimeout time.Duration
	// MaxAttempts is the total number of calls to the provider for one
	// request, including the first. 1 disables retry.
	MaxAttempts int
	// BaseBackoff is the first retry delay, doubled per attempt.
	BaseBackoff time.Duration
	// MaxBackoff caps the exponential growth.
	MaxBackoff time.Duration
	// AttemptTimeout bounds a single call to the provider.
	AttemptTimeout time.Duration
	// Budget bounds all attempts and backoffs for this provider
	// combined. It is what stops "three attempts" from meaning "three
	// times the single-attempt worst case".
	Budget time.Duration
}

func (o Options) withDefaults() Options {
	if o.FailureThreshold <= 0 {
		o.FailureThreshold = DefaultFailureThreshold
	}
	if o.RecoveryTimeout <= 0 {
		o.RecoveryTimeout = DefaultRecoveryTimeout
	}
	if o.MaxAttempts <= 0 {
		o.MaxAttempts = DefaultMaxAttempts
	}
	if o.BaseBackoff <= 0 {
		o.BaseBackoff = DefaultBaseBackoff
	}
	if o.MaxBackoff <= 0 {
		o.MaxBackoff = DefaultMaxBackoff
	}
	if o.AttemptTimeout <= 0 {
		o.AttemptTimeout = DefaultAttemptTimeout
	}
	if o.Budget <= 0 {
		o.Budget = DefaultBudget
	}
	return o
}

// Provider decorates a provider.Provider with a circuit breaker and
// bounded retry, and is itself a provider.Provider.
//
// Wrapping rather than calling from the handler is what makes the
// resilience policy total: it applies to alias routes and direct-model
// routes alike, and it cannot be forgotten at a new call site, because
// there is no unwrapped provider in the registry to call. The cost is
// that Do's error now carries breaker semantics the handler has to
// understand, which is why ErrOpen is exported.
type Provider struct {
	inner   provider.Provider
	breaker *Breaker

	maxAttempts    int
	baseBackoff    time.Duration
	maxBackoff     time.Duration
	attemptTimeout time.Duration
	budget         time.Duration
}

// Wrap returns p guarded by a new breaker and retry policy.
func Wrap(p provider.Provider, opts Options) *Provider {
	opts = opts.withDefaults()
	return &Provider{
		inner:          p,
		breaker:        NewBreaker(p.Name(), opts.FailureThreshold, opts.RecoveryTimeout),
		maxAttempts:    opts.MaxAttempts,
		baseBackoff:    opts.BaseBackoff,
		maxBackoff:     opts.MaxBackoff,
		attemptTimeout: opts.AttemptTimeout,
		budget:         opts.Budget,
	}
}

// Name implements provider.Provider. It reports the wrapped provider's
// name unchanged, which is what keeps metrics labels, config lookups,
// cost rows, and the X-Gateway-Provider header stable across this
// change: nothing downstream can tell it is talking to a wrapper.
func (p *Provider) Name() string { return p.inner.Name() }

// Supports implements provider.Provider.
func (p *Provider) Supports(model string) bool { return p.inner.Supports(model) }

// Health implements provider.Provider. The probe is passed through
// without the breaker, deliberately: Health exists to answer "is this
// provider reachable", and gating it on the breaker would make an open
// breaker report the provider as unreachable forever, with no
// independent signal to contradict it.
func (p *Provider) Health(ctx context.Context) error { return p.inner.Health(ctx) }

// Breaker exposes the underlying breaker for the admin API and metrics.
func (p *Provider) Breaker() *Breaker { return p.breaker }

// Do implements provider.Provider: it calls the wrapped provider,
// retrying transient failures with jittered backoff, and refuses
// outright when the breaker is open.
//
// Every wait in here is bounded and every bound is derived from one
// deadline. ctx caps the whole call chain from the handler, budget caps
// this provider's share of it, attemptTimeout caps one call, and the
// backoff sleep selects on the same context rather than sleeping
// blind. There is no path through this function that waits on something
// with no deadline attached.
func (p *Provider) Do(ctx context.Context, req *provider.Request) (*provider.Response, error) {
	budgetCtx, cancelBudget := context.WithTimeout(ctx, p.budget)
	defer cancelBudget()

	var lastErr error
	for attempt := 1; attempt <= p.maxAttempts; attempt++ {
		if err := p.breaker.Allow(); err != nil {
			if lastErr != nil {
				// The breaker opened partway through our own retries.
				// Report the failure that caused it, not the symptom —
				// "circuit breaker open" tells an operator nothing about
				// why this request failed.
				return nil, lastErr
			}
			return nil, err
		}

		attemptCtx, cancelAttempt := context.WithTimeout(budgetCtx, p.attemptTimeout)
		resp, err := p.inner.Do(attemptCtx, req)
		cancelAttempt()

		if err == nil {
			p.breaker.RecordSuccess()
			return resp, nil
		}

		// The caller's context — the client's connection, or the
		// gateway-wide call budget — is gone. There is nothing left to
		// retry into, and the provider is not to blame for a client that
		// hung up, so this is not recorded against its health. An
		// attempt deadline firing while ctx is still live is a different
		// thing entirely and falls through to classify below, where it
		// counts as the provider being too slow.
		if ctx.Err() != nil {
			return nil, err
		}

		retry, unhealthy := classify(err)
		if unhealthy {
			p.breaker.RecordFailure()
		}
		lastErr = err

		if !retry || attempt == p.maxAttempts {
			return nil, err
		}
		wait, ok := p.backoff(budgetCtx, attempt, err)
		if !ok {
			// No budget left to wait and try again. Returning now hands
			// the remaining time to the fallback chain instead of
			// spending it here.
			return nil, err
		}
		if sleepCtx(budgetCtx, wait) != nil {
			return nil, err
		}
	}
	return nil, lastErr
}
