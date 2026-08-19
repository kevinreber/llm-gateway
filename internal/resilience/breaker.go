// Package resilience wraps a provider.Provider with a circuit breaker
// and bounded retry, so the routing layer can treat "call this provider,
// giving up sensibly" as a single operation.
//
// This is not internal/proxy as the original layout sketched. The HTTP
// handler lives in internal/gateway, and a package named "proxy" that
// contained no proxy would be a worse lie than a slightly different
// name. What is here is exactly the two mechanisms that have no
// knowledge of HTTP routing, config files, or cost accounting: a state
// machine over failure counts, and a retry loop over a Provider. The
// fallback chain is deliberately NOT here — choosing which alias to try
// next needs the alias table, the provider registry, and the cost label,
// all of which belong to the handler.
//
// Everything in this package is safe for concurrent use. One Provider
// value is shared across every request goroutine, and a breaker's whole
// purpose is to aggregate failures across them.
package resilience

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// State is a circuit-breaker state.
type State int

// The three classic breaker states.
const (
	// StateClosed passes every call through. Steady state.
	StateClosed State = iota
	// StateOpen rejects every call without touching the provider.
	StateOpen
	// StateHalfOpen admits exactly one probe to test recovery.
	StateHalfOpen
)

func (s State) String() string {
	switch s {
	case StateClosed:
		return "closed"
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half-open"
	default:
		return fmt.Sprintf("State(%d)", int(s))
	}
}

// ErrOpen is the sentinel behind every rejection from an open breaker.
// Callers match it with errors.Is and read *OpenError with errors.As
// when they want the wait time.
var ErrOpen = errors.New("circuit breaker open")

// OpenError reports a call rejected because the breaker was open.
//
// RetryAfter is how long until the breaker will admit a probe. It exists
// so the gateway can put an honest Retry-After on the 503 it returns
// when every provider in a fallback chain is open — telling a client to
// come back at some unspecified time is barely better than telling it
// nothing.
type OpenError struct {
	Provider   string
	RetryAfter time.Duration
}

func (e *OpenError) Error() string {
	return fmt.Sprintf("%s: circuit breaker open (retry after %s)", e.Provider, e.RetryAfter.Round(time.Millisecond))
}

// Unwrap lets errors.Is(err, ErrOpen) match.
func (e *OpenError) Unwrap() error { return ErrOpen }

// Breaker is a per-provider circuit breaker.
//
// The lock discipline is the point of the design: the state check runs
// on every single request, and the overwhelming majority of those find a
// closed breaker and need no mutation at all. That path takes RLock
// only, so concurrent requests never serialize on each other. The
// exclusive lock is reserved for the three transitions and for claiming
// the half-open probe slot, all of which are rare by construction — a
// breaker that transitions often is a breaker whose thresholds are
// wrong.
//
// Failure counting is consecutive, not windowed: any success resets the
// count. That is the classic formulation and it is the right default
// here, because the failure mode this guards against — a provider that
// is down — produces unbroken runs of failures, while a provider that
// merely errors occasionally should not be taken out of rotation.
type Breaker struct {
	name      string
	threshold int
	recovery  time.Duration

	// now is a seam for tests. Production always uses time.Now; a
	// breaker test that waited out a real recovery timeout would be a
	// test nobody runs.
	now func() time.Time

	mu       sync.RWMutex
	state    State
	failures int
	openedAt time.Time
	// probing is true while a half-open probe is in flight. It caps
	// recovery traffic at one request: without it, every goroutine that
	// arrived during the open period would stampede a provider that has
	// just shown the first sign of life.
	probing bool
}

// NewBreaker returns a closed breaker. A threshold or recovery timeout
// at or below zero falls back to the package default rather than
// producing a breaker that trips instantly or never closes again.
func NewBreaker(name string, threshold int, recovery time.Duration) *Breaker {
	if threshold <= 0 {
		threshold = DefaultFailureThreshold
	}
	if recovery <= 0 {
		recovery = DefaultRecoveryTimeout
	}
	return &Breaker{
		name:      name,
		threshold: threshold,
		recovery:  recovery,
		now:       time.Now,
		state:     StateClosed,
	}
}

// Allow reports whether a call may proceed. A nil return means go; an
// *OpenError means the circuit is open and the provider must not be
// called.
//
// A caller that gets nil back MUST report the outcome with exactly one
// of RecordSuccess or RecordFailure. Skipping it leaks the half-open
// probe slot, and the breaker would then sit in half-open refusing every
// request until the process restarts.
func (b *Breaker) Allow() error {
	b.mu.RLock()
	closed := b.state == StateClosed
	b.mu.RUnlock()
	if closed {
		return nil
	}
	return b.allowSlow()
}

// allowSlow handles the open and half-open cases, which need the
// exclusive lock either to transition or to claim the probe slot. The
// state is re-read here rather than passed in: another goroutine may
// have transitioned between the RUnlock above and this Lock.
func (b *Breaker) allowSlow() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	switch b.state {
	case StateClosed:
		return nil

	case StateOpen:
		elapsed := b.now().Sub(b.openedAt)
		if elapsed < b.recovery {
			return &OpenError{Provider: b.name, RetryAfter: b.recovery - elapsed}
		}
		b.state = StateHalfOpen
		b.probing = true
		return nil

	case StateHalfOpen:
		if b.probing {
			// Someone else is already testing the water. Reject without
			// a wait hint we can't honestly compute — we don't know how
			// long that probe will take.
			return &OpenError{Provider: b.name, RetryAfter: b.recovery}
		}
		b.probing = true
		return nil

	default:
		return nil
	}
}

// RecordSuccess reports that a call allowed by this breaker succeeded.
func (b *Breaker) RecordSuccess() {
	b.mu.Lock()
	defer b.mu.Unlock()

	switch b.state {
	case StateHalfOpen:
		b.state = StateClosed
		b.failures = 0
		b.probing = false
	case StateClosed:
		b.failures = 0
	case StateOpen:
		// A call that was in flight when another goroutine tripped the
		// breaker has just come back successful. Its result is stale
		// evidence about a provider we have since decided is unhealthy,
		// so it does not get to reopen the gate early.
	}
}

// RecordFailure reports that a call allowed by this breaker failed in a
// way that counts against the provider's health. Callers decide what
// counts — see classify in retry.go — because an HTTP 400 is a failure
// of the request, not of the provider.
func (b *Breaker) RecordFailure() {
	b.mu.Lock()
	defer b.mu.Unlock()

	switch b.state {
	case StateClosed:
		b.failures++
		if b.failures >= b.threshold {
			b.trip()
		}
	case StateHalfOpen:
		// The probe failed: straight back to open with a fresh timer,
		// no second chance. Half-open is already the second chance.
		b.trip()
	case StateOpen:
		// Already open; the timer stays where it is. Restarting it on
		// in-flight stragglers would push recovery out indefinitely
		// under load.
	}
}

// trip moves the breaker to open. Caller must hold the write lock.
func (b *Breaker) trip() {
	b.state = StateOpen
	b.openedAt = b.now()
	b.failures = 0
	b.probing = false
}

// State returns the current state. Intended for the admin API and
// metrics in Phase 4; it is a point-in-time read, not a reservation, so
// nothing should gate a call on it.
func (b *Breaker) State() State {
	b.mu.RLock()
	defer b.mu.RUnlock()
	// Report half-open once the recovery timeout has elapsed even
	// though no request has driven the transition yet. Otherwise a
	// dashboard shows "open" indefinitely for an idle provider that is
	// in fact ready to be probed.
	if b.state == StateOpen && b.now().Sub(b.openedAt) >= b.recovery {
		return StateHalfOpen
	}
	return b.state
}

// Name returns the provider name this breaker guards.
func (b *Breaker) Name() string { return b.name }
