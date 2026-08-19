package resilience

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/provider"
)

// TestBreaker_ProbeSlotIsReleasedByEveryOutcome asserts the invariant
// directly, on the breaker, with no retry loop in the way.
//
// This is the test whose absence let the leak ship. breaker_test.go
// exercised Allow-then-RecordSuccess and Allow-then-RecordFailure, but
// never asked what happens to the probe slot when a caller has nothing
// to vote — which is the majority of real outcomes, since every 4xx and
// every cancelled client lands there.
func TestBreaker_ProbeSlotIsReleasedByEveryOutcome(t *testing.T) {
	outcomes := map[string]func(*Breaker){
		"success":       (*Breaker).RecordSuccess,
		"failure":       (*Breaker).RecordFailure,
		"indeterminate": (*Breaker).RecordIndeterminate,
	}

	for name, record := range outcomes {
		t.Run(name, func(t *testing.T) {
			b, clock := newTestBreaker(1, 30*time.Second)
			_ = b.Allow()
			b.RecordFailure() // open
			clock.Advance(30 * time.Second)

			if err := b.Allow(); err != nil {
				t.Fatalf("probe Allow = %v, want nil", err)
			}
			record(b)

			// Whatever the verdict, the breaker must be reachable again:
			// either closed, or open with a recovery window that expires.
			clock.Advance(30 * time.Second)
			if err := b.Allow(); err != nil {
				t.Fatalf("Allow after %s = %v; the probe slot was never released", name, err)
			}
		})
	}
}

// TestBreaker_IndeterminateDoesNotVote pins the reason RecordSuccess is
// not an acceptable stand-in: it would reset the consecutive-failure
// count, so a burst of 429s could erase the record of the 5xx before it
// and hold a failing provider in rotation indefinitely.
func TestBreaker_IndeterminateDoesNotVote(t *testing.T) {
	b, _ := newTestBreaker(3, time.Minute)

	for range 2 {
		_ = b.Allow()
		b.RecordFailure()
	}
	// Two 429s in the middle of a run of 5xx must neither trip the
	// breaker nor absolve the provider.
	for range 2 {
		_ = b.Allow()
		b.RecordIndeterminate()
	}
	if got := b.State(); got != StateClosed {
		t.Fatalf("state = %v after indeterminate outcomes, want closed", got)
	}

	_ = b.Allow()
	b.RecordFailure()
	if got := b.State(); got != StateOpen {
		t.Fatalf("state = %v, want open — the third real failure should still trip it", got)
	}
}

// TestProvider_HalfOpenProbeSurvives4xx is the end-to-end version, and
// the direct regression: before the fix, one 400 landing on the single
// half-open probe left the breaker rejecting every request forever.
func TestProvider_HalfOpenProbeSurvives4xx(t *testing.T) {
	for _, tc := range []struct {
		name  string
		probe error
	}{
		{"400", apiErr(400, 0)},
		{"429", apiErr(429, 0)},
		{"invalid request", fmt.Errorf("%w: nope", provider.ErrInvalidRequest)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inner := &scriptedProvider{name: "p", results: []error{
				apiErr(503, 0), // trips the breaker
				tc.probe,       // the half-open probe
				nil,            // healthy from here on
			}}
			opts := fastOptions()
			opts.MaxAttempts = 1
			opts.FailureThreshold = 1
			opts.RecoveryTimeout = 20 * time.Millisecond
			p := Wrap(inner, opts)

			if _, err := p.Do(context.Background(), req()); err == nil {
				t.Fatal("call 1: want the 503")
			}
			time.Sleep(30 * time.Millisecond)
			if _, err := p.Do(context.Background(), req()); err == nil {
				t.Fatal("call 2: want the probe error")
			}

			// The provider is healthy now. It must become reachable.
			time.Sleep(30 * time.Millisecond)
			if _, err := p.Do(context.Background(), req()); err != nil {
				t.Fatalf("provider never recovered after a %s probe: breaker stuck in %v (err %v)",
					tc.name, p.Breaker().State(), err)
			}
		})
	}
}

// TestProvider_HalfOpenProbeSurvivesCancellation covers the other leaking
// path: a client that hangs up while holding the probe slot.
func TestProvider_HalfOpenProbeSurvivesCancellation(t *testing.T) {
	inner := &scriptedProvider{name: "p", results: []error{apiErr(503, 0)}}
	opts := fastOptions()
	opts.MaxAttempts = 1
	opts.FailureThreshold = 1
	opts.RecoveryTimeout = 20 * time.Millisecond
	p := Wrap(inner, opts)

	if _, err := p.Do(context.Background(), req()); err == nil {
		t.Fatal("call 1: want the 503")
	}
	time.Sleep(30 * time.Millisecond)

	// The probe hangs and its caller walks away.
	inner.mu.Lock()
	inner.block, inner.results = time.Hour, []error{nil}
	inner.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	_, _ = p.Do(ctx, req())
	cancel()

	inner.mu.Lock()
	inner.block = 0
	inner.mu.Unlock()

	time.Sleep(30 * time.Millisecond)
	if _, err := p.Do(context.Background(), req()); err != nil {
		t.Fatalf("provider never recovered after a cancelled probe: breaker stuck in %v (err %v)",
			p.Breaker().State(), err)
	}
}

// TestProvider_CancelledProbeDoesNotBlameTheProvider — releasing the slot
// must not tip over into recording a failure. A client hanging up says
// nothing about the upstream.
func TestProvider_CancelledProbeDoesNotBlameTheProvider(t *testing.T) {
	inner := &scriptedProvider{name: "p", block: time.Hour}
	opts := fastOptions()
	opts.MaxAttempts = 1
	opts.FailureThreshold = 2
	p := Wrap(inner, opts)

	for range 5 {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
		_, _ = p.Do(ctx, req())
		cancel()
	}

	if got := p.Breaker().State(); got != StateClosed {
		t.Errorf("breaker = %v after five cancelled calls, want closed", got)
	}
}

// TestProvider_AuthFailureOpensTheBreakerWithoutRetrying is the second
// fix: an unusable credential must leave the rotation, but retrying it
// within one request is pointless — the key will not change mid-flight.
func TestProvider_AuthFailureOpensTheBreakerWithoutRetrying(t *testing.T) {
	inner := &scriptedProvider{name: "p", results: []error{apiErr(401, 0)}}
	opts := fastOptions()
	opts.MaxAttempts = 3
	opts.FailureThreshold = 2
	p := Wrap(inner, opts)

	for i := range 2 {
		if _, err := p.Do(context.Background(), req()); err == nil {
			t.Fatalf("call %d: want the 401", i+1)
		}
	}

	if got := inner.callCount(); got != 2 {
		t.Errorf("inner called %d times for 2 requests, want 2 — a 401 must not be retried", got)
	}
	if got := p.Breaker().State(); got != StateOpen {
		t.Errorf("breaker = %v, want open — an unauthenticated provider is unusable", got)
	}
	if _, err := p.Do(context.Background(), req()); !errors.Is(err, ErrOpen) {
		t.Errorf("err = %v, want ErrOpen once the breaker has given up on the key", err)
	}
	if got := inner.callCount(); got != 2 {
		t.Errorf("inner called %d times, want 2 — the open breaker should stop the round trips", got)
	}
}

// TestBreaker_StateChangeHookFiresOutsideTheLock guards the deadlock the
// hook would cause if it were called while holding the write lock, and
// checks the transitions reported are the real ones.
func TestBreaker_StateChangeHookFiresOutsideTheLock(t *testing.T) {
	var (
		mu          sync.Mutex
		transitions []string
	)
	clock := newClock()
	var b *Breaker
	b = NewBreaker("test", 1, 30*time.Second, func(from, to State) {
		// Re-entering the breaker from the hook is the deadlock this
		// design has to survive. Under the previous shape — notify
		// called under Lock — this line hangs the test.
		_ = b.State()
		mu.Lock()
		defer mu.Unlock()
		transitions = append(transitions, from.String()+"->"+to.String())
	})
	b.now = clock.Now

	_ = b.Allow()
	b.RecordFailure() // closed -> open
	clock.Advance(30 * time.Second)
	_ = b.Allow()     // open -> half-open
	b.RecordSuccess() // half-open -> closed

	mu.Lock()
	defer mu.Unlock()
	want := []string{"closed->open", "open->half-open", "half-open->closed"}
	if len(transitions) != len(want) {
		t.Fatalf("transitions = %v, want %v", transitions, want)
	}
	for i := range want {
		if transitions[i] != want[i] {
			t.Errorf("transition %d = %q, want %q", i, transitions[i], want[i])
		}
	}
}

// TestBreaker_NoHookIsSafe — the nil check in notify is load-bearing for
// every test and caller that does not want a hook.
func TestBreaker_NoHookIsSafe(t *testing.T) {
	b, _ := newTestBreaker(1, time.Millisecond)
	_ = b.Allow()
	b.RecordFailure()
	b.RecordIndeterminate()
	b.RecordSuccess()
}
