package resilience

import (
	"errors"
	"sync"
	"testing"
	"time"
)

// fakeClock drives the breaker's recovery timer without real sleeping.
// A test that waited out a 30s recovery window is a test nobody runs.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newClock() *fakeClock {
	return &fakeClock{t: time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

func newTestBreaker(threshold int, recovery time.Duration) (*Breaker, *fakeClock) {
	clock := newClock()
	b := NewBreaker("test", threshold, recovery, nil)
	b.now = clock.Now
	return b, clock
}

func TestBreaker_ClosedUntilThreshold(t *testing.T) {
	b, _ := newTestBreaker(3, time.Minute)

	for i := 1; i < 3; i++ {
		if err := b.Allow(); err != nil {
			t.Fatalf("failure %d: Allow = %v, want nil", i, err)
		}
		b.RecordFailure()
		if got := b.State(); got != StateClosed {
			t.Fatalf("after %d failures: state = %v, want closed", i, got)
		}
	}

	if err := b.Allow(); err != nil {
		t.Fatalf("final Allow = %v, want nil", err)
	}
	b.RecordFailure()
	if got := b.State(); got != StateOpen {
		t.Fatalf("at threshold: state = %v, want open", got)
	}
}

func TestBreaker_SuccessResetsTheCount(t *testing.T) {
	// Consecutive, not cumulative. A provider that fails twice, recovers,
	// then fails twice again is not a provider to take out of rotation.
	b, _ := newTestBreaker(3, time.Minute)

	for range 2 {
		_ = b.Allow()
		b.RecordFailure()
	}
	_ = b.Allow()
	b.RecordSuccess()
	for range 2 {
		_ = b.Allow()
		b.RecordFailure()
	}

	if got := b.State(); got != StateClosed {
		t.Fatalf("state = %v, want closed — the success should have reset the count", got)
	}
}

func TestBreaker_OpenRejectsWithoutCalling(t *testing.T) {
	b, clock := newTestBreaker(1, 30*time.Second)
	_ = b.Allow()
	b.RecordFailure()

	err := b.Allow()
	if err == nil {
		t.Fatal("Allow = nil while open, want rejection")
	}
	if !errors.Is(err, ErrOpen) {
		t.Errorf("errors.Is(err, ErrOpen) = false for %v", err)
	}

	var openErr *OpenError
	if !errors.As(err, &openErr) {
		t.Fatalf("errors.As(*OpenError) = false for %v", err)
	}
	if openErr.Provider != "test" {
		t.Errorf("Provider = %q, want test", openErr.Provider)
	}
	// The wait has to be real: it is what the gateway puts in the
	// Retry-After header on the 503.
	if openErr.RetryAfter != 30*time.Second {
		t.Errorf("RetryAfter = %v, want 30s", openErr.RetryAfter)
	}

	clock.Advance(10 * time.Second)
	err = b.Allow()
	var stillOpen *OpenError
	if !errors.As(err, &stillOpen) {
		t.Fatalf("Allow after 10s = %v, want still open", err)
	}
	if stillOpen.RetryAfter != 20*time.Second {
		t.Errorf("RetryAfter = %v, want 20s (the remaining wait, not the full timeout)", stillOpen.RetryAfter)
	}
}

func TestBreaker_HalfOpenAfterRecovery(t *testing.T) {
	b, clock := newTestBreaker(1, 30*time.Second)
	_ = b.Allow()
	b.RecordFailure()

	clock.Advance(30 * time.Second)

	if got := b.State(); got != StateHalfOpen {
		t.Fatalf("state = %v, want half-open once the timeout has elapsed", got)
	}
	if err := b.Allow(); err != nil {
		t.Fatalf("probe Allow = %v, want nil", err)
	}
}

func TestBreaker_HalfOpenSuccessCloses(t *testing.T) {
	b, clock := newTestBreaker(2, 30*time.Second)
	for range 2 {
		_ = b.Allow()
		b.RecordFailure()
	}
	clock.Advance(30 * time.Second)

	if err := b.Allow(); err != nil {
		t.Fatalf("probe Allow = %v, want nil", err)
	}
	b.RecordSuccess()

	if got := b.State(); got != StateClosed {
		t.Fatalf("state = %v, want closed", got)
	}
	// Closing must also clear the failure count, or the next single
	// failure would re-trip a breaker that just proved itself healthy.
	_ = b.Allow()
	b.RecordFailure()
	if got := b.State(); got != StateClosed {
		t.Fatalf("state = %v after one post-recovery failure, want closed", got)
	}
}

func TestBreaker_HalfOpenFailureReopensWithFreshTimer(t *testing.T) {
	b, clock := newTestBreaker(1, 30*time.Second)
	_ = b.Allow()
	b.RecordFailure()
	clock.Advance(30 * time.Second)

	if err := b.Allow(); err != nil {
		t.Fatalf("probe Allow = %v, want nil", err)
	}
	b.RecordFailure()

	if got := b.State(); got != StateOpen {
		t.Fatalf("state = %v, want open — one failed probe is enough", got)
	}
	// The timer restarts from the failed probe, not from the original
	// trip: otherwise every probe failure would be followed immediately
	// by another probe.
	clock.Advance(29 * time.Second)
	if err := b.Allow(); err == nil {
		t.Fatal("Allow = nil 29s after a failed probe, want rejection")
	}
	clock.Advance(time.Second)
	if err := b.Allow(); err != nil {
		t.Fatalf("Allow = %v 30s after a failed probe, want nil", err)
	}
}

func TestBreaker_HalfOpenAdmitsOneProbe(t *testing.T) {
	// The whole point of half-open: a provider showing its first sign of
	// life must not be hit by every request that queued up while it was
	// down.
	b, clock := newTestBreaker(1, 30*time.Second)
	_ = b.Allow()
	b.RecordFailure()
	clock.Advance(30 * time.Second)

	if err := b.Allow(); err != nil {
		t.Fatalf("first probe = %v, want nil", err)
	}
	for i := range 10 {
		if err := b.Allow(); err == nil {
			t.Fatalf("concurrent probe %d was admitted; only one may be in flight", i)
		}
	}

	// Once the probe reports back, the gate moves.
	b.RecordSuccess()
	if err := b.Allow(); err != nil {
		t.Fatalf("Allow after successful probe = %v, want nil", err)
	}
}

func TestBreaker_LateSuccessDoesNotReopenAnOpenBreaker(t *testing.T) {
	// A call that was in flight when another goroutine tripped the
	// breaker comes back successful. That is stale evidence about a
	// provider we have since decided is unhealthy.
	b, _ := newTestBreaker(1, 30*time.Second)
	_ = b.Allow()
	b.RecordFailure()

	b.RecordSuccess()

	if got := b.State(); got != StateOpen {
		t.Fatalf("state = %v, want open", got)
	}
}

func TestBreaker_ConcurrentUse(t *testing.T) {
	// Run under -race. The failing shape this guards against is the
	// read-then-write in Allow: the fast path drops RLock before
	// allowSlow takes Lock, so the state must be re-read there.
	b, _ := newTestBreaker(5, time.Millisecond)

	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for range 100 {
				if err := b.Allow(); err != nil {
					continue
				}
				if i%2 == 0 {
					b.RecordSuccess()
				} else {
					b.RecordFailure()
				}
				_ = b.State()
			}
		}(i)
	}
	wg.Wait()
}

func TestBreaker_ZeroValuesTakeDefaults(t *testing.T) {
	// A breaker built from an unset config must not trip instantly or
	// stay open forever.
	b := NewBreaker("test", 0, 0, nil)
	if b.threshold != DefaultFailureThreshold {
		t.Errorf("threshold = %d, want %d", b.threshold, DefaultFailureThreshold)
	}
	if b.recovery != DefaultRecoveryTimeout {
		t.Errorf("recovery = %v, want %v", b.recovery, DefaultRecoveryTimeout)
	}
}

func TestState_String(t *testing.T) {
	for state, want := range map[State]string{
		StateClosed:   "closed",
		StateOpen:     "open",
		StateHalfOpen: "half-open",
	} {
		if got := state.String(); got != want {
			t.Errorf("State(%d).String() = %q, want %q", int(state), got, want)
		}
	}
}
