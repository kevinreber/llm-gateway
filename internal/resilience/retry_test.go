package resilience

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/provider"
)

// scriptedProvider returns a queued sequence of outcomes, then repeats
// the last one. Recording every call's context lets a test assert that
// attempts carry a deadline rather than trusting that they do.
type scriptedProvider struct {
	mu       sync.Mutex
	name     string
	results  []error
	calls    int
	deadline []time.Duration
	block    time.Duration
}

func (s *scriptedProvider) Name() string { return s.name }

func (s *scriptedProvider) Supports(string) bool { return true }

func (s *scriptedProvider) Health(context.Context) error { return nil }

func (s *scriptedProvider) Do(ctx context.Context, _ *provider.Request) (*provider.Response, error) {
	s.mu.Lock()
	n := s.calls
	s.calls++
	if dl, ok := ctx.Deadline(); ok {
		s.deadline = append(s.deadline, time.Until(dl))
	} else {
		s.deadline = append(s.deadline, 0)
	}
	var err error
	switch {
	case len(s.results) == 0:
		err = nil
	case n < len(s.results):
		err = s.results[n]
	default:
		err = s.results[len(s.results)-1]
	}
	block := s.block
	s.mu.Unlock()

	if block > 0 {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("call %s: %w", s.name, ctx.Err())
		case <-time.After(block):
		}
	}
	if err != nil {
		return nil, err
	}
	return &provider.Response{Content: "ok", Model: "m"}, nil
}

func (s *scriptedProvider) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

func apiErr(status int, retryAfter time.Duration) *provider.APIError {
	return &provider.APIError{
		Provider:   "test",
		Status:     status,
		Type:       "test_error",
		Message:    "boom",
		RetryAfter: retryAfter,
	}
}

func fastOptions() Options {
	return Options{
		MaxAttempts:     3,
		BaseBackoff:     time.Millisecond,
		MaxBackoff:      2 * time.Millisecond,
		AttemptTimeout:  time.Second,
		Budget:          2 * time.Second,
		RecoveryTimeout: 50 * time.Millisecond,
	}
}

func req() *provider.Request {
	return &provider.Request{
		Model:    "m",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	}
}

func TestClassify(t *testing.T) {
	tests := []struct {
		name          string
		err           error
		wantRetry     bool
		wantUnhealthy bool
	}{
		{"nil", nil, false, false},
		{"500", apiErr(500, 0), true, true},
		{"502", apiErr(502, 0), true, true},
		{"503", apiErr(503, 0), true, true},
		{
			// A rate-limited provider is a healthy provider saying slow
			// down. Tripping the breaker on it would take a working
			// provider out of rotation because one alias was noisy.
			"429 retries but stays healthy", apiErr(429, 0), true, false,
		},
		{"408", apiErr(408, 0), true, true},
		{"400", apiErr(400, 0), false, false},
		{"401", apiErr(401, 0), false, false},
		{"404", apiErr(404, 0), false, false},
		{"413 context too long", apiErr(413, 0), false, false},
		{
			// Never reached a response: dial failure, reset, TLS. All
			// transient, all the provider failing to serve us.
			"transport error", &net.OpError{Op: "dial", Err: errors.New("refused")}, true, true,
		},
		{"wrapped transport error", fmt.Errorf("call anthropic: %w", errors.New("EOF")), true, true},
		{
			// Our bug, not the provider's, and identical next time.
			"invalid request", fmt.Errorf("%w: model is required", provider.ErrInvalidRequest), false, false,
		},
		{"wrapped api error", fmt.Errorf("layer: %w", apiErr(500, 0)), true, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			retry, unhealthy := classify(tc.err)
			if retry != tc.wantRetry {
				t.Errorf("retry = %v, want %v", retry, tc.wantRetry)
			}
			if unhealthy != tc.wantUnhealthy {
				t.Errorf("unhealthy = %v, want %v", unhealthy, tc.wantUnhealthy)
			}
		})
	}
}

func TestShouldFallback(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"open breaker is the whole point", &OpenError{Provider: "p", RetryAfter: time.Second}, true},
		{"500", apiErr(500, 0), true},
		{"429", apiErr(429, 0), true},
		{"400 fails the same way everywhere", apiErr(400, 0), false},
		{"401", apiErr(401, 0), false},
		{"invalid request", fmt.Errorf("%w: nope", provider.ErrInvalidRequest), false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := ShouldFallback(tc.err); got != tc.want {
				t.Errorf("ShouldFallback(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestProvider_RetriesTransientThenSucceeds(t *testing.T) {
	inner := &scriptedProvider{name: "p", results: []error{apiErr(503, 0), apiErr(503, 0), nil}}
	p := Wrap(inner, fastOptions())

	resp, err := p.Do(context.Background(), req())
	if err != nil {
		t.Fatalf("Do = %v, want success on the third attempt", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want ok", resp.Content)
	}
	if got := inner.callCount(); got != 3 {
		t.Errorf("calls = %d, want 3", got)
	}
	// The two failures then a success leaves the breaker closed: the
	// success is what resets the consecutive count.
	if got := p.Breaker().State(); got != StateClosed {
		t.Errorf("breaker = %v, want closed", got)
	}
}

func TestProvider_DoesNotRetryClientErrors(t *testing.T) {
	inner := &scriptedProvider{name: "p", results: []error{apiErr(400, 0)}}
	p := Wrap(inner, fastOptions())

	if _, err := p.Do(context.Background(), req()); err == nil {
		t.Fatal("Do = nil, want the 400 surfaced")
	}
	if got := inner.callCount(); got != 1 {
		t.Errorf("calls = %d, want 1 — a 400 replays identically", got)
	}
	if got := p.Breaker().State(); got != StateClosed {
		t.Errorf("breaker = %v, want closed — a 400 is not a provider fault", got)
	}
}

func TestProvider_StopsAtMaxAttempts(t *testing.T) {
	inner := &scriptedProvider{name: "p", results: []error{apiErr(503, 0)}}
	opts := fastOptions()
	opts.FailureThreshold = 100 // keep the breaker out of this test
	p := Wrap(inner, opts)

	if _, err := p.Do(context.Background(), req()); err == nil {
		t.Fatal("Do = nil, want failure")
	}
	if got := inner.callCount(); got != 3 {
		t.Errorf("calls = %d, want 3 (MaxAttempts)", got)
	}
}

func TestProvider_TripsBreakerAndThenRefusesToCall(t *testing.T) {
	inner := &scriptedProvider{name: "p", results: []error{apiErr(503, 0)}}
	opts := fastOptions()
	opts.FailureThreshold = 2
	p := Wrap(inner, opts)

	// Three attempts are allowed, but the breaker trips after two
	// failures and the third is never made.
	_, err := p.Do(context.Background(), req())
	if err == nil {
		t.Fatal("Do = nil, want failure")
	}
	// The caller gets the 503 that caused the trip, not "breaker open" —
	// the symptom is useless for diagnosing the request.
	var apiError *provider.APIError
	if !errors.As(err, &apiError) {
		t.Errorf("err = %v, want the underlying APIError", err)
	}
	if got := inner.callCount(); got != 2 {
		t.Errorf("calls = %d, want 2 — the breaker should stop the third", got)
	}
	if got := p.Breaker().State(); got != StateOpen {
		t.Fatalf("breaker = %v, want open", got)
	}

	// A fresh request now short-circuits without touching the provider.
	before := inner.callCount()
	_, err = p.Do(context.Background(), req())
	if !errors.Is(err, ErrOpen) {
		t.Errorf("err = %v, want ErrOpen", err)
	}
	if got := inner.callCount(); got != before {
		t.Errorf("provider called while the breaker was open (%d -> %d)", before, got)
	}
}

func TestProvider_RecoversAfterRecoveryTimeout(t *testing.T) {
	inner := &scriptedProvider{name: "p", results: []error{apiErr(503, 0), apiErr(503, 0), nil}}
	opts := fastOptions()
	opts.FailureThreshold = 2
	opts.RecoveryTimeout = 20 * time.Millisecond
	p := Wrap(inner, opts)

	if _, err := p.Do(context.Background(), req()); err == nil {
		t.Fatal("Do = nil, want the initial failure")
	}
	if got := p.Breaker().State(); got != StateOpen {
		t.Fatalf("breaker = %v, want open", got)
	}

	time.Sleep(30 * time.Millisecond)

	resp, err := p.Do(context.Background(), req())
	if err != nil {
		t.Fatalf("Do after recovery = %v, want success", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want ok", resp.Content)
	}
	if got := p.Breaker().State(); got != StateClosed {
		t.Errorf("breaker = %v, want closed after a successful probe", got)
	}
}

func TestProvider_EveryAttemptCarriesADeadline(t *testing.T) {
	// The review finding this phase had to not repeat: dependencies that
	// hang, not just ones that fail. An attempt with no deadline is an
	// unbounded wait no matter how many attempts are capped.
	inner := &scriptedProvider{name: "p", results: []error{apiErr(503, 0)}}
	opts := fastOptions()
	opts.FailureThreshold = 100
	opts.AttemptTimeout = 500 * time.Millisecond
	p := Wrap(inner, opts)

	_, _ = p.Do(context.Background(), req())

	inner.mu.Lock()
	defer inner.mu.Unlock()
	if len(inner.deadline) == 0 {
		t.Fatal("no attempts recorded")
	}
	for i, d := range inner.deadline {
		if d <= 0 {
			t.Errorf("attempt %d had no deadline", i)
			continue
		}
		if d > opts.AttemptTimeout {
			t.Errorf("attempt %d deadline = %v, want <= AttemptTimeout (%v)", i, d, opts.AttemptTimeout)
		}
	}
}

func TestProvider_TotalTimeIsBoundedByBudget(t *testing.T) {
	// Three attempts at one second each is three seconds unless
	// something bounds the total. This is the "per-attempt bound is not
	// a total bound" case stated as a test.
	inner := &scriptedProvider{name: "p", block: time.Hour}
	opts := fastOptions()
	opts.FailureThreshold = 100
	opts.AttemptTimeout = time.Second
	opts.Budget = 300 * time.Millisecond
	p := Wrap(inner, opts)

	start := time.Now()
	_, err := p.Do(context.Background(), req())
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Do = nil, want a timeout")
	}
	if elapsed > time.Second {
		t.Errorf("Do took %v, want it bounded by Budget (%v)", elapsed, opts.Budget)
	}
}

func TestProvider_LongRetryAfterIsDeclinedNotSlept(t *testing.T) {
	// The regression this pins: a 429 carrying Retry-After: 30 used to
	// park the request for 30 seconds and then do it again, turning a
	// fast actionable 429 into a minute-long hang. The upstream is
	// telling us to route elsewhere; the answer is to stop, not to wait.
	//
	// Budget is deliberately far larger than the advertised wait, so the
	// only thing that can cut this short is the Retry-After cap itself.
	inner := &scriptedProvider{name: "p", results: []error{apiErr(429, 30*time.Second)}}
	opts := fastOptions()
	opts.FailureThreshold = 100
	opts.Budget = time.Minute
	p := Wrap(inner, opts)

	start := time.Now()
	_, err := p.Do(context.Background(), req())
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Do = nil, want the 429 surfaced")
	}
	if elapsed > time.Second {
		t.Errorf("Do took %v — a %v Retry-After was slept on", elapsed, 30*time.Second)
	}
	if got := inner.callCount(); got != 1 {
		t.Errorf("calls = %d, want 1 — the wait was declined, not shortened", got)
	}
}

func TestProvider_HonorsRetryAfterButNotBeyondBudget(t *testing.T) {
	// A wait under the cap but over the remaining budget is refused too.
	// Both limits have to hold; either one alone leaves a hole.
	inner := &scriptedProvider{name: "p", results: []error{apiErr(429, 2*time.Second)}}
	opts := fastOptions()
	opts.FailureThreshold = 100
	opts.Budget = 500 * time.Millisecond
	p := Wrap(inner, opts)

	start := time.Now()
	_, err := p.Do(context.Background(), req())
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Do = nil, want the 429 surfaced")
	}
	if elapsed > time.Second {
		t.Errorf("Do took %v — the 2s Retry-After was honored past the 500ms budget", elapsed)
	}
	if got := inner.callCount(); got != 1 {
		t.Errorf("calls = %d, want 1 — there was no budget for a second attempt", got)
	}
}

func TestProvider_ShortRetryAfterIsHonored(t *testing.T) {
	// The other half of the previous test: a wait we *can* afford is
	// respected rather than replaced by our own shorter backoff.
	inner := &scriptedProvider{name: "p", results: []error{apiErr(429, 120*time.Millisecond), nil}}
	opts := fastOptions()
	opts.FailureThreshold = 100
	p := Wrap(inner, opts)

	start := time.Now()
	if _, err := p.Do(context.Background(), req()); err != nil {
		t.Fatalf("Do = %v, want success on the retry", err)
	}
	elapsed := time.Since(start)

	if elapsed < 100*time.Millisecond {
		t.Errorf("Do took %v — the 120ms Retry-After was ignored in favor of the 1ms backoff", elapsed)
	}
}

func TestProvider_ClientCancellationStopsImmediately(t *testing.T) {
	// A client that hung up is not evidence about the provider, and
	// there is nothing left to retry into.
	inner := &scriptedProvider{name: "p", block: time.Hour}
	opts := fastOptions()
	opts.FailureThreshold = 1
	p := Wrap(inner, opts)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	if _, err := p.Do(ctx, req()); err == nil {
		t.Fatal("Do = nil, want cancellation")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("Do took %v after cancellation, want prompt return", elapsed)
	}
	if got := inner.callCount(); got != 1 {
		t.Errorf("calls = %d, want 1 — a cancelled request must not retry", got)
	}
	if got := p.Breaker().State(); got != StateClosed {
		t.Errorf("breaker = %v, want closed — the client hanging up is not the provider's fault", got)
	}
}

func TestProvider_InvalidRequestNeitherRetriesNorTrips(t *testing.T) {
	inner := &scriptedProvider{
		name:    "p",
		results: []error{fmt.Errorf("%w: model is required", provider.ErrInvalidRequest)},
	}
	opts := fastOptions()
	opts.FailureThreshold = 1
	p := Wrap(inner, opts)

	if _, err := p.Do(context.Background(), req()); !errors.Is(err, provider.ErrInvalidRequest) {
		t.Fatalf("Do = %v, want ErrInvalidRequest", err)
	}
	if got := inner.callCount(); got != 1 {
		t.Errorf("calls = %d, want 1", got)
	}
	// The failing shape: a bug in one caller taking a healthy provider
	// out of rotation for everyone.
	if got := p.Breaker().State(); got != StateClosed {
		t.Errorf("breaker = %v, want closed", got)
	}
}

func TestProvider_PassesThroughIdentity(t *testing.T) {
	// Nothing downstream — metrics labels, config lookups, cost rows,
	// the X-Gateway-Provider header — may be able to tell it is talking
	// to a wrapper.
	inner := &scriptedProvider{name: "anthropic"}
	p := Wrap(inner, Options{})

	if got := p.Name(); got != "anthropic" {
		t.Errorf("Name = %q, want anthropic", got)
	}
	if !p.Supports("anything") {
		t.Error("Supports did not pass through")
	}
	if err := p.Health(context.Background()); err != nil {
		t.Errorf("Health = %v, want nil", err)
	}
}

func TestProvider_HealthIgnoresTheBreaker(t *testing.T) {
	// Health answers "is this provider reachable". Gating it on the
	// breaker would make an open breaker report the provider as
	// unreachable forever, with nothing able to contradict it.
	inner := &scriptedProvider{name: "p", results: []error{apiErr(503, 0)}}
	opts := fastOptions()
	opts.FailureThreshold = 1
	p := Wrap(inner, opts)

	_, _ = p.Do(context.Background(), req())
	if got := p.Breaker().State(); got != StateOpen {
		t.Fatalf("breaker = %v, want open", got)
	}
	if err := p.Health(context.Background()); err != nil {
		t.Errorf("Health = %v while the breaker was open, want nil", err)
	}
}

func TestOptions_ZeroValueIsUsable(t *testing.T) {
	got := Options{}.withDefaults()
	want := Options{
		FailureThreshold: DefaultFailureThreshold,
		RecoveryTimeout:  DefaultRecoveryTimeout,
		MaxAttempts:      DefaultMaxAttempts,
		BaseBackoff:      DefaultBaseBackoff,
		MaxBackoff:       DefaultMaxBackoff,
		AttemptTimeout:   DefaultAttemptTimeout,
		Budget:           DefaultBudget,
	}
	if got != want {
		t.Errorf("withDefaults() = %+v, want %+v", got, want)
	}
}

func TestBackoff_StaysWithinMaxAndNeverZero(t *testing.T) {
	// Half jitter, not full: full jitter can return a near-zero wait,
	// which under a synchronized failure puts the retry straight back on
	// the wire.
	p := Wrap(&scriptedProvider{name: "p"}, Options{
		BaseBackoff: 100 * time.Millisecond,
		MaxBackoff:  400 * time.Millisecond,
	})

	for attempt := 1; attempt <= 8; attempt++ {
		for range 50 {
			wait, ok := p.backoff(context.Background(), attempt, errors.New("x"))
			if !ok {
				t.Fatalf("attempt %d: backoff refused with no deadline set", attempt)
			}
			if wait <= 0 {
				t.Fatalf("attempt %d: wait = %v, want > 0", attempt, wait)
			}
			if wait > p.maxBackoff {
				t.Fatalf("attempt %d: wait = %v, want <= %v", attempt, wait, p.maxBackoff)
			}
		}
	}
}

func TestSleepCtx_ReturnsOnCancel(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	start := time.Now()
	err := sleepCtx(ctx, time.Hour)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("sleepCtx = nil, want the context error")
	}
	if elapsed > time.Second {
		t.Errorf("sleepCtx took %v — the backoff wait is not bounded by the context", elapsed)
	}
}
