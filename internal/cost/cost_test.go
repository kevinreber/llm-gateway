package cost_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/cost"
)

func TestCents(t *testing.T) {
	tests := []struct {
		name  string
		model string
		in    int
		out   int
		want  float64
		known bool
	}{
		{
			// Sonnet 5 lists at $3/MTok in, $15/MTok out.
			// 1M in + 1M out = $18.00 = 1800 cents.
			name:  "sonnet 5, one million each way",
			model: "claude-sonnet-5", in: 1_000_000, out: 1_000_000,
			want: 1800, known: true,
		},
		{
			// Haiku 4.5 at $1/$5. 1000 in + 500 out
			// = $0.001 + $0.0025 = $0.0035 = 0.35 cents.
			name:  "haiku 4.5, realistic request",
			model: "claude-haiku-4-5", in: 1000, out: 500,
			want: 0.35, known: true,
		},
		{
			// Dated snapshots must bill at their family's rate rather
			// than falling off the table.
			name:  "dated snapshot resolves via prefix",
			model: "claude-haiku-4-5-20251001", in: 1000, out: 500,
			want: 0.35, known: true,
		},
		{
			name:  "opus 5 at $5/$25",
			model: "claude-opus-5", in: 200_000, out: 40_000,
			want: 200, known: true,
		},
		{
			name:  "unknown model bills nothing and says so",
			model: "gpt-4o", in: 1000, out: 1000,
			want: 0, known: false,
		},
		{
			name:  "zero tokens",
			model: "claude-sonnet-5", in: 0, out: 0,
			want: 0, known: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, known := cost.Cents(tt.model, tt.in, tt.out)
			if known != tt.known {
				t.Errorf("known = %v, want %v", known, tt.known)
			}
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("Cents = %v, want %v", got, tt.want)
			}
		})
	}
}

// fakeSink records batches and can be made to fail.
type fakeSink struct {
	mu      sync.Mutex
	batches [][]cost.Event
	err     error
}

func (f *fakeSink) InsertCosts(_ context.Context, batch []cost.Event) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err != nil {
		return f.err
	}
	// Copy: Writer reuses the backing array between flushes.
	cp := make([]cost.Event, len(batch))
	copy(cp, batch)
	f.batches = append(f.batches, cp)
	return nil
}

func (f *fakeSink) total() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, b := range f.batches {
		n += len(b)
	}
	return n
}

func (f *fakeSink) sizes() []int {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]int, len(f.batches))
	for i, b := range f.batches {
		out[i] = len(b)
	}
	return out
}

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func TestWriter_BatchesBySize(t *testing.T) {
	sink := &fakeSink{}
	w := cost.NewWriter(sink, quietLogger())

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { w.Run(ctx); close(done) }()

	// 250 events with a 100-event batch trigger: expect at least two
	// size-triggered flushes before the ticker ever fires.
	for i := 0; i < 250; i++ {
		w.Track(cost.Event{Provider: "anthropic", Model: "claude-sonnet-5"})
	}

	cancel()
	<-done

	if got := sink.total(); got != 250 {
		t.Fatalf("wrote %d events, want 250", got)
	}
	sizes := sink.sizes()
	full := 0
	for _, n := range sizes {
		if n == 100 {
			full++
		}
	}
	if full < 2 {
		t.Errorf("batch sizes = %v, want at least two full 100-event batches", sizes)
	}
}

func TestWriter_FlushesOnInterval(t *testing.T) {
	sink := &fakeSink{}
	w := cost.NewWriter(sink, quietLogger())

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go w.Run(ctx)

	// Well under the batch trigger — only the ticker can flush these.
	for i := 0; i < 3; i++ {
		w.Track(cost.Event{Model: "claude-haiku-4-5"})
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if sink.total() == 3 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("ticker did not flush within 3s; wrote %d of 3", sink.total())
}

func TestWriter_DrainsOnShutdown(t *testing.T) {
	sink := &fakeSink{}
	w := cost.NewWriter(sink, quietLogger())

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { w.Run(ctx); close(done) }()

	// Queue events and cancel immediately — these are only persisted
	// if the shutdown path drains the buffer instead of returning.
	for i := 0; i < 37; i++ {
		w.Track(cost.Event{Model: "claude-opus-5"})
	}
	cancel()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("Run did not return within 3s of cancellation")
	}

	if got := sink.total(); got != 37 {
		t.Errorf("wrote %d events on shutdown, want 37", got)
	}
}

func TestWriter_DropsAreBoundedAndCounted(t *testing.T) {
	sink := &fakeSink{}
	w := cost.NewWriter(sink, quietLogger())
	// Run is deliberately not started: nothing drains the channel, so
	// everything past the buffer must be dropped rather than blocking.

	const n = 20_000
	for i := 0; i < n; i++ {
		w.Track(cost.Event{Model: "claude-sonnet-5"})
	}

	dropped := w.Dropped()
	if dropped == 0 {
		t.Fatal("no events dropped with a full buffer and no reader")
	}
	if dropped >= n {
		t.Fatalf("dropped %d of %d — the buffer accepted nothing", dropped, n)
	}
	if sink.total() != 0 {
		t.Errorf("sink received %d events with Run not started", sink.total())
	}
}

func TestWriter_SinkFailureCountsAsDropped(t *testing.T) {
	sink := &fakeSink{err: errors.New("connection refused")}
	w := cost.NewWriter(sink, quietLogger())

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { w.Run(ctx); close(done) }()

	for i := 0; i < 10; i++ {
		w.Track(cost.Event{Model: "claude-sonnet-5"})
	}
	cancel()
	<-done

	// A failed write must not be silent — it lands in the drop counter
	// so the loss shows up in the shutdown log.
	if got := w.Dropped(); got != 10 {
		t.Errorf("Dropped = %d, want 10 after a failing sink", got)
	}
}
