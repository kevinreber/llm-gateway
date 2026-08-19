package cost

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"
)

const (
	// defaultBuffer bounds how many events can be in flight before
	// Track starts dropping. Sized so a 1s flush interval at a few
	// thousand requests/sec never touches the drop path.
	defaultBuffer = 4096
	// defaultBatch triggers an early flush when a burst fills the
	// buffer faster than the ticker drains it.
	defaultBatch = 100
	// defaultInterval bounds how stale the costs table can be under
	// light traffic.
	defaultInterval = time.Second
	// shutdownFlushTimeout bounds the final drain. Cancellation is
	// stripped from that flush so it isn't killed the instant SIGTERM
	// lands, which means this is the only thing standing between a
	// wedged database and a process that never exits. Kept well inside
	// the platform's kill grace period.
	shutdownFlushTimeout = 5 * time.Second
)

// Writer batches cost events and hands them to a Sink on a fixed
// cadence or whenever a batch fills, whichever comes first.
//
// Backpressure is bounded and lossy on purpose: Track never blocks, so a
// slow or wedged database degrades cost accounting instead of stalling
// the proxy. Dropped events are counted, and the count is reported at
// shutdown so the loss is visible rather than silent.
//
// Safe for concurrent use by any number of request goroutines. Exactly
// one goroutine must call Run.
type Writer struct {
	ch       chan Event
	sink     Sink
	interval time.Duration
	maxBatch int
	logger   *slog.Logger
	dropped  atomic.Int64
}

// NewWriter constructs a Writer with the default buffer, batch size, and
// flush interval.
func NewWriter(sink Sink, logger *slog.Logger) *Writer {
	return &Writer{
		ch:       make(chan Event, defaultBuffer),
		sink:     sink,
		interval: defaultInterval,
		maxBatch: defaultBatch,
		logger:   logger,
	}
}

// Track implements Tracker. It never blocks: if the buffer is full the
// event is counted as dropped and discarded.
func (w *Writer) Track(e Event) {
	select {
	case w.ch <- e:
	default:
		w.dropped.Add(1)
	}
}

// Dropped reports how many events have been discarded due to a full
// buffer since startup.
func (w *Writer) Dropped() int64 { return w.dropped.Load() }

// Run drains the buffer until ctx is cancelled, then makes a final pass
// over whatever is still queued and flushes it.
//
// The drain-then-flush on shutdown is what makes a graceful stop
// actually graceful: requests that completed during the HTTP drain
// window still get their cost rows. Run returns only after that final
// flush, so callers should wait on it before closing the database pool.
func (w *Writer) Run(ctx context.Context) {
	tick := time.NewTicker(w.interval)
	defer tick.Stop()

	batch := make([]Event, 0, w.maxBatch)

	for {
		select {
		case <-ctx.Done():
			// Detach from the cancelled context so the final write isn't
			// killed on arrival, but keep a deadline: a database that
			// hangs rather than fails would otherwise block Run forever
			// and the process would have to be killed, losing exactly
			// the events this drain exists to save.
			flushCtx, cancel := context.WithTimeout(
				context.WithoutCancel(ctx), shutdownFlushTimeout)

			// Pull everything already buffered. Producers are gone by
			// now (the HTTP server drained first), so this terminates.
			for {
				select {
				case e := <-w.ch:
					batch = append(batch, e)
					if len(batch) >= w.maxBatch {
						batch = w.flush(flushCtx, batch)
					}
					continue
				default:
				}
				break
			}
			w.flush(flushCtx, batch)
			cancel()

			if n := w.dropped.Load(); n > 0 {
				w.logger.Warn("cost events dropped", "count", n)
			}
			return

		case e := <-w.ch:
			batch = append(batch, e)
			if len(batch) >= w.maxBatch {
				batch = w.flush(ctx, batch)
			}

		case <-tick.C:
			batch = w.flush(ctx, batch)
		}
	}
}

// flush writes the batch and returns it truncated for reuse. A failed
// write is logged and the batch dropped: retrying would grow the buffer
// behind a database that is already unhealthy, and cost rows are
// accounting data, not something the request depends on.
func (w *Writer) flush(ctx context.Context, batch []Event) []Event {
	if len(batch) == 0 {
		return batch
	}
	if err := w.sink.InsertCosts(ctx, batch); err != nil {
		w.dropped.Add(int64(len(batch)))
		w.logger.Error("cost batch write failed", "events", len(batch), "err", err)
	}
	return batch[:0]
}
