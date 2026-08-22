package gateway

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"net/http"
)

// requestIDHeader is both read and written: an ID supplied by an
// upstream proxy is reused so one identifier spans the whole hop chain,
// and the value that ended up in our logs is echoed back so a client
// holding a failed response can quote the exact string to search for.
const requestIDHeader = "X-Request-ID"

// maxRequestIDBytes bounds an inbound X-Request-ID.
//
// The header is attacker-controlled and lands in every structured log
// line for its request. Unbounded, a client could push kilobytes into
// the log stream per request at no cost to itself; 128 bytes is well
// past any real tracing format.
const maxRequestIDBytes = 128

type contextKey int

const requestIDKey contextKey = iota

// withRequestID assigns every request an identifier and puts it in the
// context, where the handler's logging picks it up.
//
// The ID is never used as a metric label. It is unique per request by
// construction, so labelling any series with it would create one time
// series per request and take the scrape target down with it — the
// classic cardinality accident, and the reason this is worth saying out
// loud next to the code that produces the value.
func withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := sanitizeRequestID(r.Header.Get(requestIDHeader))
		if id == "" {
			id = newRequestID()
		}
		w.Header().Set(requestIDHeader, id)
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), requestIDKey, id)))
	})
}

// sanitizeRequestID returns v if it is safe to log verbatim, and ""
// otherwise, which makes the caller mint a fresh ID.
//
// Rejecting rather than escaping is deliberate. A control character in
// a client-supplied field is how log forging works — a newline splits
// one record into two, and the second one is whatever the client wrote.
// slog's JSON handler would escape it correctly today, but the value
// also reaches a response header, where a bare CR or LF is response
// splitting rather than a cosmetic problem. Refusing the input once, at
// the edge, does not depend on every consumer downstream getting its own
// encoding right.
func sanitizeRequestID(v string) string {
	if v == "" || len(v) > maxRequestIDBytes {
		return ""
	}
	for i := 0; i < len(v); i++ {
		// Printable ASCII only. Anything below 0x20 is a control
		// character and 0x7f is DEL; multi-byte UTF-8 has its high bit
		// set and is refused too, since a trace ID has no business
		// carrying it and allowing it reopens the question of what a
		// "printable" rune is.
		if v[i] < 0x20 || v[i] >= 0x7f {
			return ""
		}
	}
	return v
}

// newRequestID returns a random 128-bit identifier in hex.
//
// crypto/rand rather than math/rand: these are correlation handles that
// clients quote back and that appear in shared logs, so a caller must
// not be able to predict or collide with another caller's ID by running
// the same generator. rand.Read is documented never to fail as of Go
// 1.24, so there is no error path to handle.
func newRequestID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// requestIDFrom returns the request ID carried by ctx, or "" when there
// is none — which happens only for a handler invoked outside the
// middleware, i.e. in a unit test.
func requestIDFrom(ctx context.Context) string {
	id, _ := ctx.Value(requestIDKey).(string)
	return id
}

// log returns the request-scoped logger: h.logger with the request ID
// attached, so every line emitted while serving one request can be
// grepped as a unit.
//
// Derived per call rather than once per request because the success path
// logs nothing at all — the common case pays for zero of these — and
// stashing a second value in the context to save an allocation on the
// paths that already went wrong would be the wrong trade.
func (h *handler) log(ctx context.Context) *slog.Logger {
	if id := requestIDFrom(ctx); id != "" {
		return h.logger.With("request_id", id)
	}
	return h.logger
}
