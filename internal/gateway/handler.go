package gateway

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strconv"

	"github.com/kevinreber/llm-gateway/internal/provider"
)

// maxRequestBytes bounds an inbound completion body. LLM requests can
// legitimately be large (long system prompts, chat history) but 1 MiB
// is more than enough for v0.1.0; larger requests are almost certainly
// a bug or an abuse attempt.
const maxRequestBytes = 1 << 20 // 1 MiB

// handler owns the HTTP routes for the gateway. Phase 1 wires a single
// Anthropic provider directly; Phase 2 replaces this with an alias
// router that picks a provider per request.
type handler struct {
	anthropic provider.Provider
	logger    *slog.Logger
}

func newHandler(anthropic provider.Provider, logger *slog.Logger) *handler {
	return &handler{anthropic: anthropic, logger: logger}
}

// routes returns the http.Handler for the gateway. Kept small: one
// completion route plus healthz. /metrics and the admin API land in
// Phase 4.
func (h *handler) routes() http.Handler {
	mux := http.NewServeMux()
	// Go 1.22+ method-scoped routing — anything else on this path
	// returns 405, we don't need to handle it explicitly.
	mux.HandleFunc("POST /v1/messages", h.messages)
	mux.HandleFunc("GET /healthz", h.healthz)
	return mux
}

// messages handles POST /v1/messages: decode → route to Anthropic → return.
// Phase 2 adds alias resolution, rate limiting, and cost tracking here.
func (h *handler) messages(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	if err != nil {
		writeError(w, http.StatusBadRequest, "body_read_failed", err.Error())
		return
	}

	var req provider.Request
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_json", err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "missing_model", "model is required")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "missing_messages", "at least one message is required")
		return
	}

	if !h.anthropic.Supports(req.Model) {
		writeError(w, http.StatusBadRequest, "unsupported_model", "no provider supports model "+req.Model)
		return
	}

	resp, err := h.anthropic.Do(r.Context(), &req)
	if err != nil {
		var apiErr *provider.APIError
		if errors.As(err, &apiErr) {
			// Mirror the upstream's Retry-After when present so clients
			// get an honest signal about backoff. Rounded down to whole
			// seconds per RFC 7231; sub-second precision is not portable.
			if apiErr.RetryAfter > 0 {
				w.Header().Set("Retry-After", strconv.Itoa(int(apiErr.RetryAfter.Seconds())))
			}
			// Surface the upstream status directly. This is deliberate:
			// clients can distinguish "we did the wrong thing" (4xx) from
			// "the upstream is unhappy" (5xx) without decoding a wrapped
			// error. Retry logic (Phase 3) will care about this too.
			writeError(w, apiErr.Status, apiErr.Type, apiErr.Message)
			return
		}
		h.logger.Error("provider call failed",
			"provider", h.anthropic.Name(),
			"model", req.Model,
			"err", err)
		writeError(w, http.StatusBadGateway, "provider_error", err.Error())
		return
	}

	writeJSON(w, http.StatusOK, resp)
}

// healthz returns 200 while the process is serving. This is intentionally
// dumb — a real serving-flag flip (0 during shutdown) lands in Phase 4
// alongside the shared serve state pattern from bucketd.
func (h *handler) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

type errorBody struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func writeError(w http.ResponseWriter, status int, kind, msg string) {
	writeJSON(w, status, map[string]any{
		"error": errorBody{Type: kind, Message: msg},
	})
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
