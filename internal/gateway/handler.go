package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/kevinreber/llm-gateway/internal/config"
	"github.com/kevinreber/llm-gateway/internal/cost"
	"github.com/kevinreber/llm-gateway/internal/provider"
	"github.com/kevinreber/llm-gateway/internal/ratelimit"
)

// maxRequestBytes bounds an inbound completion body. LLM requests can
// legitimately be large (long system prompts, chat history) but 1 MiB
// is more than enough for v0.1.0; larger requests are almost certainly
// a bug or an abuse attempt.
const maxRequestBytes = 1 << 20 // 1 MiB

// handler owns the HTTP routes for the gateway. Phase 3 adds retry and
// circuit-breaker fallback between resolution and the provider call.
type handler struct {
	providers map[string]provider.Provider
	cfg       *config.Config
	limiter   ratelimit.Limiter
	costs     cost.Tracker
	logger    *slog.Logger
}

// route is the outcome of resolving a client-supplied model name.
type route struct {
	// alias is the name the client asked for when it named an alias,
	// and empty when it named a concrete model directly. It is the rate
	// limit key and the cost-attribution label.
	alias    string
	provider provider.Provider
	model    string
}

// routes returns the http.Handler for the gateway. /metrics and the
// admin API land in Phase 4.
func (h *handler) routes() http.Handler {
	mux := http.NewServeMux()
	// Go 1.22+ method-scoped routing — anything else on this path
	// returns 405, we don't need to handle it explicitly.
	mux.HandleFunc("POST /v1/messages", h.messages)
	mux.HandleFunc("GET /healthz", h.healthz)
	return mux
}

// messages handles POST /v1/messages: decode, resolve the alias, take a
// rate-limit token, call the provider, record what it cost.
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

	rt, err := h.resolve(req.Model)
	if err != nil {
		writeError(w, http.StatusBadRequest, "unresolvable_model", err.Error())
		return
	}

	if !h.allow(r.Context(), w, rt) {
		return
	}

	// Send the resolved model upstream, not the alias the client typed.
	req.Model = rt.model

	resp, err := rt.provider.Do(r.Context(), &req)
	if err != nil {
		h.writeProviderError(w, rt, err)
		return
	}

	h.trackCost(rt, resp)

	// Tell the client what actually served the request. Without this an
	// alias is a black box — you can't tell `smart` routed to Anthropic
	// from `smart` falling back to Ollama (Phase 3) by reading the body.
	w.Header().Set("X-Gateway-Provider", rt.provider.Name())
	w.Header().Set("X-Gateway-Model", rt.model)
	if rt.alias != "" {
		w.Header().Set("X-Gateway-Alias", rt.alias)
	}
	writeJSON(w, http.StatusOK, resp)
}

// resolve maps the client's `model` onto a provider and concrete model.
//
// Aliases win over direct model names: if gateway.yaml defines `smart`,
// a client asking for `smart` always goes where the config says. A name
// that isn't an alias falls through to "which provider claims to support
// this model", which is what preserves the Phase 1 passthrough behavior
// for callers that name `claude-sonnet-5` outright.
func (h *handler) resolve(name string) (route, error) {
	if alias, ok := h.cfg.Resolve(name); ok {
		p, ok := h.providers[alias.Provider]
		if !ok {
			// Config names a provider this build doesn't have wired.
			return route{}, fmt.Errorf("alias %q maps to unknown provider %q", name, alias.Provider)
		}
		if !p.Supports(alias.Model) {
			// Catches a YAML edit like `smart: {provider: anthropic,
			// model: gpt-4o}` at request time, loudly, instead of
			// forwarding a request the upstream will reject.
			return route{}, fmt.Errorf("alias %q maps to model %q, which provider %q does not serve",
				name, alias.Model, alias.Provider)
		}
		return route{alias: name, provider: p, model: alias.Model}, nil
	}

	for _, p := range h.providers {
		if p.Supports(name) {
			return route{provider: p, model: name}, nil
		}
	}
	return route{}, fmt.Errorf("no alias or provider serves model %q", name)
}

// allow applies the alias's rate limit, writing a 429 and returning
// false when the request is denied.
//
// Two deliberate policies live here. Requests without a configured
// limit are unlimited rather than denied — an alias with no `ratelimits`
// entry should serve traffic, not fail closed on a config omission. And
// a limiter *error* (bucketd unreachable, RPC deadline) fails open: a
// rate limiter outage should degrade enforcement, not take the gateway
// down with it. That is the right trade for a cost-control limiter; a
// limiter guarding correctness rather than spend would want the reverse.
func (h *handler) allow(ctx context.Context, w http.ResponseWriter, rt route) bool {
	if rt.alias == "" {
		return true
	}
	limit, ok := h.cfg.LimitFor(rt.alias)
	if !ok {
		return true
	}

	verdict, err := h.limiter.Allow(ctx, "alias:"+rt.alias, ratelimit.Limit{
		Capacity:   limit.Capacity,
		RefillRate: limit.RefillRate,
	})
	if err != nil {
		h.logger.Error("rate limiter unavailable, allowing request",
			"alias", rt.alias, "err", err)
		return true
	}
	if verdict.Allowed {
		return true
	}

	if verdict.RetryAfter > 0 {
		w.Header().Set("Retry-After", strconv.Itoa(secondsCeil(verdict.RetryAfter)))
	}
	w.Header().Set("X-Gateway-Alias", rt.alias)
	writeError(w, http.StatusTooManyRequests, "rate_limited",
		"rate limit exceeded for alias "+rt.alias)
	return false
}

// trackCost records a completed request. The provider's reported model
// is preferred over the requested one: Anthropic echoes back the model
// that actually served the request, and billing should follow that.
func (h *handler) trackCost(rt route, resp *provider.Response) {
	model := resp.Model
	if model == "" {
		model = rt.model
	}

	cents, known := cost.Cents(model, resp.Usage.InputTokens, resp.Usage.OutputTokens)
	if !known {
		// Zero-cost rows for an unpriced model would quietly understate
		// spend, so say it out loud — the fix is a pricing table entry.
		h.logger.Warn("no price for model; recording zero cost",
			"provider", rt.provider.Name(), "model", model)
	}

	h.costs.Track(cost.Event{
		TS:           time.Now().UTC(),
		Provider:     rt.provider.Name(),
		Model:        model,
		Alias:        rt.alias,
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		CostCents:    cents,
	})
}

func (h *handler) writeProviderError(w http.ResponseWriter, rt route, err error) {
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
		"provider", rt.provider.Name(),
		"model", rt.model,
		"alias", rt.alias,
		"err", err)
	writeError(w, http.StatusBadGateway, "provider_error", err.Error())
}

// healthz returns 200 while the process is serving. This is intentionally
// dumb — a real serving-flag flip (0 during shutdown) lands in Phase 4
// alongside the shared serve state pattern from bucketd.
func (h *handler) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// secondsCeil rounds up so a sub-second wait advertises 1 rather than 0;
// telling a client to retry after zero seconds is telling it to hammer.
func secondsCeil(d time.Duration) int {
	return int((d + time.Second - 1) / time.Second)
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
