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
	"sync/atomic"
	"time"

	"github.com/kevinreber/llm-gateway/internal/config"
	"github.com/kevinreber/llm-gateway/internal/cost"
	"github.com/kevinreber/llm-gateway/internal/observe"
	"github.com/kevinreber/llm-gateway/internal/provider"
	"github.com/kevinreber/llm-gateway/internal/ratelimit"
	"github.com/kevinreber/llm-gateway/internal/resilience"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// maxRequestBytes bounds an inbound completion body. LLM requests can
// legitimately be large (long system prompts, chat history) but 1 MiB
// is more than enough for v0.1.0; larger requests are almost certainly
// a bug or an abuse attempt.
const maxRequestBytes = 1 << 20 // 1 MiB

// limiterTimeout bounds the rate-limit check.
//
// Without it a bucketd node that is slow rather than down would stall
// the request for as long as the client waits, which inverts the whole
// point of the fail-open policy: a degraded limiter is supposed to stop
// mattering, not become the slowest thing in the request path. On
// timeout we take the fail-open branch and serve.
const limiterTimeout = 250 * time.Millisecond

// defaultCallBudget bounds the entire provider phase of a request: every
// attempt, every backoff, and every fallback hop combined.
//
// The per-provider budget in internal/resilience bounds one provider's
// share, but N providers in a chain would otherwise multiply it, and
// "bounded per hop" is not the same as bounded. This is the number that
// makes the whole path finite, and it is the one that has to stay below
// the server's WriteTimeout — a request that outlives the write deadline
// is one the client never gets an answer to no matter how correct the
// answer was.
const defaultCallBudget = 150 * time.Second

// handler owns the HTTP routes for the gateway.
type handler struct {
	providers map[string]provider.Provider
	// providerOrder fixes the order resolve() tries providers when the
	// caller names a concrete model. Ranging the map directly would be
	// nondeterministic per request, so once two providers can serve
	// overlapping model names (a local Ollama mirror and a hosted one,
	// say) identical requests would route differently with no way to
	// reproduce it.
	providerOrder []string
	cfg           *config.Config
	limiter       ratelimit.Limiter
	costs         cost.Tracker
	logger        *slog.Logger
	// callBudget overrides defaultCallBudget. Zero means "no budget of
	// our own", which leaves the request bounded by the client's context
	// and the per-provider budgets rather than by nothing.
	callBudget time.Duration
	// serving is owned by Run and flipped to 0 when shutdown begins, so
	// /healthz reports 503 while in-flight requests drain. Nil means
	// "always serving", which is what a handler constructed directly in
	// a unit test gets.
	serving *atomic.Int32
}

// requestObs accumulates the metric labels for one request.
//
// It is initialized to the most common early exit — rejected before
// routing — and narrowed as the request gets further. A return path
// added later that forgets to narrow it therefore degrades to an honest
// bad_request rather than to no observation at all, which is the failure
// mode that matters: a missing count is invisible, a wrong one is not.
type requestObs struct {
	alias    string
	provider string
	result   string
	// reached records whether the request got as far as the provider
	// phase, which decides whether it belongs in the latency histogram.
	// See observe.ObserveRequestDuration.
	reached bool
}

func (o requestObs) record(start time.Time) {
	observe.RecordRequest(o.alias, o.provider, o.result)
	if o.reached {
		observe.ObserveRequestDuration(o.alias, o.provider, time.Since(start))
	}
}

// labelOrNone renders an empty alias as the explicit "none" label; see
// observe.LabelNone for why empty label values are avoided.
func labelOrNone(v string) string {
	if v == "" {
		return observe.LabelNone
	}
	return v
}

// route is one attempt: a concrete provider and model, plus the labels
// that follow the request into logs, headers, and the costs table.
type route struct {
	// alias is the name the client asked for when it named an alias,
	// and empty when it named a concrete model directly. It is the rate
	// limit key and the cost-attribution label, and it stays fixed
	// across fallback — a request that asked for `smart` is `smart`
	// traffic no matter which alias ended up serving it.
	alias string
	// via is the fallback alias actually serving, empty on the primary
	// route. It is what makes a fallback visible without the reader
	// having to know the config: provider and model tell you where the
	// request went, via tells you that going there was a fallback.
	via      string
	provider provider.Provider
	model    string
}

// routes returns the http.Handler for the gateway. The admin API lands
// with the rest of Phase 4.
func (h *handler) routes() http.Handler {
	mux := http.NewServeMux()
	// Go 1.22+ method-scoped routing — anything else on this path
	// returns 405, we don't need to handle it explicitly.
	mux.HandleFunc("POST /v1/messages", h.messages)
	mux.HandleFunc("GET /healthz", h.healthz)
	mux.Handle("GET /metrics", h.metrics())
	return withRequestID(mux)
}

// metrics serves the Prometheus exposition.
//
// Breaker gauges are refreshed here, at scrape time, rather than pushed
// from the breaker's state-change hook. A breaker that has been open
// long enough to admit a probe reports half-open without any transition
// having fired, because no request has driven one — so a hook-driven
// gauge would sit on "open" indefinitely for an idle provider that is in
// fact ready. Reading the state on the way out means the exported value
// cannot disagree with what the request path sees.
func (h *handler) metrics() http.Handler {
	prom := promhttp.Handler()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.refreshBreakerGauges()
		prom.ServeHTTP(w, r)
	})
}

// refreshBreakerGauges publishes the current breaker state for every
// wired provider.
//
// The type assertion is what keeps the registry typed as the plain
// provider interface: only Run knows the values in it are wrapped. A
// provider without a breaker is skipped rather than reported as zero,
// since zero is "closed" and claiming a circuit is closed when there is
// no circuit would be a worse answer than no answer.
func (h *handler) refreshBreakerGauges() {
	for _, name := range h.providerOrder {
		b, ok := h.providers[name].(interface{ Breaker() *resilience.Breaker })
		if !ok {
			continue
		}
		observe.SetBreakerState(name, breakerGaugeValue(b.Breaker().State()))
	}
}

// breakerGaugeValue maps a breaker state onto its gauge encoding.
func breakerGaugeValue(s resilience.State) float64 {
	switch s {
	case resilience.StateOpen:
		return observe.BreakerOpen
	case resilience.StateHalfOpen:
		return observe.BreakerHalfOpen
	default:
		return observe.BreakerClosed
	}
}

// messages handles POST /v1/messages: decode, resolve the alias, take a
// rate-limit token, call the provider, record what it cost.
func (h *handler) messages(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	obs := requestObs{
		alias:    observe.LabelNone,
		provider: observe.LabelNone,
		result:   observe.ResultBadRequest,
	}
	// Recorded by defer rather than at each return, for the same reason
	// the breaker reports its outcome by defer: the inline version has
	// to be repeated at every exit, and the exits that get missed are
	// the error paths nobody is looking at.
	defer func() { obs.record(start) }()

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
	obs.alias = labelOrNone(rt.alias)
	obs.provider = rt.provider.Name()

	if !h.allow(r.Context(), w, rt) {
		obs.result = observe.ResultRateLimited
		return
	}

	callCtx, cancel := h.withCallBudget(r.Context())
	defer cancel()

	// Routed and admitted: from here the elapsed time is provider time,
	// which is what the latency histogram is for.
	obs.reached = true

	served, resp, err := h.call(callCtx, rt, &req)
	if err != nil {
		obs.result = h.writeProviderError(r.Context(), w, rt, err)
		return
	}
	// Count the request against whoever actually answered. Attributing a
	// fallback to the primary would let the replacement's successes hide
	// the primary's error rate, which is the one number this counter
	// exists to expose.
	obs.provider = served.provider.Name()
	obs.result = observe.ResultOK

	h.trackCost(r.Context(), served, resp)

	// Tell the client what actually served the request. Without this an
	// alias is a black box — you can't tell `smart` routed to Anthropic
	// from `smart` falling back to OpenAI by reading the body. These
	// describe the route that produced this response, not the one the
	// client asked for, which is the whole reason they are worth having.
	w.Header().Set("X-Gateway-Provider", served.provider.Name())
	w.Header().Set("X-Gateway-Model", served.model)
	if served.alias != "" {
		w.Header().Set("X-Gateway-Alias", served.alias)
	}
	if served.via != "" {
		w.Header().Set("X-Gateway-Fallback", served.via)
	}
	writeJSON(w, http.StatusOK, resp)
}

// withCallBudget bounds the provider phase of a request.
//
// A zero budget yields a plain cancel-only context rather than an
// already-expired one. A misconfigured deadline that fails every request
// instantly is a far worse failure than a missing deadline, and the
// request still cannot run forever: the client's own context and each
// provider's budget both still apply.
func (h *handler) withCallBudget(ctx context.Context) (context.Context, context.CancelFunc) {
	if h.callBudget <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, h.callBudget)
}

// call walks the fallback chain and returns the route that served,
// alongside its response.
//
// Retry and circuit breaking happen one level down, inside the wrapped
// provider: by the time Do returns an error here, that provider has
// already been retried as much as it is going to be. This loop's only
// job is deciding whether some *other* provider deserves a turn.
func (h *handler) call(ctx context.Context, rt route, req *provider.Request) (route, *provider.Response, error) {
	chain := h.chain(ctx, rt)

	var firstErr error
	for i, hop := range chain {
		// Copy per hop: each provider needs its own resolved model, and
		// mutating the caller's request would leave the last hop's model
		// visible to anything that reads it afterwards.
		attempt := *req
		attempt.Model = hop.model

		resp, err := hop.provider.Do(ctx, &attempt)
		if err == nil {
			if i > 0 {
				h.log(ctx).Warn("request served by fallback",
					"alias", hop.alias,
					"via", hop.via,
					"provider", hop.provider.Name(),
					"model", hop.model,
					"attempts", i+1)
			}
			return hop, resp, nil
		}

		if firstErr == nil {
			firstErr = err
		}
		if i == len(chain)-1 {
			break
		}
		// The client hung up or the whole call budget is gone. Trying
		// the next provider would burn its quota on a response nobody
		// is waiting for.
		if ctx.Err() != nil {
			break
		}
		if !resilience.ShouldFallback(err) {
			// A refusal the next provider would also produce — a
			// malformed prompt, a context-length overflow. Falling back
			// turns one honest 400 into N of them and delays it.
			break
		}
		h.log(ctx).Warn("provider failed, trying fallback",
			"alias", hop.alias,
			"provider", hop.provider.Name(),
			"model", hop.model,
			"next", chain[i+1].via,
			"err", err)
	}

	// Report the primary's failure, not the last hop's: the client asked
	// for this alias, and why *it* could not be served is the answer to
	// their question. Every other hop's error was logged above.
	return rt, nil, firstErr
}

// chain returns the ordered routes to attempt for rt: the primary first,
// then each configured fallback alias resolved to its own provider and
// model.
//
// A direct model name gets no fallback. That is not an omission — a
// caller who names `claude-sonnet-5` outright has asked for that model
// specifically, and quietly answering with a different one from a
// different vendor would be the gateway lying about what it did.
func (h *handler) chain(ctx context.Context, rt route) []route {
	if rt.alias == "" {
		return []route{rt}
	}

	targets := h.cfg.FallbackFor(rt.alias)
	hops := make([]route, 0, len(targets)+1)
	hops = append(hops, rt)

	for _, name := range targets {
		fb, err := h.resolve(name)
		if err != nil {
			// Config validation proved this alias exists; it can still
			// name a provider this build doesn't have wired. Skipping is
			// right — a chain of three should not be useless because its
			// second entry is not deployed yet — and Run already warned
			// about it once at startup rather than once per request.
			h.log(ctx).Debug("skipping unroutable fallback",
				"alias", rt.alias, "fallback", name, "err", err)
			continue
		}
		hops = append(hops, route{
			alias:    rt.alias,
			via:      name,
			provider: fb.provider,
			model:    fb.model,
		})
	}
	return hops
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

	for _, providerName := range h.providerOrder {
		if p := h.providers[providerName]; p != nil && p.Supports(name) {
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
//
// This runs once per request, against the alias the client named, and
// not again for each fallback hop. The limit is a policy attached to the
// client-facing name, so one request should cost one token from it; also
// charging the fallback alias would bill a single request twice and add
// a limiter round-trip to the path that is already the degraded one.
func (h *handler) allow(ctx context.Context, w http.ResponseWriter, rt route) bool {
	if rt.alias == "" {
		return true
	}
	limit, ok := h.cfg.LimitFor(rt.alias)
	if !ok {
		return true
	}

	limitCtx, cancel := context.WithTimeout(ctx, limiterTimeout)
	defer cancel()

	verdict, err := h.limiter.Allow(limitCtx, "alias:"+rt.alias, ratelimit.Limit{
		Capacity:   limit.Capacity,
		RefillRate: limit.RefillRate,
	})
	if err != nil {
		h.log(ctx).Error("rate limiter unavailable, allowing request",
			"alias", rt.alias, "err", err)
		return true
	}
	if verdict.Allowed {
		return true
	}

	// Debug rather than Warn: for a busy alias, denial is the steady
	// state, and logging every one at Warn would flood under exactly the
	// load that makes the log worth reading. This exists so shed traffic
	// is diagnosable at all today; the denial counter in the metrics
	// work is the real answer.
	h.log(ctx).Debug("rate limit denied",
		"alias", rt.alias,
		"retry_after", verdict.RetryAfter,
		"remaining", verdict.Remaining)

	if verdict.RetryAfter > 0 {
		w.Header().Set("Retry-After", strconv.Itoa(secondsCeil(verdict.RetryAfter)))
	}
	w.Header().Set("X-Gateway-Alias", rt.alias)
	writeError(w, http.StatusTooManyRequests, "rate_limited",
		"rate limit exceeded for alias "+rt.alias)
	return false
}

// trackCost records a completed request against the route that actually
// served it.
//
// The provider's reported model is preferred over the requested one:
// both Anthropic and OpenAI echo back the model that ran, and billing
// should follow that. Fallback makes this load-bearing rather than
// merely tidy — a request that asked for Sonnet and was served by GPT-4o
// has to appear in the costs table as OpenAI/gpt-4o, at GPT-4o's rate.
//
// The alias label is the opposite case and stays as the client-facing
// name across fallback. Cost is attributed to the traffic that caused
// it, and `smart` traffic is `smart` traffic wherever it landed; the
// provider and model columns are what record where that was.
func (h *handler) trackCost(ctx context.Context, rt route, resp *provider.Response) {
	model := resp.Model
	if model == "" {
		model = rt.model
	}

	cents, known := cost.Cents(model, resp.Usage.InputTokens, resp.Usage.OutputTokens)
	if !known {
		// Zero-cost rows for an unpriced model would quietly understate
		// spend, so say it out loud — the fix is a pricing table entry.
		h.log(ctx).Warn("no price for model; recording zero cost",
			"provider", rt.provider.Name(), "model", model)
	}

	// Added even when the price is unknown, which creates a series
	// pinned at zero for that model. That is the intent: a model missing
	// from the pricing table should be visible on the cost dashboard as
	// serving traffic it cannot account for, rather than absent from it
	// and indistinguishable from a model nobody called.
	observe.AddCostCents(rt.provider.Name(), model, cents)

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

// writeProviderError writes the client-facing error and reports the
// observe result constant the request should be counted under, so the
// classification lives in exactly one place.
func (h *handler) writeProviderError(ctx context.Context, w http.ResponseWriter, rt route, err error) string {
	// Every provider in the chain is open. 503 is the honest status —
	// the gateway is fine, it has simply decided not to call anything
	// upstream — and the breaker knows exactly when it will next admit a
	// probe, so the Retry-After we advertise is a real number rather
	// than a guess.
	var openErr *resilience.OpenError
	if errors.As(err, &openErr) {
		if openErr.RetryAfter > 0 {
			w.Header().Set("Retry-After", strconv.Itoa(secondsCeil(openErr.RetryAfter)))
		}
		h.log(ctx).Warn("circuit open, refusing request",
			"provider", openErr.Provider,
			"alias", rt.alias,
			"retry_after", openErr.RetryAfter)
		writeError(w, http.StatusServiceUnavailable, "circuit_open",
			"no healthy provider for "+describeRoute(rt))
		return observe.ResultCircuitOpen
	}

	var apiErr *provider.APIError
	if errors.As(err, &apiErr) {
		// Mirror the upstream's Retry-After when present so clients get
		// an honest signal about backoff. Whole seconds per RFC 7231,
		// rounded up for the same reason as the rate-limit path: a
		// sub-second wait truncated to 0 tells an already-throttled
		// client to retry immediately.
		if apiErr.RetryAfter > 0 {
			w.Header().Set("Retry-After", strconv.Itoa(secondsCeil(apiErr.RetryAfter)))
		}
		// Surface the upstream status directly. This is deliberate:
		// clients can distinguish "we did the wrong thing" (4xx) from
		// "the upstream is unhappy" (5xx) without decoding a wrapped
		// error. Retry logic (Phase 3) will care about this too.
		writeError(w, apiErr.Status, apiErr.Type, apiErr.Message)
		return observe.ResultProviderError
	}
	h.log(ctx).Error("provider call failed",
		"provider", rt.provider.Name(),
		"model", rt.model,
		"alias", rt.alias,
		"err", err)
	writeError(w, http.StatusBadGateway, "provider_error", err.Error())
	return observe.ResultProviderError
}

// healthz reports whether this process is accepting new traffic.
//
// Run flips the flag to 0 before it starts draining, so a load balancer
// pulls the instance out of rotation while in-flight requests finish
// rather than after. The window between the two is the entire point:
// returning 200 until the listener closes guarantees a burst of
// connections refused at the TCP level, which clients see as an error
// and not as a graceful deploy.
func (h *handler) healthz(w http.ResponseWriter, _ *http.Request) {
	if h.serving != nil && h.serving.Load() == 0 {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "shutting_down"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// describeRoute names the route in client-facing error text: the alias
// when there is one, the model otherwise. The client asked in one of
// those two vocabularies and should be answered in the same one.
func describeRoute(rt route) string {
	if rt.alias != "" {
		return "alias " + rt.alias
	}
	return "model " + rt.model
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
