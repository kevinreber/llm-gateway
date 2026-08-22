// Package admin serves the gateway's operational surface: what the
// running process thinks its configuration is, what its providers are
// doing, and the Prometheus exposition.
//
// It is deliberately a separate http.Handler mounted on its own
// listener rather than more routes on the request path. Two reasons,
// and the second is the load-bearing one.
//
// The exposition discloses cumulative spend, which vendors are wired,
// and which of them are currently failing — none of which belongs on a
// port that serves clients. And POST /admin/reload changes the routing
// of live traffic while carrying no authentication whatsoever. There is
// no credential to leak because there is no credential: the security
// boundary is the network, and the whole design assumes this listener is
// bound somewhere only operators can reach. Exposing it publicly hands
// anyone who can reach it the ability to repoint the gateway's aliases.
package admin

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"sort"

	"github.com/kevinreber/llm-gateway/internal/config"
	"github.com/kevinreber/llm-gateway/internal/observe"
	"github.com/kevinreber/llm-gateway/internal/provider"
	"github.com/kevinreber/llm-gateway/internal/resilience"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// requestsMetric is the metric family /admin/stats reads per-provider
// counts out of.
//
// Reading them back from the registry rather than keeping a second set
// of counters is what guarantees /admin/stats and /metrics can never
// disagree. Two independent tallies of the same events drift the moment
// one of them is updated on a path the other missed, and the version an
// operator is looking at during an incident is whichever one they
// happened to open.
const requestsMetric = "llm_gateway_requests_total"

// DropReporter is the part of the cost writer this package needs: how
// many events it has discarded. Narrow rather than taking *cost.Writer
// so a test can supply a count without a database behind it.
type DropReporter interface {
	Dropped() int64
}

// Handler serves the admin routes.
type Handler struct {
	// Store is the live config, read for GET /admin/aliases and
	// replaced by POST /admin/reload.
	Store *config.Store
	// Providers is the wired registry, in Order. Values are expected to
	// be resilience-wrapped; one that is not simply reports no breaker.
	Providers map[string]provider.Provider
	Order     []string
	// Costs reports dropped cost events. Optional.
	Costs  DropReporter
	Logger *slog.Logger
	// Gatherer defaults to prometheus.DefaultGatherer.
	Gatherer prometheus.Gatherer

	// publishedDrops is the drop count already reflected in the metrics
	// counter. The writer exposes a running total and a Prometheus
	// counter only moves by addition, so a scrape publishes the delta
	// since it last looked.
	publishedDrops int64
}

// Routes returns the handler for the admin listener.
func (h *Handler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /admin/aliases", h.aliases)
	mux.HandleFunc("GET /admin/stats", h.stats)
	mux.HandleFunc("POST /admin/reload", h.reload)
	mux.Handle("GET /metrics", h.metrics())
	return mux
}

func (h *Handler) gatherer() prometheus.Gatherer {
	if h.Gatherer != nil {
		return h.Gatherer
	}
	return prometheus.DefaultGatherer
}

func (h *Handler) logger() *slog.Logger {
	if h.Logger != nil {
		return h.Logger
	}
	return slog.Default()
}

// metrics serves the Prometheus exposition.
//
// Gauges are refreshed here, at scrape time, rather than pushed from
// the breaker's state-change hook. A breaker that has been open long
// enough to admit a probe reports half-open without any transition
// having fired, because no request has driven one, so a hook-driven
// gauge would sit on "open" indefinitely for an idle provider that is in
// fact ready. Reading state on the way out means the exported value
// cannot disagree with what the request path sees.
func (h *Handler) metrics() http.Handler {
	prom := promhttp.Handler()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.RefreshGauges()
		prom.ServeHTTP(w, r)
	})
}

// RefreshGauges publishes the current breaker state for every wired
// provider and any newly dropped cost events.
//
// Exported because a scrape is not the only reason to want the gauges
// current — GET /admin/stats reads the same underlying state, and a
// deployment with no Prometheus at all should still see honest numbers
// there.
func (h *Handler) RefreshGauges() {
	for _, name := range h.Order {
		b, ok := h.breaker(name)
		if !ok {
			continue
		}
		observe.SetBreakerState(name, breakerGaugeValue(b.State()))
	}
	h.refreshCostDrops()
}

// breaker returns the circuit breaker guarding a provider.
//
// The type assertion keeps the registry typed as the plain provider
// interface: only startup knows the values in it are wrapped. A
// provider without a breaker reports none rather than zero, since zero
// is "closed" and claiming a circuit is closed when there is no circuit
// would be a worse answer than no answer.
func (h *Handler) breaker(name string) (*resilience.Breaker, bool) {
	p, ok := h.Providers[name]
	if !ok {
		return nil, false
	}
	wrapped, ok := p.(interface{ Breaker() *resilience.Breaker })
	if !ok {
		return nil, false
	}
	return wrapped.Breaker(), true
}

// refreshCostDrops publishes cost events dropped since the last look.
//
// Not atomic, and does not need to be: this runs from an http.Handler
// on the admin listener, which is a single Prometheus target scraped
// serially. A second concurrent scraper would at worst double-publish
// one interval's delta, which is a cosmetic error in a number whose
// only real question is "is it zero or not".
func (h *Handler) refreshCostDrops() {
	if h.Costs == nil {
		return
	}
	total := h.Costs.Dropped()
	if total > h.publishedDrops {
		observe.AddDroppedCostEvents(float64(total - h.publishedDrops))
		h.publishedDrops = total
	}
}

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

// aliasesResponse is the shape of GET /admin/aliases.
type aliasesResponse struct {
	// Path is the file this config came from, empty when the gateway is
	// running on defaults. It answers the first question anyone asks
	// when the routing is not what they expected, which is whether the
	// file they have been editing is the one in use.
	Path       string                    `json:"path"`
	Reloadable bool                      `json:"reloadable"`
	Aliases    map[string]config.Alias   `json:"aliases"`
	RateLimits map[string]config.Limit   `json:"ratelimits"`
	Breakers   map[string]config.Breaker `json:"breakers"`
	Fallback   map[string][]string       `json:"fallback"`
}

func (h *Handler) aliases(w http.ResponseWriter, _ *http.Request) {
	cfg := h.Store.Load()
	writeJSON(w, http.StatusOK, aliasesResponse{
		Path:       h.Store.Path(),
		Reloadable: h.Store.Reloadable(),
		Aliases:    cfg.Aliases,
		RateLimits: cfg.RateLimits,
		Breakers:   cfg.Breakers,
		Fallback:   cfg.Fallback,
	})
}

// providerStats is one provider's live state.
type providerStats struct {
	Name string `json:"name"`
	// Breaker is the state name, or "none" for a provider with no
	// breaker in front of it.
	Breaker string `json:"breaker"`
	// Healthy mirrors the provider_health gauge: would a call be
	// admitted right now. Half-open counts as healthy, because the
	// breaker is admitting a probe.
	Healthy bool `json:"healthy"`
	// Requests counts by result label, e.g. {"ok": 12}. Absent results
	// are omitted rather than zeroed, so the shape reflects what has
	// actually happened.
	Requests map[string]float64 `json:"requests,omitempty"`
}

// statsResponse is the shape of GET /admin/stats.
type statsResponse struct {
	Providers []providerStats `json:"providers"`
	// CostEventsDropped is non-zero when recorded spend is an
	// undercount, which is the one number here worth alerting on.
	CostEventsDropped int64 `json:"cost_events_dropped"`
}

func (h *Handler) stats(w http.ResponseWriter, _ *http.Request) {
	h.RefreshGauges()

	counts, err := h.requestCounts()
	if err != nil {
		// Report the live state we do have rather than failing the whole
		// endpoint. Breaker state is the part an operator is looking at
		// during an incident, and it does not come from the registry.
		h.logger().Warn("admin: could not gather request counts", "err", err)
	}

	out := statsResponse{Providers: make([]providerStats, 0, len(h.Order))}
	for _, name := range h.Order {
		ps := providerStats{Name: name, Breaker: "none", Healthy: true, Requests: counts[name]}
		if b, ok := h.breaker(name); ok {
			state := b.State()
			ps.Breaker = state.String()
			ps.Healthy = state != resilience.StateOpen
		}
		out.Providers = append(out.Providers, ps)
	}
	if h.Costs != nil {
		out.CostEventsDropped = h.Costs.Dropped()
	}
	writeJSON(w, http.StatusOK, out)
}

// requestCounts reads per-provider, per-result totals out of the metric
// registry, keyed provider then result.
func (h *Handler) requestCounts() (map[string]map[string]float64, error) {
	families, err := h.gatherer().Gather()
	if err != nil {
		return nil, err
	}
	out := make(map[string]map[string]float64)
	for _, fam := range families {
		if fam.GetName() != requestsMetric {
			continue
		}
		for _, m := range fam.GetMetric() {
			var providerName, result string
			for _, l := range m.GetLabel() {
				switch l.GetName() {
				case "provider":
					providerName = l.GetValue()
				case "result":
					result = l.GetValue()
				}
			}
			if providerName == "" || result == "" {
				continue
			}
			if out[providerName] == nil {
				out[providerName] = make(map[string]float64)
			}
			out[providerName][result] += m.GetCounter().GetValue()
		}
	}
	return out, nil
}

// reloadResponse is the shape of POST /admin/reload.
type reloadResponse struct {
	Status     string   `json:"status"`
	Path       string   `json:"path,omitempty"`
	Aliases    int      `json:"aliases"`
	RateLimits int      `json:"ratelimits"`
	Fallback   int      `json:"fallback"`
	Unroutable []string `json:"unroutable,omitempty"`
}

// reload re-reads the config file and swaps it in.
//
// A parse or validation failure is a 400 and leaves the running config
// untouched, which is the contract Store.Reload provides: the failure
// mode of a bad edit is "the change did not take effect", never
// "routing is now broken".
func (h *Handler) reload(w http.ResponseWriter, _ *http.Request) {
	cfg, err := h.Store.Reload()
	switch {
	case errors.Is(err, config.ErrNotReloadable):
		writeError(w, http.StatusConflict, "not_reloadable",
			"this gateway was started without a config file, so there is nothing to re-read")
		return
	case err != nil:
		h.logger().Warn("admin: config reload rejected, keeping the running config", "err", err)
		writeError(w, http.StatusBadRequest, "invalid_config", err.Error())
		return
	}

	unroutable := h.unroutable(cfg)
	h.logger().Info("admin: config reloaded",
		"path", h.Store.Path(),
		"aliases", len(cfg.Aliases),
		"ratelimits", len(cfg.RateLimits),
		"fallback_chains", len(cfg.Fallback),
		"unroutable", len(unroutable))

	writeJSON(w, http.StatusOK, reloadResponse{
		Status:     "reloaded",
		Path:       h.Store.Path(),
		Aliases:    len(cfg.Aliases),
		RateLimits: len(cfg.RateLimits),
		Fallback:   len(cfg.Fallback),
		Unroutable: unroutable,
	})
}

// unroutable names aliases the new config declares that this binary
// cannot serve, because the provider they name is not wired here.
//
// Returned in the reload response rather than only logged, and this is
// the point of doing the check at reload time at all: the operator who
// just made the edit is standing right there, and telling them now is
// the difference between a typo caught in a second and one discovered
// by a request at 3am. It is not an error — the same config is meant to
// deploy where a key is set and where it is not.
func (h *Handler) unroutable(cfg *config.Config) []string {
	var out []string
	for name := range cfg.Aliases {
		a := cfg.Aliases[name]
		p, ok := h.Providers[a.Provider]
		if !ok || !p.Supports(a.Model) {
			out = append(out, name)
		}
	}
	// Sorted because Go map iteration is randomized and an operator
	// diffing two reload responses should not have to sort them first.
	sort.Strings(out)
	return out
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func writeError(w http.ResponseWriter, status int, kind, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]string{"type": kind, "message": msg},
	})
}
