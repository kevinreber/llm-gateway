// Package cost turns provider token usage into billable amounts and
// persists them off the request path.
//
// The split is deliberate: pricing is a pure function (this file) and
// persistence is a buffered background writer (writer.go). A request
// never blocks on Postgres to record what it cost.
package cost

import (
	"context"
	"log/slog"
	"strings"
	"time"
)

// Event is one billable completion. Written to the costs table in
// batches; see Writer.
type Event struct {
	TS           time.Time
	Provider     string
	Model        string
	Alias        string // empty when the caller named a concrete model
	InputTokens  int
	OutputTokens int
	CostCents    float64
}

// Tracker records completed requests. Writer implements it; Discard is
// the no-op used when no cost sink is configured.
type Tracker interface {
	Track(Event)
}

// Discard drops every event. Used when the gateway runs without a cost
// backend (local development, or a deployment that only wants the proxy).
type Discard struct{}

// Track implements Tracker.
func (Discard) Track(Event) {}

// Price is a model's list price in US dollars per million tokens.
type Price struct {
	Input  float64
	Output float64
}

// prices is the list price table, keyed by model ID.
//
// Keys are the canonical IDs providers return in the response body, so
// we bill against what actually served the request rather than what the
// client asked for. That matters more now that a request can fail over:
// an alias that asked for Sonnet and was served by GPT-4o must bill at
// GPT-4o's rate, and keying on the response model is what makes that
// automatic rather than something the fallback path has to remember.
//
// Lookup falls back to longest-prefix match, which covers date-suffixed
// IDs (claude-haiku-4-5-20251001, gpt-4o-mini-2024-07-18) without
// needing a row per snapshot. Longest wins, so gpt-4o-mini bills as
// itself and not as gpt-4o.
//
// One flat table is safe only while every model ID is unique to one
// vendor's first-party API, which is true today. It stops being true for
// resold endpoints — Bedrock and Vertex serve Claude IDs at partner
// rates, Azure serves GPT IDs at its own — so those need a table keyed
// by {provider, model} rather than a row added here.
var prices = map[string]Price{
	// Anthropic first-party rates, USD per million tokens.
	"claude-fable-5":    {Input: 10, Output: 50},
	"claude-mythos-5":   {Input: 10, Output: 50},
	"claude-opus-5":     {Input: 5, Output: 25},
	"claude-opus-4-8":   {Input: 5, Output: 25},
	"claude-opus-4-7":   {Input: 5, Output: 25},
	"claude-opus-4-6":   {Input: 5, Output: 25},
	"claude-sonnet-5":   {Input: 3, Output: 15},
	"claude-sonnet-4-6": {Input: 3, Output: 15},
	"claude-haiku-4-5":  {Input: 1, Output: 5},

	// OpenAI first-party rates, USD per million tokens.
	"gpt-5":        {Input: 1.25, Output: 10},
	"gpt-5-mini":   {Input: 0.25, Output: 2},
	"gpt-5-nano":   {Input: 0.05, Output: 0.40},
	"gpt-4.1":      {Input: 2, Output: 8},
	"gpt-4.1-mini": {Input: 0.40, Output: 1.60},
	"gpt-4.1-nano": {Input: 0.10, Output: 0.40},
	"gpt-4o":       {Input: 2.50, Output: 10},
	"gpt-4o-mini":  {Input: 0.15, Output: 0.60},
	"o3":           {Input: 2, Output: 8},
	"o4-mini":      {Input: 1.10, Output: 4.40},
}

// PriceFor returns the list price for a model. The false return means
// the model is not in the table — callers should record a zero cost and
// say so loudly rather than guessing at a rate.
func PriceFor(model string) (Price, bool) {
	if p, ok := prices[model]; ok {
		return p, true
	}
	// Longest-prefix match so a dated snapshot bills at its family's
	// rate. Longest wins because "claude-opus-4-8" and a hypothetical
	// "claude-opus-4" would both prefix-match the same ID.
	var (
		best    Price
		bestLen int
	)
	for id, p := range prices {
		if len(id) > bestLen && hasModelPrefix(model, id) {
			best, bestLen = p, len(id)
		}
	}
	return best, bestLen > 0
}

// hasModelPrefix reports whether model is id or a dated snapshot of it.
//
// The boundary check is the whole point: a bare strings.HasPrefix would
// make any future model whose ID merely starts with a known one inherit
// that price and report as priced, so "claude-sonnet-50" would silently
// bill at Sonnet 5 rates with no unpriced-model warning to catch it.
// Billing something plausible-but-wrong is worse than billing zero,
// because nothing downstream looks wrong enough to investigate.
func hasModelPrefix(model, id string) bool {
	if !strings.HasPrefix(model, id) {
		return false
	}
	return len(model) == len(id) || model[len(id)] == '-'
}

// Cents computes the cost of a completion in US cents. The false return
// propagates PriceFor's "unknown model" signal.
func Cents(model string, inputTokens, outputTokens int) (float64, bool) {
	p, ok := PriceFor(model)
	if !ok {
		return 0, false
	}
	const (
		perMillion  = 1_000_000.0
		centsPerUSD = 100.0
	)
	usd := (float64(inputTokens)/perMillion)*p.Input +
		(float64(outputTokens)/perMillion)*p.Output
	return usd * centsPerUSD, true
}

// Sink persists a batch of events. Implemented by internal/store for
// Postgres; the interface keeps Writer testable without a database.
type Sink interface {
	InsertCosts(ctx context.Context, batch []Event) error
}

// LogSink writes batch summaries to the log instead of a database. It is
// the sink used when no DATABASE_URL is configured, so local development
// exercises the same buffering and batching path as production and can
// see what a request cost.
type LogSink struct {
	Logger *slog.Logger
}

// InsertCosts implements Sink.
func (s LogSink) InsertCosts(_ context.Context, batch []Event) error {
	var cents float64
	for _, e := range batch {
		cents += e.CostCents
	}
	s.Logger.Info("cost batch", "events", len(batch), "cents", cents)
	return nil
}
