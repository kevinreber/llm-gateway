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
// Lookup falls back to longest-prefix match, but ONLY across a date
// suffix (claude-haiku-4-5-20251001, gpt-4o-mini-2024-07-18), so a
// snapshot bills at its family's rate without needing a row per release.
// Longest wins, so gpt-4o-mini bills as itself and not as gpt-4o. See
// hasModelPrefix for why a variant suffix must never inherit.
//
// Every row is a published first-party rate, not an interpolation. A
// model that is missing here bills zero and warns, which is loud and
// fixable; a model that inherits a plausible-looking wrong rate is
// neither. When in doubt, leave it out.
//
// One flat table is safe only while every model ID is unique to one
// vendor's first-party API, which is true today. It stops being true for
// resold endpoints — Bedrock and Vertex serve Claude IDs at partner
// rates, Azure serves GPT IDs at its own — so those need a table keyed
// by {provider, model} rather than a row added here.
//
// OpenAI rates verified against developers.openai.com/api/docs/pricing
// on 2026-08-18. The GPT-5.5 and GPT-5.4 families are published as a
// "<272K context" tier; if a long-context tier is introduced, these stop
// being a function of the model alone and the table needs a context
// dimension rather than more rows.
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
	"gpt-5.6-sol":   {Input: 5, Output: 30},
	"gpt-5.6-terra": {Input: 2, Output: 12},
	"gpt-5.6-luna":  {Input: 0.20, Output: 1.20},
	"gpt-5.5":       {Input: 5, Output: 30},
	"gpt-5.5-pro":   {Input: 30, Output: 180},
	"gpt-5.4":       {Input: 2.50, Output: 15},
	"gpt-5.4-mini":  {Input: 0.75, Output: 4.50},
	"gpt-5.4-nano":  {Input: 0.20, Output: 1.25},
	"gpt-5.4-pro":   {Input: 30, Output: 180},
	"gpt-5.2":       {Input: 1.75, Output: 14},
	"gpt-5.2-pro":   {Input: 21, Output: 168},
	"gpt-5.1":       {Input: 1.25, Output: 10},
	"gpt-5":         {Input: 1.25, Output: 10},
	"gpt-5-mini":    {Input: 0.25, Output: 2},
	"gpt-5-nano":    {Input: 0.05, Output: 0.40},
	"gpt-5-pro":     {Input: 15, Output: 120},
	"gpt-4.1":       {Input: 2, Output: 8},
	"gpt-4.1-mini":  {Input: 0.40, Output: 1.60},
	"gpt-4.1-nano":  {Input: 0.10, Output: 0.40},
	"gpt-4o":        {Input: 2.50, Output: 10},
	// The May 2024 GPT-4o snapshot never got the later price cut, so it
	// needs its own row: it is the case that proves date suffixes cannot
	// be assumed to share the family's current rate either.
	"gpt-4o-2024-05-13": {Input: 5, Output: 15},
	"gpt-4o-mini":       {Input: 0.15, Output: 0.60},
	"o1":                {Input: 15, Output: 60},
	"o1-pro":            {Input: 150, Output: 600},
	"o3":                {Input: 2, Output: 8},
	"o3-pro":            {Input: 20, Output: 80},
	"o3-mini":           {Input: 1.10, Output: 4.40},
	"o4-mini":           {Input: 1.10, Output: 4.40},

	// OpenAI legacy tier. Deprecated rather than retired, and the
	// provider's Supports() still routes them, so pricing them beats
	// billing them at zero. Each 3.5 snapshot is listed separately
	// because they were priced differently and their suffixes are not
	// dates.
	"gpt-4-turbo-2024-04-09": {Input: 10, Output: 30},
	"gpt-4-0613":             {Input: 30, Output: 60},
	"gpt-3.5-turbo":          {Input: 0.50, Output: 1.50},
	"gpt-3.5-turbo-0125":     {Input: 0.50, Output: 1.50},
	"gpt-3.5-turbo-1106":     {Input: 1, Output: 2},
	"gpt-3.5-turbo-instruct": {Input: 1.50, Output: 2},

	// Gemini, USD per million tokens.
	"gemini-2.5-pro":   {Input: 1.25, Output: 10},
	"gemini-2.5-flash": {Input: 0.30, Output: 2.50},
	"gemini-2.0-flash": {Input: 0.10, Output: 0.40},
	"gemini-1.5-pro":   {Input: 1.25, Output: 5},
	"gemini-1.5-flash": {Input: 0.075, Output: 0.30},

	// Every locally-served model, at its real price. Local inference
	// costs electricity, not tokens, and there is no per-token rate to
	// look up. See LocalModelID for why they all share one entry.
	LocalModelID: {Input: 0, Output: 0},

	// Deliberately absent: the ChatGPT-surface model published at $5/$30.
	// The pricing page shows it as "chat-latest", which reads as a
	// display label rather than an API identifier, and the real ID is
	// plausibly chatgpt-4o-latest or gpt-5-chat-latest. Guessing which
	// would either do nothing (a key that never matches) or attach a
	// rate to the wrong model; leaving it out warns instead.
}

// localModelPrefix marks a model served by a local runtime. Kept as a
// literal rather than imported from internal/provider, so the pricing
// table does not take a dependency on the client packages; the two are
// asserted to agree by a test in the provider package.
const localModelPrefix = "ollama/"

// LocalModelID is the single pricing-table entry every locally-served
// model bills and reports under.
const LocalModelID = "ollama/local"

// PriceFor returns the list price for a model. The false return means
// the model is not in the table — callers should record a zero cost and
// say so loudly rather than guessing at a rate.
func PriceFor(model string) (Price, bool) {
	id, ok := CanonicalModel(model)
	if !ok {
		return Price{}, false
	}
	return prices[id], true
}

// CanonicalModel returns the pricing-table ID that model bills under,
// collapsing a dated snapshot onto its family: both "gpt-4o" and
// "gpt-4o-2024-08-06" answer "gpt-4o". The false return means the model
// is not in the table.
//
// This exists to bound a metric label. The model recorded against a
// completion is whatever the upstream echoed back, so labelling a
// Prometheus series with that string directly hands cardinality control
// to the provider — one new series for every distinct value it decides
// to return, forever, because a counter series is never retired. The
// pricing table is finite, so a label derived from it is too.
//
// Collapsing snapshots is not merely a cardinality tax either. Spend on
// "gpt-4o" split across a series per release date is the wrong shape for
// the question a cost dashboard is asked, which is what a model family
// costs and not what each of its snapshots cost.
func CanonicalModel(model string) (string, bool) {
	if _, ok := prices[model]; ok {
		return model, true
	}
	// Every locally-served model collapses onto one ID. Two reasons,
	// and they point the same way: local inference has no per-token
	// price to distinguish, and the set of things an operator can pull
	// onto a box is unbounded, so a label per local model is a label
	// the provider — here, whoever runs the box — controls.
	if strings.HasPrefix(model, localModelPrefix) {
		return LocalModelID, true
	}
	// Longest-prefix match so a dated snapshot bills at its family's
	// rate. Longest wins because "claude-opus-4-8" and a hypothetical
	// "claude-opus-4" would both prefix-match the same ID.
	var best string
	for id := range prices {
		if len(id) > len(best) && hasModelPrefix(model, id) {
			best = id
		}
	}
	return best, best != ""
}

// hasModelPrefix reports whether model is id or a dated snapshot of it.
//
// Inheriting a price across a suffix is only ever safe for a date. The
// earlier version of this accepted any hyphen-delimited suffix, on the
// theory that a bare strings.HasPrefix was the only real hazard —
// "claude-sonnet-50" must not bill at Sonnet 5's rate. That reasoning
// was right about the hazard and wrong about its extent. Vendors ship
// capability variants under hyphenated names too, and those are the ones
// priced differently: gpt-5-pro is 12x gpt-5, o3-pro is 10x o3, and
// o3-mini is cheaper than o3. All three read as "the family, dated"
// under a delimiter-only rule, so all three billed at the base model's
// rate AND reported as priced — which suppressed the unpriced-model
// warning that exists to catch exactly this.
//
// Requiring the suffix to look like a date inverts the default. An
// unrecognized variant now falls out of the table and warns instead of
// silently inheriting, so the failure mode of a new model release is a
// log line rather than an invisible billing error. That is the trade
// this function was always trying to make: billing something
// plausible-but-wrong is worse than billing zero, because nothing
// downstream looks wrong enough to investigate.
//
// Note this excludes "-latest" aliases by design, and that is not
// collateral damage: chatgpt-4o-latest is priced at twice gpt-4o, so
// "latest" is not reliably "the family's current rate" either. It also
// rarely matters in practice, because both providers echo back the
// concrete dated ID in the response and trackCost bills against that.
func hasModelPrefix(model, id string) bool {
	if !strings.HasPrefix(model, id) {
		return false
	}
	return isDateSuffix(model[len(id):])
}

// isDateSuffix reports whether s is empty (an exact match) or a release
// date suffix in one of the two forms the providers actually use:
// "-20251001" and "-2024-07-18".
//
// Deliberately an allow-list of shapes rather than a deny-list of known
// variant names. A deny-list has to be updated before each new suffix a
// vendor invents can be billed correctly, and until someone notices, the
// mis-billing is silent. This way the unknown case is the safe case.
func isDateSuffix(s string) bool {
	if s == "" {
		return true
	}
	if s[0] != '-' {
		return false
	}
	switch rest := s[1:]; len(rest) {
	case len("20251001"):
		return allDigits(rest)
	case len("2024-07-18"):
		return allDigits(rest[0:4]) && rest[4] == '-' &&
			allDigits(rest[5:7]) && rest[7] == '-' &&
			allDigits(rest[8:10])
	default:
		return false
	}
}

func allDigits(s string) bool {
	for i := range len(s) {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
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
