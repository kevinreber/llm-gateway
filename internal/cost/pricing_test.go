package cost_test

import (
	"math"
	"testing"

	"github.com/kevinreber/llm-gateway/internal/cost"
)

// TestPriceFor_VariantSuffixesDoNotInherit pins the regression that
// prompted this table's rewrite.
//
// Under a delimiter-only prefix rule these all resolved to their base
// model and reported as priced, so the "no price for model" warning
// never fired and the wrong number went into the costs table looking
// exactly like a right one. The direction of the error is not the point
// — o3-mini was billed too high, the rest too low — the point is that
// nothing downstream could tell.
func TestPriceFor_VariantSuffixesDoNotInherit(t *testing.T) {
	tests := []struct {
		model       string
		wantIn      float64
		wantOut     float64
		wouldHaveIn float64 // what the old delimiter-only rule produced
	}{
		{"gpt-5-pro", 15, 120, 1.25},
		{"o3-pro", 20, 80, 2},
		{"o3-mini", 1.10, 4.40, 2},
		{"gpt-4o-2024-05-13", 5, 15, 2.50},
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			got, known := cost.PriceFor(tc.model)
			if !known {
				t.Fatalf("PriceFor(%q) reported unknown; it needs an explicit row", tc.model)
			}
			if got.Input != tc.wantIn || got.Output != tc.wantOut {
				t.Errorf("PriceFor(%q) = $%v/$%v, want $%v/$%v",
					tc.model, got.Input, got.Output, tc.wantIn, tc.wantOut)
			}
			if got.Input == tc.wouldHaveIn {
				t.Errorf("PriceFor(%q) still inherits the base model's $%v input rate",
					tc.model, tc.wouldHaveIn)
			}
		})
	}
}

// TestPriceFor_UnknownVariantWarnsRatherThanInherits is the other half of
// the fix and the more important one. The models above are now explicit
// rows, so they would pass even with the old rule restored. This covers
// the shape of the next one: a variant nobody has added yet must fall
// out of the table loudly instead of quietly borrowing a rate.
func TestPriceFor_UnknownVariantWarnsRatherThanInherits(t *testing.T) {
	for _, model := range []string{
		"gpt-5-turbo",       // a variant suffix that does not exist yet
		"gpt-4o-ultra",      // ditto
		"claude-opus-5-pro", // the same hazard on the Anthropic side
		"o4-mini-pro",
		"chatgpt-4o-latest", // -latest is priced differently from gpt-4o
		"gpt-5-chat-latest",
	} {
		t.Run(model, func(t *testing.T) {
			if _, known := cost.PriceFor(model); known {
				t.Errorf("PriceFor(%q) reported a price; an unrecognized variant must warn instead", model)
			}
		})
	}
}

// TestPriceFor_DateSuffixesStillInherit keeps the fix from overshooting.
// Pricing every snapshot by hand is what the prefix rule exists to
// avoid, and both providers ship dated IDs constantly.
func TestPriceFor_DateSuffixesStillInherit(t *testing.T) {
	tests := []struct {
		model   string
		wantIn  float64
		wantOut float64
	}{
		{"claude-haiku-4-5-20251001", 1, 5},     // -YYYYMMDD
		{"claude-sonnet-5-20260514", 3, 15},     // -YYYYMMDD
		{"gpt-4o-mini-2024-07-18", 0.15, 0.60},  // -YYYY-MM-DD, longest prefix wins
		{"gpt-5-pro-2026-03-01", 15, 120},       // dated snapshot of a variant row
		{"gpt-5.4-mini-2026-06-30", 0.75, 4.50}, // dotted family, dated
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			got, known := cost.PriceFor(tc.model)
			if !known {
				t.Fatalf("PriceFor(%q) reported unknown; dated snapshots must still resolve", tc.model)
			}
			if got.Input != tc.wantIn || got.Output != tc.wantOut {
				t.Errorf("PriceFor(%q) = $%v/$%v, want $%v/$%v",
					tc.model, got.Input, got.Output, tc.wantIn, tc.wantOut)
			}
		})
	}
}

// TestPriceFor_MalformedDateSuffixesDoNotInherit checks the allow-list
// is a real shape check and not just "contains digits". A four-digit
// suffix is a snapshot label on the legacy 3.5 models, not a date, and
// those were priced differently from their base.
func TestPriceFor_MalformedDateSuffixesDoNotInherit(t *testing.T) {
	for _, model := range []string{
		"gpt-4o-2024",         // truncated
		"gpt-4o-2024-07",      // truncated
		"gpt-4o-20240718x",    // trailing junk
		"gpt-4o-2024-07-18-1", // over-long
		"gpt-4o-abcdefgh",     // right length, not digits
		"gpt-4o-2024_07_18",   // wrong delimiters
	} {
		t.Run(model, func(t *testing.T) {
			if _, known := cost.PriceFor(model); known {
				t.Errorf("PriceFor(%q) inherited a price from a malformed date suffix", model)
			}
		})
	}
}

// TestPriceFor_CurrentOpenAICatalog is the audit that found the bug,
// kept as a test. Every entry is a published first-party rate from
// developers.openai.com/api/docs/pricing as of 2026-08-18.
//
// This exists because the failure it guards against is invisible at
// runtime: a stale table does not break a request, it just bills the
// wrong number forever. Refreshing it is a deliberate act with a diff,
// which is the only way anyone notices a price moved.
func TestPriceFor_CurrentOpenAICatalog(t *testing.T) {
	catalog := map[string]cost.Price{
		"gpt-5.6-sol":            {Input: 5, Output: 30},
		"gpt-5.6-terra":          {Input: 2, Output: 12},
		"gpt-5.6-luna":           {Input: 0.20, Output: 1.20},
		"gpt-5.5":                {Input: 5, Output: 30},
		"gpt-5.5-pro":            {Input: 30, Output: 180},
		"gpt-5.4":                {Input: 2.50, Output: 15},
		"gpt-5.4-mini":           {Input: 0.75, Output: 4.50},
		"gpt-5.4-nano":           {Input: 0.20, Output: 1.25},
		"gpt-5.4-pro":            {Input: 30, Output: 180},
		"gpt-5.2":                {Input: 1.75, Output: 14},
		"gpt-5.2-pro":            {Input: 21, Output: 168},
		"gpt-5.1":                {Input: 1.25, Output: 10},
		"gpt-5":                  {Input: 1.25, Output: 10},
		"gpt-5-mini":             {Input: 0.25, Output: 2},
		"gpt-5-nano":             {Input: 0.05, Output: 0.40},
		"gpt-5-pro":              {Input: 15, Output: 120},
		"gpt-4.1":                {Input: 2, Output: 8},
		"gpt-4.1-mini":           {Input: 0.40, Output: 1.60},
		"gpt-4.1-nano":           {Input: 0.10, Output: 0.40},
		"gpt-4o":                 {Input: 2.50, Output: 10},
		"gpt-4o-2024-05-13":      {Input: 5, Output: 15},
		"gpt-4o-mini":            {Input: 0.15, Output: 0.60},
		"o1":                     {Input: 15, Output: 60},
		"o1-pro":                 {Input: 150, Output: 600},
		"o3":                     {Input: 2, Output: 8},
		"o3-pro":                 {Input: 20, Output: 80},
		"o3-mini":                {Input: 1.10, Output: 4.40},
		"o4-mini":                {Input: 1.10, Output: 4.40},
		"gpt-4-turbo-2024-04-09": {Input: 10, Output: 30},
		"gpt-4-0613":             {Input: 30, Output: 60},
		"gpt-3.5-turbo":          {Input: 0.50, Output: 1.50},
		"gpt-3.5-turbo-0125":     {Input: 0.50, Output: 1.50},
		"gpt-3.5-turbo-1106":     {Input: 1, Output: 2},
		"gpt-3.5-turbo-instruct": {Input: 1.50, Output: 2},
	}

	for model, want := range catalog {
		t.Run(model, func(t *testing.T) {
			got, known := cost.PriceFor(model)
			if !known {
				t.Fatalf("PriceFor(%q) is unpriced; it bills zero and warns", model)
			}
			if got != want {
				t.Errorf("PriceFor(%q) = $%v/$%v, want $%v/$%v",
					model, got.Input, got.Output, want.Input, want.Output)
			}
		})
	}
}

// TestCents_ProVariantIsNotUnderbilled is the same regression measured
// in the unit that matters, because a 12x error in a rate is abstract
// and a 12x error on a bill is not.
func TestCents_ProVariantIsNotUnderbilled(t *testing.T) {
	// 1M input + 1M output at gpt-5-pro's $15/$120 = $135 = 13,500c.
	// Inheriting gpt-5's $1.25/$10 would have produced 1,125c.
	got, known := cost.Cents("gpt-5-pro", 1_000_000, 1_000_000)
	if !known {
		t.Fatal("gpt-5-pro is unpriced")
	}
	if math.Abs(got-13_500) > 1e-6 {
		t.Errorf("Cents = %v, want 13500", got)
	}
}
