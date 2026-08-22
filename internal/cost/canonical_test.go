package cost

import "testing"

func TestCanonicalModel(t *testing.T) {
	cases := []struct {
		name  string
		model string
		want  string
		ok    bool
	}{
		{"exact match", "claude-sonnet-5", "claude-sonnet-5", true},
		{"compact date suffix", "claude-sonnet-5-20251001", "claude-sonnet-5", true},
		{"hyphenated date suffix", "gpt-4o-2024-08-06", "gpt-4o", true},
		{"unknown model", "nonexistent-model-xyz", "", false},
		// A capability variant is not a date, so it must not inherit the
		// family's identity any more than it inherits the family's price.
		{"capability variant is not a snapshot", "gpt-5-pro-imaginary", "", false},
		{"empty", "", "", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := CanonicalModel(tc.model)
			if ok != tc.ok {
				t.Fatalf("ok = %v, want %v", ok, tc.ok)
			}
			if got != tc.want {
				t.Errorf("CanonicalModel(%q) = %q, want %q", tc.model, got, tc.want)
			}
		})
	}
}

func TestCanonicalModel_AgreesWithPriceFor(t *testing.T) {
	// The two must not drift: a model that prices must canonicalize, or
	// the metric would bucket spend under "unpriced" while the cost row
	// records a real number.
	for _, m := range []string{
		"claude-sonnet-5", "claude-sonnet-5-20251001", "gpt-4o",
		"gpt-4o-2024-08-06", "nonexistent-model-xyz",
	} {
		_, priced := PriceFor(m)
		_, canonical := CanonicalModel(m)
		if priced != canonical {
			t.Errorf("%q: PriceFor ok=%v but CanonicalModel ok=%v", m, priced, canonical)
		}
	}
}

func TestCanonicalModel_LocalModelsCollapseOntoOneID(t *testing.T) {
	// The set of things an operator can pull onto a box is unbounded,
	// so a label per local model is a label somebody else controls.
	// They also all cost the same: nothing per token.
	for _, m := range []string{"ollama/llama3", "ollama/mistral", "ollama/qwen2.5-coder:7b"} {
		got, ok := CanonicalModel(m)
		if !ok {
			t.Errorf("CanonicalModel(%q) reported unknown; local models are priced at zero", m)
		}
		if got != LocalModelID {
			t.Errorf("CanonicalModel(%q) = %q, want %q", m, got, LocalModelID)
		}
	}
}

func TestCents_LocalModelsAreFreeButKnown(t *testing.T) {
	// Known matters as much as free. An unknown model logs a warning
	// and asks somebody to add a price; a local model has no price to
	// add, so warning about it every request would be noise that
	// trains people to ignore the real ones.
	cents, known := Cents("ollama/llama3", 100_000, 100_000)
	if !known {
		t.Error("local model reported as unpriced")
	}
	if cents != 0 {
		t.Errorf("cents = %v, want 0", cents)
	}
}
