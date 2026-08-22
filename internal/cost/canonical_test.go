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
