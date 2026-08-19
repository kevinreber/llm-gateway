package config_test

import (
	"strings"
	"testing"

	"github.com/kevinreber/llm-gateway/internal/config"
)

const goodYAML = `
aliases:
  fast:  { provider: anthropic, model: claude-haiku-4-5 }
  smart: { provider: anthropic, model: claude-sonnet-5 }

ratelimits:
  fast:  { capacity: 100, refill_rate: 50 }
  smart: { capacity: 50,  refill_rate: 20 }
`

func TestParse_HappyPath(t *testing.T) {
	cfg, err := config.Parse(strings.NewReader(goodYAML))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	alias, ok := cfg.Resolve("smart")
	if !ok {
		t.Fatal("alias smart not found")
	}
	if alias.Provider != "anthropic" || alias.Model != "claude-sonnet-5" {
		t.Errorf("smart = %+v, want {anthropic claude-sonnet-5}", alias)
	}

	limit, ok := cfg.LimitFor("fast")
	if !ok {
		t.Fatal("ratelimit fast not found")
	}
	if limit.Capacity != 100 || limit.RefillRate != 50 {
		t.Errorf("fast limit = %+v, want {100 50}", limit)
	}
}

func TestParse_EmptyDocument(t *testing.T) {
	// No config file is a supported deployment: the gateway falls back
	// to direct model names with no rate limiting.
	cfg, err := config.Parse(strings.NewReader(""))
	if err != nil {
		t.Fatalf("Parse empty: %v", err)
	}
	if _, ok := cfg.Resolve("smart"); ok {
		t.Error("empty config resolved an alias")
	}
	if _, ok := cfg.LimitFor("smart"); ok {
		t.Error("empty config returned a limit")
	}
}

func TestParse_Rejects(t *testing.T) {
	tests := []struct {
		name string
		yaml string
		want string
	}{
		{
			name: "unknown top-level key",
			yaml: "alises:\n  fast: { provider: anthropic, model: m }\n",
			want: "field alises not found",
		},
		{
			name: "unknown alias key",
			yaml: "aliases:\n  fast: { provider: anthropic, modle: m }\n",
			want: "field modle not found",
		},
		{
			name: "alias missing provider",
			yaml: "aliases:\n  fast: { model: claude-haiku-4-5 }\n",
			want: `alias "fast": provider is required`,
		},
		{
			name: "alias missing model",
			yaml: "aliases:\n  fast: { provider: anthropic }\n",
			want: `alias "fast": model is required`,
		},
		{
			name: "ratelimit for unknown alias",
			yaml: "aliases:\n  fast: { provider: anthropic, model: m }\nratelimits:\n  fst: { capacity: 1, refill_rate: 1 }\n",
			want: `ratelimit "fst": no alias with that name`,
		},
		{
			name: "zero capacity",
			yaml: "aliases:\n  fast: { provider: anthropic, model: m }\nratelimits:\n  fast: { capacity: 0, refill_rate: 1 }\n",
			want: `ratelimit "fast": capacity must be > 0`,
		},
		{
			name: "zero refill rate",
			yaml: "aliases:\n  fast: { provider: anthropic, model: m }\nratelimits:\n  fast: { capacity: 1, refill_rate: 0 }\n",
			want: `ratelimit "fast": refill_rate must be > 0`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := config.Parse(strings.NewReader(tt.yaml))
			if err == nil {
				t.Fatal("Parse succeeded, want error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %q, want it to contain %q", err, tt.want)
			}
		})
	}
}

func TestValidate_DeterministicError(t *testing.T) {
	// Two broken aliases: the reported one must not depend on map
	// iteration order, so run it enough times to catch randomization.
	const yaml = `
aliases:
  aaa: { provider: anthropic }
  zzz: { provider: anthropic }
`
	for i := 0; i < 50; i++ {
		_, err := config.Parse(strings.NewReader(yaml))
		if err == nil {
			t.Fatal("Parse succeeded, want error")
		}
		if !strings.Contains(err.Error(), `alias "aaa"`) {
			t.Fatalf("iteration %d: err = %q, want the alphabetically first alias", i, err)
		}
	}
}
