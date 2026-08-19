package config_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/config"
)

const resilienceYAML = `
aliases:
  fast:      { provider: anthropic, model: claude-haiku-4-5 }
  smart:     { provider: anthropic, model: claude-sonnet-5 }
  smart-alt: { provider: openai,    model: gpt-4o }

breakers:
  anthropic: { failure_threshold: 5, recovery_timeout: 30s }
  openai:    { failure_threshold: 3, recovery_timeout: 1m }

fallback:
  smart: [smart-alt, fast]
  fast:  []
`

func TestParse_BreakersAndFallback(t *testing.T) {
	cfg, err := config.Parse(strings.NewReader(resilienceYAML))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	b, ok := cfg.BreakerFor("anthropic")
	if !ok {
		t.Fatal("breaker anthropic not found")
	}
	if b.FailureThreshold != 5 {
		t.Errorf("FailureThreshold = %d, want 5", b.FailureThreshold)
	}
	// yaml.v3 decodes a duration string into time.Duration natively, so
	// `30s` in the file is 30 seconds here and not a bare integer that
	// would land as 30 nanoseconds.
	if b.RecoveryTimeout != 30*time.Second {
		t.Errorf("RecoveryTimeout = %v, want 30s", b.RecoveryTimeout)
	}
	if ob, _ := cfg.BreakerFor("openai"); ob.RecoveryTimeout != time.Minute {
		t.Errorf("openai RecoveryTimeout = %v, want 1m", ob.RecoveryTimeout)
	}
	if _, ok := cfg.BreakerFor("gemini"); ok {
		t.Error("BreakerFor returned a breaker for an unconfigured provider")
	}

	chain := cfg.FallbackFor("smart")
	if len(chain) != 2 || chain[0] != "smart-alt" || chain[1] != "fast" {
		t.Errorf("FallbackFor(smart) = %v, want [smart-alt fast]", chain)
	}
	// An explicit empty list and an absent key are deliberately the same
	// thing, so `fast: []` can document "this one does not fail over".
	if got := cfg.FallbackFor("fast"); len(got) != 0 {
		t.Errorf("FallbackFor(fast) = %v, want empty", got)
	}
	if got := cfg.FallbackFor("nonexistent"); got != nil {
		t.Errorf("FallbackFor(nonexistent) = %v, want nil", got)
	}
}

func TestParse_ResilienceBlocksAreOptional(t *testing.T) {
	// A Phase 2 config must keep parsing unchanged. Adding fields to the
	// struct is safe under strict parsing, but their absence has to stay
	// valid or every existing deployment breaks on upgrade.
	cfg, err := config.Parse(strings.NewReader(goodYAML))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(cfg.Breakers) != 0 {
		t.Errorf("Breakers = %v, want empty", cfg.Breakers)
	}
	if len(cfg.Fallback) != 0 {
		t.Errorf("Fallback = %v, want empty", cfg.Fallback)
	}
}

func TestValidate_SampleConfigIsValid(t *testing.T) {
	// The committed gateway.yaml is the first thing anyone runs, and
	// strict parsing means a stale sample is a startup error rather than
	// a cosmetic problem.
	f, err := os.Open(filepath.Join("..", "..", "gateway.yaml"))
	if err != nil {
		t.Fatalf("open sample config: %v", err)
	}
	defer func() { _ = f.Close() }()

	cfg, err := config.Parse(f)
	if err != nil {
		t.Fatalf("gateway.yaml does not parse: %v", err)
	}
	if len(cfg.Aliases) == 0 {
		t.Error("sample config has no aliases")
	}
	if len(cfg.Fallback) == 0 {
		t.Error("sample config has no fallback chains")
	}
	if len(cfg.Breakers) == 0 {
		t.Error("sample config has no breakers")
	}
}
