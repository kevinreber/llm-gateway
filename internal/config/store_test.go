package config

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const storeYAML = `
aliases:
  smart: { provider: anthropic, model: claude-sonnet-5 }
`

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "gateway.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return path
}

func TestStore_LoadReturnsTheSeededConfig(t *testing.T) {
	cfg, err := Parse(strings.NewReader(storeYAML))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	s := NewStore("gateway.yaml", cfg)

	if got := s.Load(); got != cfg {
		t.Error("Load returned a different config than it was seeded with")
	}
	if !s.Reloadable() {
		t.Error("a store with a path should be reloadable")
	}
}

func TestStore_NilSeedIsUsable(t *testing.T) {
	// The zero Config means "no aliases, no limits", which is a valid
	// deployment. A nil here must not become a nil dereference on the
	// first request.
	s := NewStore("", nil)
	if s.Load() == nil {
		t.Fatal("Load returned nil")
	}
	if len(s.Load().Aliases) != 0 {
		t.Error("expected an empty config")
	}
}

func TestStore_StaticCannotReload(t *testing.T) {
	s := Static(&Config{})

	_, err := s.Reload()
	if !errors.Is(err, ErrNotReloadable) {
		t.Errorf("err = %v, want ErrNotReloadable", err)
	}
}

func TestStore_ReloadPicksUpEdits(t *testing.T) {
	path := writeConfig(t, storeYAML)
	initial, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	s := NewStore(path, initial)

	if err := os.WriteFile(path, []byte(`
aliases:
  smart: { provider: anthropic, model: claude-opus-5 }
  fast:  { provider: anthropic, model: claude-haiku-4-5 }
`), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}

	cfg, err := s.Reload()
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if len(cfg.Aliases) != 2 {
		t.Errorf("reloaded aliases = %d, want 2", len(cfg.Aliases))
	}
	if got := s.Load().Aliases["smart"].Model; got != "claude-opus-5" {
		t.Errorf("smart model = %q, want the edited value", got)
	}
}

func TestStore_RejectedReloadKeepsTheRunningConfig(t *testing.T) {
	// The contract that makes reload safe to expose at all: the failure
	// mode of a bad edit is "the change did not take effect", never
	// "routing is now broken", because the second is discovered by
	// traffic rather than by the operator standing there.
	path := writeConfig(t, storeYAML)
	initial, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	s := NewStore(path, initial)

	if err := os.WriteFile(path, []byte("aliases:\n  broken: { provider: anthropic }\n"), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}

	if _, err := s.Reload(); err == nil {
		t.Fatal("expected the invalid config to be rejected")
	}
	if got := s.Load().Aliases["smart"].Model; got != "claude-sonnet-5" {
		t.Errorf("running config changed to %q; a rejected reload must not swap", got)
	}
}

func TestStore_ReloadOfAMissingFileKeepsTheRunningConfig(t *testing.T) {
	path := writeConfig(t, storeYAML)
	initial, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	s := NewStore(path, initial)

	if err := os.Remove(path); err != nil {
		t.Fatalf("remove: %v", err)
	}

	if _, err := s.Reload(); err == nil {
		t.Fatal("expected an error for a missing file")
	}
	if len(s.Load().Aliases) != 1 {
		t.Error("running config was replaced after a failed reload")
	}
}
