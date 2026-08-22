package config

import "sync/atomic"

// Store holds the live configuration behind an atomic pointer, so the
// request path reads it without a lock and a reload replaces it without
// stopping traffic.
//
// atomic.Pointer rather than the atomic.Value the design sketch called
// for. Value carries a "first store fixes the dynamic type" rule that
// panics at runtime if anything ever stores a different type into it,
// and it hands back an `any` that every reader has to assert. Pointer is
// the same lock-free swap with the type checked once, at compile time.
//
// The zero Store is not usable; construct one with NewStore or Static.
type Store struct {
	// path is the file Reload re-reads. Empty marks a static store —
	// one built from a config value rather than from a file — and
	// Reload says so rather than inventing a path to read.
	path string
	cur  atomic.Pointer[Config]
}

// NewStore returns a store backed by path and seeded with initial.
//
// initial is passed in rather than read here because startup has
// already loaded and logged it, and because a store whose constructor
// can fail would push error handling into every test that needs one.
func NewStore(path string, initial *Config) *Store {
	if initial == nil {
		initial = &Config{}
	}
	s := &Store{path: path}
	s.cur.Store(initial)
	return s
}

// Static returns a store that always yields cfg and cannot reload.
//
// This is what a test wants, and what production gets when no config
// file was found: the gateway serves direct model names with no limits,
// and there is no file to re-read.
func Static(cfg *Config) *Store {
	return NewStore("", cfg)
}

// Load returns the configuration in force right now. Never nil.
//
// Callers serving one request should call this once and pass the result
// down rather than calling it at each decision point. A reload landing
// mid-request would otherwise let a single request resolve its alias
// against one config and take its rate limit from another — rare, and
// correspondingly miserable to reproduce.
func (s *Store) Load() *Config { return s.cur.Load() }

// Path reports the file backing this store, empty when it is static.
func (s *Store) Path() string { return s.path }

// Reloadable reports whether Reload can do anything.
func (s *Store) Reloadable() bool { return s.path != "" }

// Reload re-reads the backing file and swaps it in, returning the new
// configuration.
//
// The parse and its validation both happen before the swap, so a file
// with a typo in it leaves the running config alone. That ordering is
// the whole point: the failure mode of a bad edit has to be "the change
// did not take effect", never "routing is now broken", because the
// second one is discovered by traffic.
func (s *Store) Reload() (*Config, error) {
	if !s.Reloadable() {
		return s.Load(), ErrNotReloadable
	}
	cfg, err := Load(s.path)
	if err != nil {
		return s.Load(), err
	}
	s.cur.Store(cfg)
	return cfg, nil
}
