// Package config parses and validates gateway.yaml: the alias table that
// maps client-facing model names onto concrete {provider, model} pairs,
// and the per-alias rate-limit policies handed to bucketd.
//
// Parsing is strict — an unknown key is an error, not a silent no-op. A
// typo in a production config should fail at startup, not quietly
// disable a rate limit.
//
// Phase 4 adds an fsnotify watcher and an atomic.Value swap on top of
// this package; Parse and Validate stay as they are, and the watcher
// just calls them again on change.
package config

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sort"

	"gopkg.in/yaml.v3"
)

// Alias resolves a client-facing model name to a concrete provider and
// model. Clients ask for `model: smart`; the gateway sends
// `claude-sonnet-5` to Anthropic.
type Alias struct {
	Provider string `yaml:"provider"`
	Model    string `yaml:"model"`
}

// Limit is a token-bucket policy. It mirrors bucketd's client.Limit but
// is declared here so the config package doesn't drag a gRPC dependency
// into everything that reads config.
type Limit struct {
	// Capacity is the burst ceiling — the most tokens the bucket holds.
	Capacity int32 `yaml:"capacity"`
	// RefillRate is tokens added per second, i.e. the sustained rate.
	RefillRate float64 `yaml:"refill_rate"`
}

// Config is a parsed gateway.yaml.
//
// The zero value is usable and means "no aliases, no limits": every
// request is treated as a direct model name and no rate limiting is
// applied. That is deliberately the Phase 1 behavior, so running without
// a config file still serves traffic.
type Config struct {
	Aliases    map[string]Alias `yaml:"aliases"`
	RateLimits map[string]Limit `yaml:"ratelimits"`
}

// Load reads and validates a config file.
func Load(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open config: %w", err)
	}
	defer func() { _ = f.Close() }()

	cfg, err := Parse(f)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return cfg, nil
}

// Parse decodes and validates YAML from r. Unknown fields are rejected.
func Parse(r io.Reader) (*Config, error) {
	dec := yaml.NewDecoder(r)
	dec.KnownFields(true)

	var cfg Config
	if err := dec.Decode(&cfg); err != nil {
		// An empty document decodes to io.EOF rather than an empty
		// struct. Treat that as "no config", not a failure.
		if errors.Is(err, io.EOF) {
			return &Config{}, nil
		}
		return nil, err
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// Validate reports the first structural problem with the config.
//
// Problems are checked in sorted-key order because Go map iteration is
// randomized: without this, a config with two mistakes would report a
// different one on each restart, turning "fix the error, get a new
// error" into a guessing game.
func (c *Config) Validate() error {
	for _, name := range sortedKeys(c.Aliases) {
		a := c.Aliases[name]
		if a.Provider == "" {
			return fmt.Errorf("alias %q: provider is required", name)
		}
		if a.Model == "" {
			return fmt.Errorf("alias %q: model is required", name)
		}
	}

	for _, name := range sortedKeys(c.RateLimits) {
		l := c.RateLimits[name]
		// A rate limit naming an alias that doesn't exist is dead
		// config, and almost always a typo in one of the two names.
		if _, ok := c.Aliases[name]; !ok {
			return fmt.Errorf("ratelimit %q: no alias with that name", name)
		}
		if l.Capacity <= 0 {
			return fmt.Errorf("ratelimit %q: capacity must be > 0", name)
		}
		if l.RefillRate <= 0 {
			return fmt.Errorf("ratelimit %q: refill_rate must be > 0", name)
		}
	}
	return nil
}

// Resolve looks up an alias by its client-facing name.
func (c *Config) Resolve(name string) (Alias, bool) {
	a, ok := c.Aliases[name]
	return a, ok
}

// LimitFor returns the rate-limit policy for an alias. The false return
// means "no limit configured", which callers treat as unlimited — not as
// a zero-capacity bucket that denies everything.
func (c *Config) LimitFor(alias string) (Limit, bool) {
	l, ok := c.RateLimits[alias]
	return l, ok
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
