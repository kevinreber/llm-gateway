// Package config parses and validates gateway.yaml: the alias table that
// maps client-facing model names onto concrete {provider, model} pairs,
// the per-alias rate-limit policies handed to bucketd, the per-provider
// circuit-breaker settings, and the per-alias fallback chains.
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
	"time"

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

// Breaker is a per-provider circuit-breaker policy.
//
// It is keyed by provider rather than by alias because it describes the
// health of an upstream dependency, and every alias pointing at that
// provider shares one connection pool, one API key, and one outage.
// Tracking failures per alias would need `failure_threshold` failures on
// each alias separately before any of them stopped calling a provider
// that is already known to be down.
type Breaker struct {
	// FailureThreshold is the number of consecutive failures that trips
	// the breaker from closed to open.
	FailureThreshold int `yaml:"failure_threshold"`
	// RecoveryTimeout is how long the breaker stays open before it
	// admits a single probe request (half-open).
	RecoveryTimeout time.Duration `yaml:"recovery_timeout"`
}

// Config is a parsed gateway.yaml.
//
// The zero value is usable and means "no aliases, no limits": every
// request is treated as a direct model name and no rate limiting is
// applied. That is deliberately the Phase 1 behavior, so running without
// a config file still serves traffic.
type Config struct {
	Aliases    map[string]Alias   `yaml:"aliases"`
	RateLimits map[string]Limit   `yaml:"ratelimits"`
	Breakers   map[string]Breaker `yaml:"breakers"`

	// Cache maps an alias to its response-cache policy. An alias absent
	// from this table is not cached, which is the safe default: caching
	// changes what a caller gets back, so it has to be asked for.
	Cache map[string]CachePolicy `yaml:"cache"`

	// Fallback maps an alias to the ordered list of aliases to try when
	// it cannot be served.
	//
	// Entries are aliases rather than bare provider names, which is a
	// deliberate departure from the obvious design. A fallback has to
	// name a model, not just a vendor: an alias resolving to
	// {anthropic, claude-sonnet-5} cannot fail over to "openai" without
	// someone deciding which OpenAI model stands in for Sonnet. An alias
	// already carries that {provider, model} pair, so a chain of aliases
	// is the only shape that is complete on its own — and it lets every
	// key and entry be checked against the alias table at parse time.
	//
	// Chains are followed exactly one level deep: the fallbacks of a
	// fallback are never consulted. That makes cycles structurally
	// impossible rather than something to detect, and keeps the worst
	// case for a request equal to the length of one declared list.
	Fallback map[string][]string `yaml:"fallback"`
}

// ErrNotReloadable is returned by Store.Reload when the store was built
// from a value rather than a file, so there is nothing to re-read.
var ErrNotReloadable = errors.New("config: store has no backing file")

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

	// Breaker keys are provider names, which this package deliberately
	// knows nothing about — the provider registry is a property of the
	// binary, not of the file. An entry for a provider that isn't wired
	// is therefore unused rather than invalid, and Run warns about it at
	// startup where the registry is actually in scope.
	for _, name := range sortedKeys(c.Breakers) {
		b := c.Breakers[name]
		if b.FailureThreshold <= 0 {
			return fmt.Errorf("breaker %q: failure_threshold must be > 0", name)
		}
		if b.RecoveryTimeout <= 0 {
			return fmt.Errorf("breaker %q: recovery_timeout must be > 0", name)
		}
	}

	for _, name := range sortedKeys(c.Cache) {
		// Same reasoning as ratelimits: a cache policy naming an alias
		// that does not exist is dead config, and silently not caching
		// is the worst way to discover the typo — the gateway would look
		// like it was working and just be slower and more expensive.
		if _, ok := c.Aliases[name]; !ok {
			return fmt.Errorf("cache %q: no alias with that name", name)
		}
	}

	for _, name := range sortedKeys(c.Fallback) {
		if _, ok := c.Aliases[name]; !ok {
			return fmt.Errorf("fallback %q: no alias with that name", name)
		}
		seen := make(map[string]bool, len(c.Fallback[name]))
		for i, target := range c.Fallback[name] {
			switch {
			case target == "":
				return fmt.Errorf("fallback %q: entry %d is empty", name, i)
			case target == name:
				// Self-fallback would retry the same {provider, model}
				// that just failed, which is what the retry layer
				// already did — and did with backoff, which this would
				// not.
				return fmt.Errorf("fallback %q: cannot fall back to itself", name)
			case seen[target]:
				return fmt.Errorf("fallback %q: duplicate entry %q", name, target)
			}
			if _, ok := c.Aliases[target]; !ok {
				return fmt.Errorf("fallback %q: entry %q is not an alias", name, target)
			}
			seen[target] = true
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

// BreakerFor returns the circuit-breaker policy for a provider. The
// false return means "not configured", which callers turn into the
// package default rather than into a breaker that never trips.
func (c *Config) BreakerFor(provider string) (Breaker, bool) {
	b, ok := c.Breakers[provider]
	return b, ok
}

// FallbackFor returns the ordered aliases to try after alias fails.
// Nil means no fallback, which is also what an explicitly empty list in
// the YAML means — the two are deliberately the same thing, so writing
// `fast: []` to document "this one does not fail over" costs nothing.
func (c *Config) FallbackFor(alias string) []string {
	return c.Fallback[alias]
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
