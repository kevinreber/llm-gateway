package config

import (
	"fmt"
	"time"

	"gopkg.in/yaml.v3"
)

// CachePolicy is an alias's response-cache configuration.
type CachePolicy struct {
	// TTL is how long a response stays cached. Zero or absent disables
	// caching for the alias.
	TTL Duration `yaml:"ttl"`
}

// Duration is a time.Duration that decodes from YAML's "5m" rather than
// from a bare nanosecond count.
//
// yaml.v3 will happily decode 300000000000 into a time.Duration and
// nothing else, which in a hand-edited config is a trap: `ttl: 5`
// parses cleanly and means five nanoseconds. Requiring a unit makes the
// mistake impossible to write.
type Duration time.Duration

// UnmarshalYAML decodes a Go duration string.
func (d *Duration) UnmarshalYAML(node *yaml.Node) error {
	var raw string
	if err := node.Decode(&raw); err != nil {
		return fmt.Errorf("ttl must be a duration string such as \"5m\": %w", err)
	}
	parsed, err := time.ParseDuration(raw)
	if err != nil {
		return fmt.Errorf("ttl %q: %w", raw, err)
	}
	if parsed < 0 {
		return fmt.Errorf("ttl %q must not be negative", raw)
	}
	*d = Duration(parsed)
	return nil
}

// Std returns the value as a time.Duration.
func (d Duration) Std() time.Duration { return time.Duration(d) }

// CacheFor returns the cache policy for an alias.
func (c *Config) CacheFor(alias string) (CachePolicy, bool) {
	p, ok := c.Cache[alias]
	if !ok || p.TTL <= 0 {
		return CachePolicy{}, false
	}
	return p, true
}
