// Package cache holds the gateway's response caches.
//
// The exact cache keys on a hash of the resolved request and deflects
// an identical retry before it reaches a provider. A semantic cache
// keyed on embedding similarity sits behind it later; both share the
// Cache interface here so the handler consults them the same way.
package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"time"

	"github.com/kevinreber/llm-gateway/internal/provider"
)

// Layer names the cache tier a lookup hit, for metrics and logs.
const (
	LayerExact = "exact"
)

// Cache stores completions keyed by request hash.
//
// Implementations must be safe for concurrent use and must never make a
// request fail: a cache is an optimization, and an unreachable Redis
// should cost latency at worst, not availability. Get reports a miss
// alongside its error so a caller can ignore the error and proceed.
type Cache interface {
	Get(ctx context.Context, key string) (*provider.Response, bool, error)
	Set(ctx context.Context, key string, resp *provider.Response, ttl time.Duration) error
	Close() error
}

// Disabled is the no-op cache used when no backend is configured.
type Disabled struct{}

func (Disabled) Get(context.Context, string) (*provider.Response, bool, error) {
	return nil, false, nil
}

func (Disabled) Set(context.Context, string, *provider.Response, time.Duration) error { return nil }

func (Disabled) Close() error { return nil }

// keyInput is the canonical form hashed into a cache key.
//
// A struct rather than the client's raw bytes, and that is the whole
// point of it: two callers can send the same completion with different
// key order, different whitespace, or an omitted default and mean
// exactly the same request. Hashing the parsed value collapses all of
// that, because Go marshals struct fields in declaration order and this
// type has no maps in it to iterate randomly.
//
// Message text is deliberately not normalized. Trimming or collapsing
// whitespace inside a prompt would make two genuinely different inputs
// share a cache entry, and leading whitespace is load-bearing often
// enough in prompt engineering that quietly discarding it would be the
// cache changing the answer.
type keyInput struct {
	// Provider and Model are the resolved destination, not the alias
	// the client named. Two aliases pointing at the same model should
	// share entries — that is a feature, and it is also why the alias
	// itself must not be in the key.
	Provider string `json:"provider"`
	Model    string `json:"model"`

	System   string             `json:"system"`
	Messages []provider.Message `json:"messages"`

	// MaxTokens and Temperature change the output, so they change the
	// key. Temperature especially: it is the one field where a hit
	// returns a sample the caller might have expected to differ.
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`

	// Version namespaces every key, so a change to what a cached entry
	// means can invalidate the whole space by incrementing rather than
	// by flushing a shared Redis somebody else is also using.
	Version int `json:"v"`
}

// keyVersion is bumped whenever the meaning of a cached entry changes.
const keyVersion = 1

// Key derives the cache key for a request resolved to a concrete
// provider and model.
//
// SHA-256 rather than a faster non-cryptographic hash: entries are
// keyed by user-supplied content, so a caller able to find collisions
// could serve one tenant's completion to another. That is a
// confidentiality bug, not a performance one, and the hash is not the
// expensive part of a request that is about to cross the internet.
func Key(providerName, model string, req *provider.Request) string {
	input := keyInput{
		Provider:    providerName,
		Model:       model,
		System:      req.System,
		Messages:    req.Messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Version:     keyVersion,
	}
	// json.Marshal on this type cannot fail: every field is a string,
	// int, float, or a slice of two strings. Errors would only come
	// from unsupported types (channels, funcs) or NaN/Inf floats, and
	// Temperature reaching NaN would have failed JSON decoding on the
	// way in.
	b, _ := json.Marshal(input)
	sum := sha256.Sum256(b)
	return "llmgw:" + LayerExact + ":" + hex.EncodeToString(sum[:])
}
