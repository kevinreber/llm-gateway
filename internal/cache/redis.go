package cache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/kevinreber/llm-gateway/internal/provider"
	"github.com/redis/go-redis/v9"
)

// opTimeout bounds a single cache operation.
//
// A cache is only worth having if it is faster than the thing it
// replaces, and the thing it replaces here is an LLM call measured in
// seconds. But an unbounded Redis call on a degraded network inverts
// that: the lookup becomes the slowest part of a request that was
// supposed to be the fast path. 100ms is generous for a local Redis and
// short enough that giving up and calling the provider is still the
// better outcome.
const opTimeout = 100 * time.Millisecond

// Redis is a Cache backed by a Redis string per entry.
type Redis struct {
	client *redis.Client
}

// NewRedis connects to a Redis instance from a redis:// URL.
//
// Does not ping. A cache that refuses to start because its backend is
// briefly unavailable would take down a gateway that is perfectly
// capable of serving every request by calling providers directly, which
// is the opposite of what a cache is for. Failures surface per
// operation, where they degrade to a miss.
func NewRedis(url string) (*Redis, error) {
	opt, err := redis.ParseURL(url)
	if err != nil {
		return nil, fmt.Errorf("parse redis url: %w", err)
	}
	return &Redis{client: redis.NewClient(opt)}, nil
}

// Get returns a cached response, reporting a miss when there is none.
//
// A decode failure is treated as a miss rather than an error worth
// propagating, because there is exactly one way it happens: an entry
// written by a different version of this code. Refusing to serve the
// request would turn a rolling deploy into an outage; ignoring the
// entry means the new response simply overwrites it.
func (r *Redis) Get(ctx context.Context, key string) (*provider.Response, bool, error) {
	opCtx, cancel := context.WithTimeout(ctx, opTimeout)
	defer cancel()

	raw, err := r.client.Get(opCtx, key).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}

	var resp provider.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, false, nil
	}
	return &resp, true, nil
}

// Set stores a response under key for ttl. A non-positive ttl is a
// no-op rather than an entry that never expires: "cache this forever"
// is never what a missing config value meant.
func (r *Redis) Set(ctx context.Context, key string, resp *provider.Response, ttl time.Duration) error {
	if ttl <= 0 {
		return nil
	}
	b, err := json.Marshal(resp)
	if err != nil {
		return err
	}

	opCtx, cancel := context.WithTimeout(ctx, opTimeout)
	defer cancel()
	return r.client.Set(opCtx, key, b, ttl).Err()
}

func (r *Redis) Close() error { return r.client.Close() }
