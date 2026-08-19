// Package ratelimit adapts github.com/kevinreber/bucketd/client to the
// gateway's request path.
//
// The Limiter interface exists so the proxy layer can be tested without
// a running bucketd cluster, and so a deployment with no bucketd
// configured degrades to AllowAll rather than failing to boot.
package ratelimit

import (
	"context"
	"time"

	"github.com/kevinreber/bucketd/client"
)

// Limit is a token-bucket policy: Capacity tokens of burst, refilled at
// RefillRate tokens per second.
type Limit struct {
	Capacity   int32
	RefillRate float64
}

// Verdict is the outcome of an Allow call.
//
// Remaining is advisory only — bucketd computes it at decision time and
// concurrent callers may already have consumed more. Use it for client
// hints, never for scheduling or fairness decisions.
type Verdict struct {
	Allowed    bool
	Remaining  int32
	RetryAfter time.Duration
}

// Limiter is the rate-limiting dependency of the proxy layer.
//
// Implementations must be safe for concurrent use: one Limiter is shared
// across every request goroutine.
type Limiter interface {
	// Allow asks for one token against key under the given policy.
	// A non-nil error means the limiter itself failed (network, dead
	// node) — it does NOT mean "denied". Callers decide the fail-open
	// vs fail-closed policy.
	Allow(ctx context.Context, key string, limit Limit) (Verdict, error)

	// Close releases any underlying connections.
	Close() error
}

// AllowAll is the limiter used when no bucketd addresses are configured.
// Every request is allowed. This keeps local development and the Phase 1
// deployment shape working without standing up a bucketd cluster.
type AllowAll struct{}

// Allow implements Limiter.
func (AllowAll) Allow(context.Context, string, Limit) (Verdict, error) {
	return Verdict{Allowed: true}, nil
}

// Close implements Limiter.
func (AllowAll) Close() error { return nil }

// Bucketd is a Limiter backed by a bucketd cluster. The underlying
// client consistently hashes keys across nodes, so the same alias always
// lands on the same bucketd instance and bucket state stays coherent
// across gateway replicas.
type Bucketd struct {
	cli *client.Client
}

// NewBucketd dials (lazily) the given bucketd addresses.
func NewBucketd(addrs []string) (*Bucketd, error) {
	cli, err := client.New(addrs)
	if err != nil {
		return nil, err
	}
	return &Bucketd{cli: cli}, nil
}

// Allow implements Limiter. One request costs one token; token cost is
// uniform for now because we rate-limit on requests, not on tokens
// consumed by the model. Cost-weighted limiting would take input token
// count as the bucket cost instead, but that isn't knowable until after
// the upstream call.
func (b *Bucketd) Allow(ctx context.Context, key string, limit Limit) (Verdict, error) {
	v, err := b.cli.Allow(ctx, key, 1, client.Limit{
		Capacity:   limit.Capacity,
		RefillRate: limit.RefillRate,
	})
	if err != nil {
		return Verdict{}, err
	}
	return Verdict{
		Allowed:    v.Allowed,
		Remaining:  v.Remaining,
		RetryAfter: time.Duration(v.RetryAfterMs) * time.Millisecond,
	}, nil
}

// Close implements Limiter.
func (b *Bucketd) Close() error { return b.cli.Close() }
