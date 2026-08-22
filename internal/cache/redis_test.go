package cache

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/provider"
)

// newTestRedis connects to the instance named by TEST_REDIS_URL, or
// skips. Same convention as internal/store's Postgres tests: a laptop
// without Redis can still run `go test ./...`, and CI sets the variable
// so these actually run somewhere.
func newTestRedis(t *testing.T) *Redis {
	t.Helper()
	url := os.Getenv("TEST_REDIS_URL")
	if url == "" {
		t.Skip("TEST_REDIS_URL not set")
	}
	c, err := NewRedis(url)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func TestRedis_RoundTrip(t *testing.T) {
	c := newTestRedis(t)
	ctx := context.Background()
	key := Key("anthropic", "claude-sonnet-5", req())

	want := &provider.Response{
		Content:    "cached body",
		Model:      "claude-sonnet-5",
		StopReason: "end_turn",
		Usage:      provider.Usage{InputTokens: 10, OutputTokens: 20},
	}
	if err := c.Set(ctx, key, want, time.Minute); err != nil {
		t.Fatalf("set: %v", err)
	}

	got, hit, err := c.Get(ctx, key)
	if err != nil || !hit {
		t.Fatalf("Get = (hit %v, err %v), want a hit", hit, err)
	}
	if got.Content != want.Content || got.Model != want.Model {
		t.Errorf("got %+v, want %+v", got, want)
	}
	// Usage has to survive: it is what the cost row would have been
	// built from, and a hit that lost it would silently zero the
	// token accounting for every repeat request.
	if got.Usage != want.Usage {
		t.Errorf("usage = %+v, want %+v", got.Usage, want.Usage)
	}
}

func TestRedis_MissOnUnknownKey(t *testing.T) {
	c := newTestRedis(t)

	_, hit, err := c.Get(context.Background(), "llmgw:exact:definitely-not-stored")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if hit {
		t.Error("reported a hit for a key never written")
	}
}

func TestRedis_NonPositiveTTLStoresNothing(t *testing.T) {
	// "Cache this forever" is never what a missing config value meant.
	c := newTestRedis(t)
	ctx := context.Background()
	key := Key("anthropic", "ttl-zero-test", req())

	if err := c.Set(ctx, key, &provider.Response{Content: "x"}, 0); err != nil {
		t.Fatalf("set: %v", err)
	}
	if _, hit, _ := c.Get(ctx, key); hit {
		t.Error("a zero TTL wrote an entry")
	}
}

func TestRedis_CorruptEntryReadsAsAMiss(t *testing.T) {
	// The one way this happens is an entry written by a different
	// version of this code. Failing the request would turn a rolling
	// deploy into an outage.
	c := newTestRedis(t)
	ctx := context.Background()
	key := "llmgw:exact:corrupt-entry-test"

	if err := c.client.Set(ctx, key, []byte("{not json"), time.Minute).Err(); err != nil {
		t.Fatalf("seed: %v", err)
	}
	resp, hit, err := c.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get returned an error rather than a miss: %v", err)
	}
	if hit || resp != nil {
		t.Error("a corrupt entry was reported as a hit")
	}
}

func TestRedis_UnreachableBackendDegradesToAnError(t *testing.T) {
	// Never a panic and never a hang: the handler turns this into a
	// miss and calls the provider.
	c, err := NewRedis("redis://127.0.0.1:1")
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	defer func() { _ = c.Close() }()

	start := time.Now()
	if _, hit, err := c.Get(context.Background(), "k"); hit || err == nil {
		t.Errorf("Get = (hit %v, err %v), want a failure", hit, err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("took %s; the op timeout is not bounding this", elapsed)
	}
}
