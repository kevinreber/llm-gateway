package ratelimit_test

import (
	"context"
	"testing"

	"github.com/kevinreber/llm-gateway/internal/ratelimit"
)

func TestAllowAll(t *testing.T) {
	var l ratelimit.Limiter = ratelimit.AllowAll{}

	// The no-bucketd deployment must serve traffic, not deny it — a
	// zero-value Verdict would silently 429 every request.
	v, err := l.Allow(context.Background(), "alias:smart", ratelimit.Limit{})
	if err != nil {
		t.Fatalf("Allow: %v", err)
	}
	if !v.Allowed {
		t.Error("AllowAll denied a request")
	}
	if err := l.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

func TestNewBucketd_RequiresAddresses(t *testing.T) {
	// Catching this at construction keeps a misconfigured BUCKETD_ADDRS
	// from producing a limiter that fails on every request instead.
	if _, err := ratelimit.NewBucketd(nil); err == nil {
		t.Error("NewBucketd(nil) succeeded, want an error")
	}
}
