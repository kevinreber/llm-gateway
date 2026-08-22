package gateway

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/cache"
	"github.com/kevinreber/llm-gateway/internal/provider"
)

// fakeCache is an in-memory Cache with injectable failures.
type fakeCache struct {
	mu      sync.Mutex
	entries map[string]*provider.Response
	getErr  error
	setErr  error
	sets    []string
}

func newFakeCache() *fakeCache {
	return &fakeCache{entries: make(map[string]*provider.Response)}
}

func (f *fakeCache) Get(_ context.Context, key string) (*provider.Response, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.getErr != nil {
		return nil, false, f.getErr
	}
	resp, ok := f.entries[key]
	return resp, ok, nil
}

func (f *fakeCache) Set(_ context.Context, key string, resp *provider.Response, _ time.Duration) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.setErr != nil {
		return f.setErr
	}
	f.sets = append(f.sets, key)
	f.entries[key] = resp
	return nil
}

func (f *fakeCache) Close() error { return nil }

func (f *fakeCache) size() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.entries)
}

func (f *fakeCache) has(key string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	_, ok := f.entries[key]
	return ok
}

func withCache(t *testing.T) (*fallbackHarness, *fakeCache) {
	t.Helper()
	hn := newFallbackHarness(t, noRetry())
	fc := newFakeCache()
	hn.h.cache = fc
	return hn, fc
}

func TestCache_SecondIdenticalRequestSkipsTheProvider(t *testing.T) {
	hn, _ := withCache(t)

	if rec := hn.post(t, "smart"); rec.Code != http.StatusOK {
		t.Fatalf("first post = %d", rec.Code)
	}
	if hn.anth.calls != 1 {
		t.Fatalf("provider calls after first request = %d, want 1", hn.anth.calls)
	}

	rec := hn.post(t, "smart")
	if rec.Code != http.StatusOK {
		t.Fatalf("second post = %d", rec.Code)
	}
	if hn.anth.calls != 1 {
		t.Errorf("provider calls after a cache hit = %d, want still 1", hn.anth.calls)
	}
	if got := rec.Header().Get("X-Gateway-Cache"); got != "hit" {
		t.Errorf("X-Gateway-Cache = %q, want hit", got)
	}
}

func TestCache_MissIsLabelledOnTheResponse(t *testing.T) {
	hn, _ := withCache(t)

	if got := hn.post(t, "smart").Header().Get("X-Gateway-Cache"); got != "miss" {
		t.Errorf("X-Gateway-Cache = %q, want miss", got)
	}
}

func TestCache_HitIsNotBilled(t *testing.T) {
	// A hit costs nothing upstream, so recording a cost row for it
	// would overstate spend by exactly the amount the cache saved —
	// which would make a working cache look like it had no effect.
	hn, _ := withCache(t)

	hn.post(t, "smart")
	hn.post(t, "smart")

	if n := len(hn.costs.events); n != 1 {
		t.Errorf("cost events = %d, want 1 (the miss only)", n)
	}
}

func TestCache_UncachedAliasIsNeverStored(t *testing.T) {
	// `lonely` has no cache policy. Caching an alias nobody asked to
	// cache would change what its callers get back.
	hn, fc := withCache(t)

	hn.post(t, "lonely")
	hn.post(t, "lonely")

	if fc.size() != 0 {
		t.Errorf("cache holds %d entries for an alias with no policy", fc.size())
	}
	if hn.anth.calls != 2 {
		t.Errorf("provider calls = %d, want 2 — nothing should have been deflected", hn.anth.calls)
	}
}

func TestCache_DirectModelNamesAreNeverCached(t *testing.T) {
	hn, fc := withCache(t)

	hn.post(t, "claude-sonnet-5")

	if fc.size() != 0 {
		t.Error("a direct model name was cached; there is no policy that could have asked for it")
	}
}

func TestCache_FallbackResponseIsNotStoredUnderThePrimaryKey(t *testing.T) {
	// Storing it under the primary's key would serve the stand-in to
	// every later request for that alias until the entry expired,
	// silently pinning traffic to it long after the primary recovered —
	// and with no fallback header, because as far as those requests are
	// concerned nothing failed.
	hn, fc := withCache(t)
	hn.anth.err = &provider.APIError{Provider: "anthropic", Status: 503, Message: "down"}
	hn.openai.resp = &provider.Response{
		Content: "from openai", Model: "gpt-4o",
		Usage: provider.Usage{InputTokens: 1, OutputTokens: 1},
	}

	rec := hn.post(t, "smart")
	if rec.Code != http.StatusOK {
		t.Fatalf("post = %d, want 200 via fallback", rec.Code)
	}
	if rec.Header().Get("X-Gateway-Fallback") == "" {
		t.Fatal("expected this request to have fallen back")
	}

	primaryKey := cache.Key("anthropic", "claude-sonnet-5", &provider.Request{
		Model:    "smart",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	})
	if fc.has(primaryKey) {
		t.Error("the fallback's response was stored under the primary's key")
	}

	// And the recovered primary is genuinely retried rather than
	// answered from cache.
	hn.anth.err = nil
	rec = hn.post(t, "smart")
	if got := rec.Header().Get("X-Gateway-Provider"); got != "anthropic" {
		t.Errorf("after recovery, served by %q, want anthropic", got)
	}
	if rec.Header().Get("X-Gateway-Cache") != "miss" {
		t.Error("the recovered primary request was answered from cache")
	}
}

func TestCache_LookupFailureFallsThroughToTheProvider(t *testing.T) {
	// A cache is an optimization. An unreachable backend costs latency,
	// never availability.
	hn, fc := withCache(t)
	fc.getErr = errors.New("redis is on fire")

	rec := hn.post(t, "smart")
	if rec.Code != http.StatusOK {
		t.Fatalf("post = %d, want 200 despite the cache being down", rec.Code)
	}
	if hn.anth.calls != 1 {
		t.Errorf("provider calls = %d, want 1", hn.anth.calls)
	}
}

func TestCache_StoreFailureDoesNotFailTheRequest(t *testing.T) {
	hn, fc := withCache(t)
	fc.setErr = errors.New("redis is read-only")

	if rec := hn.post(t, "smart"); rec.Code != http.StatusOK {
		t.Errorf("post = %d, want 200 despite the store failing", rec.Code)
	}
}

func TestCache_DifferentPromptsDoNotShareAnEntry(t *testing.T) {
	hn, _ := withCache(t)

	hn.post(t, "smart")
	rec := hn.postBody(t, `{"model":"smart","messages":[{"role":"user","content":"different"}]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("post = %d", rec.Code)
	}
	if hn.anth.calls != 2 {
		t.Errorf("provider calls = %d, want 2 — a different prompt must not hit", hn.anth.calls)
	}
}

func TestCache_HitPreservesTheRoutingHeaders(t *testing.T) {
	// A cached answer still has to say which provider and model
	// produced it, or an alias becomes a black box the moment caching
	// is turned on.
	hn, _ := withCache(t)

	hn.post(t, "smart")
	rec := hn.post(t, "smart")

	if got := rec.Header().Get("X-Gateway-Provider"); got != "anthropic" {
		t.Errorf("provider header = %q", got)
	}
	if got := rec.Header().Get("X-Gateway-Model"); got != "claude-sonnet-5" {
		t.Errorf("model header = %q", got)
	}
	if got := rec.Header().Get("X-Gateway-Alias"); got != "smart" {
		t.Errorf("alias header = %q", got)
	}
}

// postBody sends a raw request body, for the cases where the harness's
// fixed prompt is the thing under test.
func (hn *fallbackHarness) postBody(t *testing.T, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	hn.h.routes().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)))
	return rec
}
