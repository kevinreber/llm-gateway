package config

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

// settleTimeout is generous on purpose. These tests wait on the OS
// delivering a filesystem event, which is fast but not bounded by
// anything this process controls — a tight deadline here buys nothing
// and produces a test that fails on a loaded CI runner.
const settleTimeout = 5 * time.Second

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// waitFor polls until cond holds or the deadline passes.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(settleTimeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func startWatch(t *testing.T, path string) (*Store, *atomic.Int64) {
	t.Helper()
	initial, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	store := NewStore(path, initial)

	var reloads atomic.Int64
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	if err := Watch(ctx, store, discardLogger(), func(*Config) { reloads.Add(1) }); err != nil {
		t.Fatalf("watch: %v", err)
	}
	return store, &reloads
}

func TestWatch_StaticStoreIsRefused(t *testing.T) {
	err := Watch(context.Background(), Static(&Config{}), discardLogger(), nil)
	if !errors.Is(err, ErrNotReloadable) {
		t.Errorf("err = %v, want ErrNotReloadable", err)
	}
}

func TestWatch_PicksUpAnInPlaceEdit(t *testing.T) {
	path := writeConfig(t, storeYAML)
	store, reloads := startWatch(t, path)

	if err := os.WriteFile(path, []byte(`
aliases:
  smart: { provider: anthropic, model: claude-opus-5 }
`), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}

	waitFor(t, "the edit to be picked up", func() bool {
		return store.Load().Aliases["smart"].Model == "claude-opus-5"
	})
	if reloads.Load() == 0 {
		t.Error("onReload was never called")
	}
}

func TestWatch_SurvivesRepeatedAtomicReplaces(t *testing.T) {
	// The case that motivates watching the directory instead of the
	// file. Editors and deploy tooling replace a config by writing a
	// temporary file and renaming it over the target, which leaves any
	// watch on the original path pointing at a deleted inode.
	//
	// Replaced twice because one replace proves less than it looks: the
	// rename event can fire before a path-level watch dies, and the
	// reload then re-opens the path by name and succeeds anyway.
	//
	// Worth being straight about what this does and does not show. It
	// pins the required behavior — repeated atomic replaces keep getting
	// picked up — on every platform. It does not demonstrate that
	// watching the directory is what achieves that, because fsnotify's
	// kqueue backend re-establishes a file-level watch after a replace,
	// so this passes either way on macOS. The case that separates them
	// is inotify, and it is CI that runs it.
	path := writeConfig(t, storeYAML)
	store, _ := startWatch(t, path)

	replaceWith := func(model string) {
		t.Helper()
		tmp := filepath.Join(filepath.Dir(path), ".gateway.yaml.tmp")
		if err := os.WriteFile(tmp, []byte(
			"aliases:\n  smart: { provider: anthropic, model: "+model+" }\n"), 0o600); err != nil {
			t.Fatalf("write temp: %v", err)
		}
		if err := os.Rename(tmp, path); err != nil {
			t.Fatalf("rename: %v", err)
		}
	}

	replaceWith("claude-opus-5")
	waitFor(t, "the first replacement", func() bool {
		return store.Load().Aliases["smart"].Model == "claude-opus-5"
	})

	replaceWith("claude-haiku-4-5")
	waitFor(t, "the second replacement (a file-level watch is deaf by now)", func() bool {
		return store.Load().Aliases["smart"].Model == "claude-haiku-4-5"
	})
}

func TestWatch_KeepsRunningAfterABadEdit(t *testing.T) {
	// The property that makes this usable: recovery from a typo is
	// saving the file again, not restarting the process. A watcher that
	// gave up on the first parse error would be worse than no watcher,
	// because it would look like it was still working.
	path := writeConfig(t, storeYAML)
	store, _ := startWatch(t, path)

	if err := os.WriteFile(path, []byte("aliases:\n  broken: { provider: anthropic }\n"), 0o600); err != nil {
		t.Fatalf("write bad: %v", err)
	}
	// The running config must survive the bad edit.
	time.Sleep(400 * time.Millisecond)
	if got := store.Load().Aliases["smart"].Model; got != "claude-sonnet-5" {
		t.Fatalf("running config = %q after a bad edit, want it unchanged", got)
	}

	if err := os.WriteFile(path, []byte(`
aliases:
  smart: { provider: anthropic, model: claude-haiku-4-5 }
`), 0o600); err != nil {
		t.Fatalf("write good: %v", err)
	}
	waitFor(t, "recovery after fixing the file", func() bool {
		return store.Load().Aliases["smart"].Model == "claude-haiku-4-5"
	})
}

func TestWatch_DebouncesABurstOfWrites(t *testing.T) {
	// One save is rarely one event, and a rapid sequence of them should
	// cost one reload rather than one per event — otherwise the watcher
	// parses half-written files and logs errors for a config that is
	// about to be fine.
	path := writeConfig(t, storeYAML)
	store, reloads := startWatch(t, path)

	for i := range 10 {
		model := "claude-sonnet-5"
		if i == 9 {
			model = "claude-opus-5"
		}
		if err := os.WriteFile(path, []byte(
			"aliases:\n  smart: { provider: anthropic, model: "+model+" }\n"), 0o600); err != nil {
			t.Fatalf("write %d: %v", i, err)
		}
		time.Sleep(5 * time.Millisecond)
	}

	waitFor(t, "the last write to land", func() bool {
		return store.Load().Aliases["smart"].Model == "claude-opus-5"
	})
	// Let anything still in flight settle before counting.
	time.Sleep(2 * DefaultDebounce)
	if n := reloads.Load(); n > 3 {
		t.Errorf("reloads = %d for a burst of 10 writes; debounce is not coalescing", n)
	}
}

func TestWatch_StopsWithTheContext(t *testing.T) {
	path := writeConfig(t, storeYAML)
	initial, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	store := NewStore(path, initial)

	ctx, cancel := context.WithCancel(context.Background())
	var reloads atomic.Int64
	if err := Watch(ctx, store, discardLogger(), func(*Config) { reloads.Add(1) }); err != nil {
		t.Fatalf("watch: %v", err)
	}
	cancel()
	time.Sleep(100 * time.Millisecond)

	if err := os.WriteFile(path, []byte(
		"aliases:\n  smart: { provider: anthropic, model: claude-opus-5 }\n"), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}
	time.Sleep(3 * DefaultDebounce)

	if n := reloads.Load(); n != 0 {
		t.Errorf("reloads = %d after cancel, want 0", n)
	}
}
