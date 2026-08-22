package config

import (
	"context"
	"log/slog"
	"path/filepath"
	"time"

	"github.com/fsnotify/fsnotify"
)

// DefaultDebounce is how long the watcher waits for a file to stop
// changing before it reloads.
//
// One save is rarely one event. An editor writing in place produces
// WRITE and CHMOD; one writing through a temporary file produces CREATE
// then RENAME. Reloading on each of them would parse a half-written
// file, log a validation error for a config that is about to be fine,
// and do it several times per keystroke-save.
const DefaultDebounce = 150 * time.Millisecond

// Watch reloads store whenever its backing file changes on disk, until
// ctx is cancelled.
//
// It watches the file's *directory* rather than the file, because the
// common ways a config is replaced — an editor saving through a
// temporary file, a deploy writing then renaming, any atomic swap —
// leave the original inode deleted and a path-level watch pointing at
// it.
//
// Whether that actually breaks a file-level watch is backend-specific,
// which is the reason to not depend on it either way: fsnotify's kqueue
// backend re-establishes the watch after a replace, so this survives on
// macOS, while inotify delivers IN_MOVE_SELF and the watch is simply
// over. Depending on the kqueue behavior would mean automatic reload
// quietly not working on Linux, which is where this runs. Watching the
// directory and filtering by name behaves the same everywhere.
//
// A reload that fails is logged and the watcher keeps running. That is
// the behavior that makes this usable: the recovery from a typo is
// saving the file again, not restarting the process. Store.Reload
// leaves the running config in place on failure, so a broken edit costs
// nothing but a log line.
//
// onReload, when set, is called after each successful reload. It runs
// on the watcher's goroutine, so it should not block.
func Watch(ctx context.Context, store *Store, logger *slog.Logger, onReload func(*Config)) error {
	if !store.Reloadable() {
		return ErrNotReloadable
	}
	if logger == nil {
		logger = slog.Default()
	}

	w, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}

	// Resolve before watching so the directory being watched is the one
	// the file is actually in, whatever relative path was configured.
	path, err := filepath.Abs(store.Path())
	if err != nil {
		_ = w.Close()
		return err
	}
	dir, name := filepath.Split(path)
	if err := w.Add(filepath.Clean(dir)); err != nil {
		_ = w.Close()
		return err
	}

	go watchLoop(ctx, w, store, name, logger, onReload)
	logger.Info("watching config for changes", "path", path, "debounce", DefaultDebounce)
	return nil
}

// watchLoop owns the watcher and closes it on the way out.
func watchLoop(
	ctx context.Context,
	w *fsnotify.Watcher,
	store *Store,
	name string,
	logger *slog.Logger,
	onReload func(*Config),
) {
	defer func() { _ = w.Close() }()

	// A stopped timer with a drained channel is the idle state; the
	// select below only ever sees it fire after a relevant event has
	// reset it.
	timer := time.NewTimer(DefaultDebounce)
	if !timer.Stop() {
		<-timer.C
	}
	defer timer.Stop()

	pending := false
	for {
		select {
		case <-ctx.Done():
			return

		case event, ok := <-w.Events:
			if !ok {
				return
			}
			// Chmod alone is not a content change — an editor touching
			// permissions, or a backup tool walking the directory,
			// should not cost a reload.
			if filepath.Base(event.Name) != name || event.Op == fsnotify.Chmod {
				continue
			}
			if pending && !timer.Stop() {
				<-timer.C
			}
			timer.Reset(DefaultDebounce)
			pending = true

		case <-timer.C:
			pending = false
			cfg, err := store.Reload()
			if err != nil {
				// Keep watching. The running config is untouched, and
				// the fix is to save the file again.
				logger.Warn("config reload failed; keeping the running config",
					"path", store.Path(), "err", err)
				continue
			}
			logger.Info("config reloaded from disk",
				"path", store.Path(),
				"aliases", len(cfg.Aliases),
				"ratelimits", len(cfg.RateLimits),
				"fallback_chains", len(cfg.Fallback))
			if onReload != nil {
				onReload(cfg)
			}

		case err, ok := <-w.Errors:
			if !ok {
				return
			}
			logger.Warn("config watcher error", "err", err)
		}
	}
}
