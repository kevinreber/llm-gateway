// Package gateway wires the HTTP server, provider(s), and (in later
// phases) rate limiter + circuit breaker + cost tracker into a single
// Run(ctx, cfg) entrypoint. The shape mirrors bucketd's internal/server.
package gateway

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/kevinreber/llm-gateway/internal/provider"
)

// Config is the runtime configuration for the gateway. Populate it
// yourself in tests; use LoadConfigFromEnv in production.
type Config struct {
	// Addr is the HTTP listen address, e.g. ":8080".
	Addr string
	// ShutdownTimeout bounds how long we wait for in-flight requests
	// to drain after receiving SIGTERM. Fly.io sends SIGINT ~5s before
	// SIGKILL; keep this comfortably under whatever the platform grants.
	ShutdownTimeout time.Duration
	// AnthropicAPIKey is the x-api-key value. Required in production;
	// tests supply their own via a pre-constructed Provider (see Run's
	// docstring for the injection story once we add DI in Phase 2).
	AnthropicAPIKey string
	// AnthropicBaseURL overrides the production endpoint. Empty string
	// means production. Used by integration tests to point at a fake.
	AnthropicBaseURL string
}

// LoadConfigFromEnv reads Config from environment variables. Called by
// cmd/gateway/main.go; not used in unit tests (they build Config directly).
func LoadConfigFromEnv() (Config, error) {
	cfg := Config{
		Addr:             getenv("ADDR", ":8080"),
		ShutdownTimeout:  getenvDuration("SHUTDOWN_TIMEOUT", 15*time.Second),
		AnthropicAPIKey:  os.Getenv("ANTHROPIC_API_KEY"),
		AnthropicBaseURL: os.Getenv("ANTHROPIC_BASE_URL"),
	}
	if cfg.AnthropicAPIKey == "" {
		return cfg, errors.New("ANTHROPIC_API_KEY is required")
	}
	return cfg, nil
}

// Run starts the HTTP server and blocks until ctx is cancelled or the
// server terminates with an error. Shutdown drains up to ShutdownTimeout.
//
// Later phases add rate limiting, cost tracking, and hot-reload; the
// signature is stable — new dependencies attach via Config.
func Run(ctx context.Context, cfg Config) error {
	logger := slog.Default()

	anth := buildAnthropic(cfg)
	handler := newHandler(anth, logger)

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           handler.routes(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		logger.Info("gateway listening", "addr", cfg.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
			return
		}
		errCh <- nil
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		logger.Info("gateway shutting down", "timeout", cfg.ShutdownTimeout)
		shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("shutdown: %w", err)
		}
		return nil
	}
}

func buildAnthropic(cfg Config) provider.Provider {
	if cfg.AnthropicBaseURL != "" {
		return provider.NewAnthropicWithBaseURL(cfg.AnthropicAPIKey, cfg.AnthropicBaseURL)
	}
	return provider.NewAnthropic(cfg.AnthropicAPIKey)
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getenvDuration(key string, def time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	if d, err := time.ParseDuration(v); err == nil {
		return d
	}
	// Also accept a bare integer count of seconds.
	if n, err := strconv.Atoi(v); err == nil {
		return time.Duration(n) * time.Second
	}
	return def
}
