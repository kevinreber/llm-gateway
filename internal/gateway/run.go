// Package gateway wires the HTTP server, providers, config, rate
// limiter, and cost tracker into a single Run(ctx, cfg) entrypoint. The
// shape mirrors bucketd's internal/server.
package gateway

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/kevinreber/llm-gateway/internal/config"
	"github.com/kevinreber/llm-gateway/internal/cost"
	"github.com/kevinreber/llm-gateway/internal/provider"
	"github.com/kevinreber/llm-gateway/internal/ratelimit"
	"github.com/kevinreber/llm-gateway/internal/store"
)

// defaultConfigPath is used when CONFIG_PATH is unset. A missing file at
// this path is not an error — it means "run with no aliases and no rate
// limits", which is a valid single-provider deployment.
const defaultConfigPath = "gateway.yaml"

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
	// tests pair it with AnthropicBaseURL to point at a fake upstream.
	AnthropicAPIKey string
	// AnthropicBaseURL overrides the production endpoint. Empty string
	// means production. Used by integration tests to point at a fake.
	AnthropicBaseURL string
	// ConfigPath is the gateway.yaml location. Empty means "try
	// defaultConfigPath, tolerate its absence".
	ConfigPath string
	// BucketdAddrs are the rate-limiter nodes. Empty disables rate
	// limiting entirely (ratelimit.AllowAll).
	BucketdAddrs []string
	// DatabaseURL is the Postgres DSN for cost tracking. Empty logs
	// cost batches instead of persisting them, which keeps local
	// development useful without standing up Postgres.
	DatabaseURL string
}

// LoadConfigFromEnv reads Config from environment variables. Called by
// cmd/gateway/main.go; not used in unit tests (they build Config directly).
func LoadConfigFromEnv() (Config, error) {
	cfg := Config{
		Addr:             getenv("ADDR", ":8080"),
		ShutdownTimeout:  getenvDuration("SHUTDOWN_TIMEOUT", 15*time.Second),
		AnthropicAPIKey:  os.Getenv("ANTHROPIC_API_KEY"),
		AnthropicBaseURL: os.Getenv("ANTHROPIC_BASE_URL"),
		ConfigPath:       os.Getenv("CONFIG_PATH"),
		BucketdAddrs:     splitList(os.Getenv("BUCKETD_ADDRS")),
		DatabaseURL:      os.Getenv("DATABASE_URL"),
	}
	if cfg.AnthropicAPIKey == "" {
		return cfg, errors.New("ANTHROPIC_API_KEY is required")
	}
	return cfg, nil
}

// Run starts the HTTP server and blocks until ctx is cancelled or the
// server terminates with an error.
//
// Shutdown order matters and is deliberate: drain HTTP first (bounded by
// ShutdownTimeout), then stop the cost writer so requests that finished
// during the drain still get their rows, then close the database and
// limiter connections the writer was using.
func Run(ctx context.Context, cfg Config) error {
	logger := slog.Default()

	gwCfg, err := loadGatewayConfig(cfg.ConfigPath, logger)
	if err != nil {
		return err
	}

	limiter, err := buildLimiter(cfg.BucketdAddrs, logger)
	if err != nil {
		return err
	}
	defer func() { _ = limiter.Close() }()

	sink, closeSink, err := buildCostSink(ctx, cfg.DatabaseURL, logger)
	if err != nil {
		return err
	}
	defer closeSink()

	writer := cost.NewWriter(sink, logger)
	writerCtx, stopWriter := context.WithCancel(context.Background())
	var writerDone sync.WaitGroup
	writerDone.Add(1)
	go func() {
		defer writerDone.Done()
		writer.Run(writerCtx)
	}()
	// Ordered against the deferred sink close above: defers unwind
	// last-registered-first, so the writer stops and flushes before the
	// pool it writes through is closed.
	defer func() {
		stopWriter()
		writerDone.Wait()
	}()

	h := &handler{
		providers: buildProviders(cfg),
		cfg:       gwCfg,
		limiter:   limiter,
		costs:     writer,
		logger:    logger,
	}

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           h.routes(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		logger.Info("gateway listening",
			"addr", cfg.Addr,
			"aliases", len(gwCfg.Aliases),
			"ratelimits", len(gwCfg.RateLimits))
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

// loadGatewayConfig reads gateway.yaml. An explicitly configured path
// that doesn't exist is a hard error — the operator asked for that file,
// so silently ignoring it would run production with no rate limits. The
// default path is allowed to be absent.
func loadGatewayConfig(path string, logger *slog.Logger) (*config.Config, error) {
	explicit := path != ""
	if !explicit {
		path = defaultConfigPath
	}

	cfg, err := config.Load(path)
	if err == nil {
		logger.Info("loaded gateway config", "path", path)
		return cfg, nil
	}
	if explicit || !os.IsNotExist(errors.Unwrap(err)) {
		return nil, err
	}
	logger.Info("no gateway config found; serving direct model names with no rate limits",
		"path", path)
	return &config.Config{}, nil
}

func buildProviders(cfg Config) map[string]provider.Provider {
	anth := provider.NewAnthropic(cfg.AnthropicAPIKey)
	if cfg.AnthropicBaseURL != "" {
		anth = provider.NewAnthropicWithBaseURL(cfg.AnthropicAPIKey, cfg.AnthropicBaseURL)
	}
	return map[string]provider.Provider{
		provider.AnthropicName: anth,
	}
}

func buildLimiter(addrs []string, logger *slog.Logger) (ratelimit.Limiter, error) {
	if len(addrs) == 0 {
		logger.Info("BUCKETD_ADDRS unset; rate limiting disabled")
		return ratelimit.AllowAll{}, nil
	}
	limiter, err := ratelimit.NewBucketd(addrs)
	if err != nil {
		return nil, fmt.Errorf("bucketd client: %w", err)
	}
	logger.Info("rate limiting via bucketd", "nodes", len(addrs))
	return limiter, nil
}

// buildCostSink returns the cost destination and a close function. With
// no DATABASE_URL the sink logs batches rather than dropping them, so
// the cost path is still observable in local development.
func buildCostSink(ctx context.Context, dsn string, logger *slog.Logger) (cost.Sink, func(), error) {
	if dsn == "" {
		logger.Info("DATABASE_URL unset; cost events will be logged, not persisted")
		return cost.LogSink{Logger: logger}, func() {}, nil
	}

	pg, err := store.Open(ctx, dsn)
	if err != nil {
		return nil, nil, fmt.Errorf("cost store: %w", err)
	}
	if err := pg.Migrate(ctx); err != nil {
		pg.Close()
		return nil, nil, fmt.Errorf("migrate: %w", err)
	}
	logger.Info("cost tracking enabled")
	return pg, pg.Close, nil
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

// splitList parses a comma-separated env var, tolerating stray spaces
// and trailing commas.
func splitList(v string) []string {
	if strings.TrimSpace(v) == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
