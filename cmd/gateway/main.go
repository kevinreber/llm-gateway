// Command gateway is the LLM API gateway server. It resolves client
// aliases to concrete models, enforces per-alias rate limits via bucketd,
// reverse-proxies the completion to a provider (Anthropic today), and
// records what the request cost. Circuit-breaker fallback across multiple
// providers lands next; see the roadmap in README.md.
//
// Configuration is via environment variables (see internal/gateway/run.go).
// Shutdown is graceful: SIGTERM/SIGINT triggers Run's shutdown path,
// bounded by SHUTDOWN_TIMEOUT (default 15s).
package main

import (
	"context"
	"log"
	"os/signal"
	"syscall"

	"github.com/kevinreber/llm-gateway/internal/gateway"
)

// version is stamped at build time via
// -ldflags="-X main.version=...". The Dockerfile passes it from a
// VERSION build arg. It defaults to "dev" so a `go build` with no flags
// still produces something honest rather than an empty string.
var version = "dev"

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	cfg, err := gateway.LoadConfigFromEnv()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}
	cfg.Version = version

	if err := gateway.Run(ctx, cfg); err != nil {
		log.Fatalf("run: %v", err)
	}
}
