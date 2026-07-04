// Command gateway is the LLM API gateway server. It reverse-proxies
// completion requests to an upstream provider (Phase 1: Anthropic only),
// with the shape ready to grow into alias-based multi-provider routing,
// rate limiting via bucketd, cost tracking, and circuit-breaker fallback
// in later phases.
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

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	cfg, err := gateway.LoadConfigFromEnv()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	if err := gateway.Run(ctx, cfg); err != nil {
		log.Fatalf("run: %v", err)
	}
}
