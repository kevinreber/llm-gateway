# llm-gateway

A production-grade LLM API gateway in Go. Reverse-proxies completion requests to Anthropic (with OpenAI, Gemini, and Ollama planned), resolving client-facing aliases to concrete models, enforcing distributed rate limits via [bucketd](https://github.com/kevinreber/bucketd), and recording per-request token cost to Postgres.

Built on the `net/http` standard library — no web framework.

## Status

Working today: HTTP proxy to Anthropic, alias-based routing, per-alias distributed rate limiting, and cost tracking. Circuit-breaker fallback, caching, and metrics are in progress — see [Roadmap](#roadmap).

Not yet tagged for release. The HTTP surface may still change before `v0.1.0`.

## How it works

```mermaid
flowchart LR
    C[Client<br/>model: smart] --> M[POST /v1/messages]

    subgraph GW [llm-gateway]
        M --> R[Alias router<br/>gateway.yaml]
        R --> L[Rate limiter]
        L --> P[Provider client]
        P --> T[Cost tracker<br/>buffered channel]
    end

    L -.->|denied| E[429 + Retry-After]
    T -.->|batched every 1s| DB[(Postgres<br/>costs)]
    L <-->|Allow RPC| B[(bucketd)]
    P --> A[Anthropic<br/>Messages API]

    A --> RESP[200 + usage]
```

A request carries an alias rather than a model ID. The gateway resolves `smart` to `{provider: anthropic, model: claude-sonnet-5}` from `gateway.yaml`, takes a token from that alias's bucket, forwards the resolved model upstream, and emits a cost event. Editing the YAML shifts traffic without touching code.

Three response headers report what actually served the request: `X-Gateway-Provider`, `X-Gateway-Model`, and `X-Gateway-Alias`. Without them an alias is a black box — you cannot tell a request that went to Anthropic from one that fell back elsewhere by reading the body.

### Design decisions worth knowing

- **Cost tracking never blocks a request.** Events go to a bounded buffered channel; a single writer goroutine batches them into Postgres via `COPY` every second or every 100 events, whichever comes first. When the buffer fills, events are counted and dropped rather than applying backpressure — a slow database degrades accounting, not availability. The drop count is logged at shutdown so the loss is visible.
- **Shutdown drains in order.** SIGTERM drains in-flight HTTP requests first (bounded by `SHUTDOWN_TIMEOUT`), then stops the cost writer so requests that finished during the drain still get their rows, then closes the database pool.
- **The rate limiter fails open.** If bucketd is unreachable, the request is allowed and the error logged. A rate-limiter outage should degrade enforcement, not take the gateway down with it. This is the right trade for a limiter that controls spend; a limiter guarding correctness would want the opposite.
- **Config parsing is strict.** An unknown key in `gateway.yaml` is a startup error. A typo should never silently disable a rate limit.
- **Every dependency is optional.** With no `gateway.yaml`, clients name concrete models directly. With no `BUCKETD_ADDRS`, nothing is rate limited. With no `DATABASE_URL`, cost batches are logged instead of persisted. The gateway runs usefully with none of them configured.

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...
go run ./cmd/gateway
```

```bash
curl -s localhost:8080/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model": "smart", "messages": [{"role": "user", "content": "hello"}]}'
```

`gateway.yaml` in the working directory is picked up automatically. Without it, pass a real model ID (`claude-sonnet-5`) instead of an alias.

## Configuration

### Alias table (`gateway.yaml`)

```yaml
aliases:
  fast:  { provider: anthropic, model: claude-haiku-4-5 }
  smart: { provider: anthropic, model: claude-sonnet-5 }
  best:  { provider: anthropic, model: claude-opus-5 }

# Token-bucket policy per alias, enforced across every gateway instance.
# An alias with no entry here is unlimited.
ratelimits:
  fast:  { capacity: 100, refill_rate: 50 }
  smart: { capacity: 50, refill_rate: 20 }
  best:  { capacity: 10, refill_rate: 2 }
```

`capacity` is the burst ceiling; `refill_rate` is the sustained rate in tokens per second. One request costs one token.

### Environment

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Anthropic `x-api-key` value. |
| `ADDR` | `:8080` | HTTP listen address. |
| `CONFIG_PATH` | `gateway.yaml` | Alias config. A missing file at the default path is tolerated; a missing file at an explicitly configured path is a startup error. |
| `BUCKETD_ADDRS` | — | Comma-separated bucketd nodes. Unset disables rate limiting. |
| `DATABASE_URL` | — | Postgres DSN for cost tracking. Unset logs cost batches instead of persisting them. |
| `SHUTDOWN_TIMEOUT` | `15s` | How long to drain in-flight requests on SIGTERM. |
| `ANTHROPIC_BASE_URL` | production | Override the upstream endpoint. Used by tests. |

Migrations in `migrations/` are embedded in the binary and applied at startup when `DATABASE_URL` is set — no sidecar container and no separate deploy step. Concurrent replicas are safe: each migration is claimed through a ledger row before it runs.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/messages` | Proxy a completion. Body matches the provider-agnostic `Request` shape. |
| `GET` | `/healthz` | Liveness. Returns 200 while the process is serving. |

`/metrics` and the admin API arrive with the observability work below.

## Development

```bash
go test ./...              # unit tests; Postgres tests skip without a database
go test -race -cover ./...
```

The cost-store tests need a real Postgres and skip when `TEST_DATABASE_URL` is unset, so the suite stays runnable without Docker. CI provides one via a service container, which is where those tests actually execute:

```bash
TEST_DATABASE_URL='postgres://user:pass@localhost:5432/gateway_test?sslmode=disable' go test ./internal/store/
```

## Roadmap

Delivered:

1. **HTTP skeleton and Anthropic provider** — `POST /v1/messages`, graceful shutdown, typed upstream errors with `Retry-After` pass-through.
2. **Rate limiting and cost tracking** — alias routing, per-alias limits via bucketd, batched cost writes to Postgres.

Planned:

3. **Resilience** — per-provider circuit breaker (`sync.RWMutex` fast path on the read-dominated state check), retry with jittered backoff, and a declarative fallback chain.
4. **Observability** — Prometheus metrics for request counts, latency, breaker state and cost totals; structured JSON logs; an admin API; hot config reload via `fsnotify` and an `atomic.Value` swap for a lock-free read path.
5. **Caching and providers** — exact cache keyed on a SHA-256 of the canonicalized request, optional pgvector semantic cache behind it, plus OpenAI, Gemini, and Ollama clients.
6. **Deployment** — multi-stage distroless image, Fly.io config, chaos testing against injected upstream failures.

## License

MIT. See [LICENSE](LICENSE).
