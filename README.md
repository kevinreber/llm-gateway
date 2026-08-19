# llm-gateway

A production-grade LLM API gateway in Go. Reverse-proxies completion requests to Anthropic and OpenAI (with Gemini and Ollama planned), resolving client-facing aliases to concrete models, enforcing distributed rate limits via [bucketd](https://github.com/kevinreber/bucketd), recording per-request token cost to Postgres, and failing over between providers when one degrades.

Built on the `net/http` standard library — no web framework.

## Status

Working today: HTTP proxy to Anthropic and OpenAI, alias-based routing, per-alias distributed rate limiting, cost tracking, and resilience — per-provider circuit breaker, bounded retry with jittered backoff, and a declarative fallback chain. Caching and metrics are in progress — see [Roadmap](#roadmap).

Not yet tagged for release. The HTTP surface may still change before `v0.1.0`.

## How it works

```mermaid
flowchart LR
    C[Client<br/>model: smart] --> M[POST /v1/messages]

    subgraph GW [llm-gateway]
        M --> R[Alias router<br/>gateway.yaml]
        R --> L[Rate limiter]
        L --> CH[Fallback chain<br/>smart, smart-alt, fast]
        CH --> BR[Breaker + retry<br/>per provider]
        BR --> T[Cost tracker<br/>buffered channel]
    end

    L -.->|denied| E[429 + Retry-After]
    L <-->|Allow RPC| B[(bucketd)]
    T -.->|batched every 1s| DB[(Postgres<br/>costs)]

    BR --> A[Anthropic<br/>Messages API]
    BR -.->|primary open or 5xx| O[OpenAI<br/>Chat Completions]
    BR -.->|every provider open| U[503 + Retry-After]

    A --> RESP[200 + usage]
    O --> RESP
```

A request carries an alias rather than a model ID. The gateway resolves `smart` to `{provider: anthropic, model: claude-sonnet-5}` from `gateway.yaml`, takes a token from that alias's bucket, forwards the resolved model upstream, and emits a cost event. Editing the YAML shifts traffic without touching code.

When the resolved provider cannot serve the request, the gateway walks that alias's fallback chain. Each hop is another alias, so each one carries its own provider and model, and the request is retried against a model that provider can actually serve. Cost is attributed to whichever provider ultimately answered, at that provider's rates.

Four response headers report what actually served the request: `X-Gateway-Provider`, `X-Gateway-Model`, `X-Gateway-Alias`, and — only when the request failed over — `X-Gateway-Fallback`, naming the alias that stepped in. Without them an alias is a black box; you cannot tell a request that went to Anthropic from one that fell over to OpenAI by reading the body.

### Design decisions worth knowing

- **Cost tracking never blocks a request.** Events go to a bounded buffered channel; a single writer goroutine batches them into Postgres via `COPY` every second or every 100 events, whichever comes first. When the buffer fills, events are counted and dropped rather than applying backpressure — a slow database degrades accounting, not availability. The drop count is logged at shutdown so the loss is visible.
- **Shutdown drains in order.** SIGTERM drains in-flight HTTP requests first (bounded by `SHUTDOWN_TIMEOUT`), then stops the cost writer so requests that finished during the drain still get their rows, then closes the database pool.
- **The rate limiter fails open.** If bucketd is unreachable, the request is allowed and the error logged. A rate-limiter outage should degrade enforcement, not take the gateway down with it. This is the right trade for a limiter that controls spend; a limiter guarding correctness would want the opposite.
- **Config parsing is strict.** An unknown key in `gateway.yaml` is a startup error. A typo should never silently disable a rate limit.
- **Every dependency is optional.** With no `gateway.yaml`, clients name concrete models directly. With no `BUCKETD_ADDRS`, nothing is rate limited. With no `DATABASE_URL`, cost batches are logged instead of persisted. One provider key is enough to boot. The gateway runs usefully with none of the rest configured.
- **Fallback chains are lists of aliases, not lists of providers.** "Fail over from Anthropic to OpenAI" is not actionable until someone decides which OpenAI model stands in for Sonnet — and an alias is exactly that decision, already written down. Keying the chain on aliases means every entry names a model its provider can serve, and every name in the block is checked against the alias table at parse time instead of turning out to be dead at 3am. Chains are followed one level deep, which makes cycles structurally impossible rather than something to detect.
- **Retry and the breaker read typed errors, never message text.** `provider.APIError` carries `Status`, `Type`, and `RetryAfter`, so a reworded upstream error message cannot silently change routing behavior. A 5xx is retried and counts against the provider's health. A 429 is retried but does *not* trip the breaker — a rate-limited provider is a healthy provider telling us to slow down, and taking it out of rotation would punish every caller for one noisy alias. A 4xx is neither retried nor blamed on the provider, and does not trigger fallback, because a malformed prompt fails the same way at every vendor.
- **Every wait is bounded, including the total.** Per-attempt bounds multiplied by attempts multiplied by chain length is not a bound. One call budget covers the entire provider phase of a request — every retry and every fallback hop — and the server's write timeout is derived from it so the two can never drift apart. Backoff sleeps select on that context rather than sleeping blind.
- **A long `Retry-After` is declined, not obeyed and not truncated.** A short wait is worth honoring: the upstream knows its own quota better than our backoff curve does. A provider saying "come back in 30 seconds" is telling us to route elsewhere, so the gateway stops retrying it and moves down the chain, returning the upstream's own `Retry-After` to the client if the chain is exhausted. Truncating the sleep instead would be the worst option — a retry aimed at an upstream that just said its quota is not free yet.
- **The breaker trades availability for not hammering a sick dependency, and that trade only pays off if there is a chain.** The chaos suite measures both sides: with a fallback chain, a provider failing half its requests still yields a 100% success rate; the same flaky provider behind an alias with no chain drops to roughly 15%, because an open breaker sheds load it has nowhere to send. Configure a chain for any alias whose availability matters.

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...      # optional; needed for the sample fallback chains
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

  # Second-vendor equivalents. Ordinary aliases a client can ask for
  # directly, which also serve as the fallback targets below.
  fast-alt:  { provider: openai, model: gpt-4o-mini }
  smart-alt: { provider: openai, model: gpt-4o }

# Token-bucket policy per alias, enforced across every gateway instance.
# An alias with no entry here is unlimited.
ratelimits:
  fast:  { capacity: 100, refill_rate: 50 }
  smart: { capacity: 50, refill_rate: 20 }
  best:  { capacity: 10, refill_rate: 2 }

# Circuit breaker, per provider: consecutive failures before the circuit
# opens, and how long it stays open before admitting one probe.
breakers:
  anthropic: { failure_threshold: 5, recovery_timeout: 30s }
  openai:    { failure_threshold: 5, recovery_timeout: 30s }

# Fallback chains, per alias. Entries are aliases, so each one carries
# its own provider and model. Followed one level deep.
fallback:
  best:  [smart, smart-alt]
  smart: [smart-alt, fast]
  fast:  [fast-alt]
```

`capacity` is the burst ceiling; `refill_rate` is the sustained rate in tokens per second. One request costs one token, charged against the alias the client named — a request that falls back does not also spend a token from the alias it fell back to.

A breaker entry naming a provider this build has no key for is ignored with a startup warning rather than refusing to boot: whether `openai` exists is a property of the environment, not of the file, and the same config is meant to deploy to both. Everything checkable from the file alone — an unknown key, a fallback entry that is not an alias, a chain that includes itself — is still a hard parse error.

### Environment

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic `x-api-key` value. At least one provider key is required. |
| `OPENAI_API_KEY` | — | OpenAI bearer token. At least one provider key is required. |
| `ADDR` | `:8080` | HTTP listen address. |
| `CONFIG_PATH` | `gateway.yaml` | Alias config. A missing file at the default path is tolerated; a missing file at an explicitly configured path is a startup error. |
| `BUCKETD_ADDRS` | — | Comma-separated bucketd nodes. Unset disables rate limiting. |
| `DATABASE_URL` | — | Postgres DSN for cost tracking. Unset logs cost batches instead of persisting them. |
| `SHUTDOWN_TIMEOUT` | `15s` | How long to drain in-flight requests on SIGTERM. |
| `ANTHROPIC_BASE_URL` | production | Override the upstream endpoint. Used by tests. |
| `OPENAI_BASE_URL` | production | Override the upstream endpoint. Also the hook for an OpenAI-compatible server such as vLLM or LiteLLM. |

Migrations in `migrations/` are embedded in the binary and applied at startup when `DATABASE_URL` is set — no sidecar container and no separate deploy step. Concurrent replicas are safe: each migration is claimed through a ledger row before it runs.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/messages` | Proxy a completion. Body matches the provider-agnostic `Request` shape. |
| `GET` | `/healthz` | Liveness. Returns 200 while the process is serving. |

Every 200 response carries the routing headers:

| Header | Meaning |
|---|---|
| `X-Gateway-Provider` | The provider that actually served the request. |
| `X-Gateway-Model` | The concrete model sent upstream. |
| `X-Gateway-Alias` | The alias the client asked for. Absent when a concrete model was named. |
| `X-Gateway-Fallback` | The alias that stepped in. Present only when the request failed over. |

A request that exhausts its chain returns the primary's own failure — the client asked for that alias, so why *it* could not be served is the answer to their question, and the other hops are in the log. When every provider in the chain has an open circuit, the response is `503` with a `circuit_open` error type and a `Retry-After` computed from when the breaker will next admit a probe.

`/metrics` and the admin API arrive with the observability work below.

## Development

```bash
go test ./...              # unit tests; Postgres tests skip without a database
go test -race -cover ./...
go test -race -v -run TestChaos ./internal/gateway/    # randomized upstream failures
```

The chaos tests stand up two `httptest` upstreams speaking the real Anthropic and OpenAI wire formats, fail a fixed fraction of requests, and drive traffic through the production provider clients and resilience layer concurrently. The failure injection is seeded deterministically — a chaos test that passes on the weather gets marked flaky and then ignored — while arrival order stays concurrent, so breaker transitions are still exercised under `-race`. The suite asserts a floor on the success rate with a fallback chain, a ceiling on it without one, and a cap on how much traffic reaches a provider that is fully down.

The cost-store tests need a real Postgres and skip when `TEST_DATABASE_URL` is unset, so the suite stays runnable without Docker. CI provides one via a service container, which is where those tests actually execute:

```bash
TEST_DATABASE_URL='postgres://user:pass@localhost:5432/gateway_test?sslmode=disable' go test ./internal/store/
```

## Roadmap

Delivered:

1. **HTTP skeleton and Anthropic provider** — `POST /v1/messages`, graceful shutdown, typed upstream errors with `Retry-After` pass-through.
2. **Rate limiting and cost tracking** — alias routing, per-alias limits via bucketd, batched cost writes to Postgres.
3. **Resilience** — per-provider circuit breaker (`sync.RWMutex` fast path on the read-dominated state check), bounded retry with jittered backoff, a declarative fallback chain, an OpenAI provider client to fail over to, and a chaos suite driving randomized upstream failures.

Planned:

4. **Observability** — Prometheus metrics for request counts, latency, breaker state and cost totals; structured JSON logs; an admin API; hot config reload via `fsnotify` and an `atomic.Value` swap for a lock-free read path.
5. **Caching and remaining providers** — exact cache keyed on a SHA-256 of the canonicalized request, optional pgvector semantic cache behind it, plus Gemini and Ollama clients.
6. **Deployment** — multi-stage distroless image, Fly.io config.

## License

MIT. See [LICENSE](LICENSE).
