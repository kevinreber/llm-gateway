# llm-gateway

A production-grade LLM API gateway in Go. Reverse-proxies completion requests to Anthropic and OpenAI (with Gemini and Ollama planned), resolving client-facing aliases to concrete models, enforcing distributed rate limits via [bucketd](https://github.com/kevinreber/bucketd), recording per-request token cost to Postgres, and failing over between providers when one degrades.

Built on the `net/http` standard library — no web framework.

## Status

Working today: HTTP proxy to Anthropic and OpenAI, alias-based routing, per-alias distributed rate limiting, cost tracking, resilience — per-provider circuit breaker, bounded retry with jittered backoff, and a declarative fallback chain — plus Prometheus metrics and structured request-scoped logs. Caching, the admin API, and hot config reload are in progress — see [Roadmap](#roadmap).

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

- **A price is only ever inherited across a date.** The pricing table maps model IDs to list rates, and a dated snapshot (`gpt-4o-mini-2024-07-18`, `claude-haiku-4-5-20251001`) inherits its family's rate so every release does not need a row. Nothing else inherits. Vendors ship capability variants under hyphen-delimited names too, and those are priced differently — `gpt-5-pro` is 12x `gpt-5`, `o3-pro` is 10x `o3` — so a rule that accepted any hyphenated suffix billed all of them at the base model's rate *and* reported them as priced, suppressing the unpriced-model warning that exists to catch it. Requiring the suffix to parse as a date inverts the default: an unrecognized variant falls out of the table, warns, and bills zero, so the failure mode of a new model release is a log line rather than an invisible billing error.
- **Cost tracking never blocks a request.** Events go to a bounded buffered channel; a single writer goroutine batches them into Postgres via `COPY` every second or every 100 events, whichever comes first. When the buffer fills, events are counted and dropped rather than applying backpressure — a slow database degrades accounting, not availability. The drop count is logged at shutdown so the loss is visible.
- **Shutdown drains in order.** SIGTERM drains in-flight HTTP requests first (bounded by `SHUTDOWN_TIMEOUT`), then stops the cost writer so requests that finished during the drain still get their rows, then closes the database pool.
- **The rate limiter fails open.** If bucketd is unreachable, the request is allowed and the error logged. A rate-limiter outage should degrade enforcement, not take the gateway down with it. This is the right trade for a limiter that controls spend; a limiter guarding correctness would want the opposite.
- **Config parsing is strict.** An unknown key in `gateway.yaml` is a startup error. A typo should never silently disable a rate limit.
- **Every dependency is optional.** With no `gateway.yaml`, clients name concrete models directly. With no `BUCKETD_ADDRS`, nothing is rate limited. With no `DATABASE_URL`, cost batches are logged instead of persisted. One provider key is enough to boot. The gateway runs usefully with none of the rest configured.
- **Fallback chains are lists of aliases, not lists of providers.** "Fail over from Anthropic to OpenAI" is not actionable until someone decides which OpenAI model stands in for Sonnet — and an alias is exactly that decision, already written down. Keying the chain on aliases means every entry names a model its provider can serve, and every name in the block is checked against the alias table at parse time instead of turning out to be dead at 3am. Chains are followed one level deep, which makes cycles structurally impossible rather than something to detect.
- **Retry and the breaker read typed errors, never message text.** `provider.APIError` carries `Status`, `Type`, and `RetryAfter`, so a reworded upstream error message cannot silently change routing behavior. A 5xx is retried and counts against the provider's health. A 429 is retried but does *not* trip the breaker — a rate-limited provider is a healthy provider telling us to slow down, and taking it out of rotation would punish every caller for one noisy alias. Most 4xx responses are neither retried nor blamed on the provider, and do not trigger fallback, because a malformed prompt fails the same way at every vendor. The exceptions are the ones that are about the provider rather than the request: a 401 or 403 is not retried — a credential does not fix itself between attempts — but it does trip the breaker and does trigger fallback, because a provider we cannot authenticate to can serve nothing, and key rotation and billing lapses are exactly the incident failover exists to absorb. A 408 is both retried and counted against health, as a provider that timed out is a provider that was too slow.
- **Every wait is bounded, including the total.** Per-attempt bounds multiplied by attempts multiplied by chain length is not a bound. One call budget covers the entire provider phase of a request — every retry and every fallback hop — and the server's write timeout is derived from it so the two can never drift apart. Backoff sleeps select on that context rather than sleeping blind.
- **A long `Retry-After` is declined, not obeyed and not truncated.** A short wait is worth honoring: the upstream knows its own quota better than our backoff curve does. A provider saying "come back in 30 seconds" is telling us to route elsewhere, so the gateway stops retrying it and moves down the chain, returning the upstream's own `Retry-After` to the client if the chain is exhausted. Truncating the sleep instead would be the worst option — a retry aimed at an upstream that just said its quota is not free yet.
- **The breaker trades availability for not hammering a sick dependency, and that trade only pays off if there is a chain.** The chaos suite measures both sides against a provider failing half its requests: the suite asserts at least 95% success with a fallback chain and below 95% without one, and a representative run answers 200 of 200 with a chain against 4 of 200 without, because an open breaker sheds load it has nowhere to send. Configure a chain for any alias whose availability matters.

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

> **Rolling back:** strict parsing couples the config file to the binary in one direction. A build that predates a config block treats it as an unknown key, which is a startup error and not a warning, so the process exits without binding a listener rather than degrading. Roll `gateway.yaml` back in the same step as the binary — a `breakers:` or `fallback:` block left in place will stop an older gateway from starting at all.

### Environment

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic `x-api-key` value. At least one provider key is required. |
| `OPENAI_API_KEY` | — | OpenAI bearer token. At least one provider key is required. |
| `ADDR` | `:8080` | Request-path listen address. |
| `ADMIN_ADDR` | `:9090` | Admin API and Prometheus exposition. Set to `off` to disable, which also removes `/metrics` — there is no second place it is served. |
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
| `GET` | `/healthz` | Readiness. 200 while serving, 503 once shutdown begins so a load balancer drains the instance before the listener closes. |

The operational surface is on a **separate listener** (`ADMIN_ADDR`, default `:9090`):

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin/aliases` | The live alias table, plus the file it came from and whether it can be reloaded. |
| `GET` | `/admin/stats` | Per-provider breaker state, health, and request counts by result, plus dropped cost events. |
| `POST` | `/admin/reload` | Re-read the config file and swap it in. |
| `GET` | `/metrics` | Prometheus exposition. Breaker gauges are read at scrape time, so they cannot drift from what the request path sees. |

That split is a security boundary, not organization. The exposition discloses cumulative spend, which vendors are wired, and which of them are currently failing. `POST /admin/reload` repoints live traffic and carries **no authentication at all** — there is no credential to leak because there is no credential. Bind `ADMIN_ADDR` somewhere only operators can reach, or set it to `off`.

`POST /admin/reload` parses and validates before swapping, so a file with a typo in it returns `400` and leaves the running config untouched. The failure mode of a bad edit is "the change did not take effect", never "routing is now broken", because the second one is discovered by traffic. The response also names any alias the new config declares that this binary cannot serve, which tells the operator who just made the edit rather than leaving it for a request at 3am.

A request reads the configuration once, at the top, and uses that snapshot throughout. A reload landing mid-request cannot make one request resolve its alias against one config and take its rate limit from another.


Every 200 response carries the routing headers:

| Header | Meaning |
|---|---|
| `X-Gateway-Provider` | The provider that actually served the request. |
| `X-Gateway-Model` | The concrete model sent upstream. |
| `X-Gateway-Alias` | The alias the client asked for. Absent when a concrete model was named. |
| `X-Gateway-Fallback` | The alias that stepped in. Present only when the request failed over. |

A request that exhausts its chain returns the primary's own failure — the client asked for that alias, so why *it* could not be served is the answer to their question, and the other hops are in the log. So when the primary was refused by its own open circuit and no fallback served, the response is `503` with a `circuit_open` error type and a `Retry-After` computed from when that breaker will next admit a probe — even if a later hop failed some other way, since the primary's error is the one reported.

Every response also carries `X-Request-ID`, which is the identifier the request's log lines are tagged with. An inbound `X-Request-ID` is reused so one identifier spans the whole hop chain, provided it is printable ASCII and under 128 bytes; anything else is replaced with a generated ID rather than escaped, because the value reaches a response header where a bare CR or LF is response splitting and not a cosmetic problem.

### Metrics

| Metric | Type | Labels |
|---|---|---|
| `llm_gateway_requests_total` | counter | `alias`, `provider`, `result` |
| `llm_gateway_request_duration_seconds` | histogram | `alias`, `provider` |
| `llm_gateway_cost_cents_total` | counter | `provider`, `model` |
| `llm_gateway_breaker_state` | gauge | `provider` |
| `llm_gateway_provider_health` | gauge | `provider` |
| `llm_gateway_cost_events_dropped_total` | counter | — |

`result` is one of `ok`, `bad_request`, `rate_limited`, `circuit_open`, `upstream_rejected`, `provider_error`. The last two are deliberately distinct: `upstream_rejected` is the provider declining one request on its own terms (a malformed prompt, a 429), while `provider_error` is the provider itself failing. Folding them together would make the obvious alert — rate of `provider_error` — page somebody because one caller is sending bad prompts. It is the same line `internal/resilience` already draws when it decides a 400 must not count toward opening a circuit. `alias` and `provider` read `none` when a request carried neither, which happens for a direct model name and for a request rejected before routing.

Every label value is drawn from a finite set the gateway controls. `model` in particular is the pricing table's own ID rather than the string the upstream echoed back, so a dated snapshot such as `gpt-4o-2024-08-06` is recorded under `gpt-4o`. Labelling with the echoed string would hand cardinality control to the provider — one permanent series per distinct value it returns — and it would also split a family's spend across a series per release date, which is the wrong shape for the question a cost dashboard is asked. A model absent from the table is recorded under `unpriced`; its real name survives on the cost row and in the warning that path logs.

The latency histogram covers only requests that reached the provider phase. Requests refused at the door are counted by `llm_gateway_requests_total` instead — folding their microsecond-scale timings into a histogram measuring provider latency would produce percentiles describing neither population. A request refused by an open breaker *is* in the histogram, in the bottom bucket, because turning a slow failure into an immediate one is what the breaker is for and it should be visible.

`llm_gateway_breaker_state` is `0` closed, `1` half-open, `2` open. `llm_gateway_provider_health` is the same fact for alerting: `1` when the gateway would currently admit a call, `0` when the circuit is open. It is derived from breaker state, not from an upstream probe, so scraping is free. Half-open counts as healthy — the breaker is admitting a probe, and paging on a provider that is recovering on its own is how an alert teaches people to ignore it.

`llm_gateway_cost_events_dropped_total` is the one number worth alerting on unconditionally. The cost writer drops rather than blocks when its buffer fills, which is the right trade — a slow Postgres must not become the gateway's latency — but any non-zero rate means recorded spend is an undercount, and that is something to learn from a dashboard rather than from a finance question three weeks later.

`/admin/stats` reads its per-provider request counts back out of the metric registry rather than keeping a second tally, so it and `/metrics` cannot disagree. Two independent counts of the same events drift the moment one is updated on a path the other missed, and the version an operator sees during an incident is whichever one they happened to open.

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

4. **Observability** *(partly delivered)* — Prometheus metrics for request counts, latency, breaker state and cost totals, structured JSON logs keyed by request ID, a shutdown-aware `/healthz`, and an admin API on its own listener with config reload through a lock-free atomic swap. Still to come: picking up file edits automatically via `fsnotify`, rather than requiring a `POST /admin/reload`.
5. **Caching and remaining providers** — exact cache keyed on a SHA-256 of the canonicalized request, optional pgvector semantic cache behind it, plus Gemini and Ollama clients.
6. **Deployment** — multi-stage distroless image, Fly.io config.

## License

MIT. See [LICENSE](LICENSE).
