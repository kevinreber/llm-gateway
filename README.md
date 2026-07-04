# llm-gateway

A production-grade LLM API gateway in Go. Reverse-proxies calls to Anthropic, OpenAI, Gemini, and Ollama with rate limiting (via [bucketd](https://github.com/kevinreber/bucketd)), cost tracking, circuit-breaker fallback, hot-reloadable config, and Prometheus metrics.

## Status

Sprint 3 of Kevin Reber's [Anthropic infra interview prep plan](https://github.com/kevinreber/aura-utils). Active as of Jul 4, 2026. Full sprint plan: [`sprints/03-llm-gateway-go.md`](https://github.com/kevinreber/brain-vault) (private).

## Features (target v0.1.0)

- **Alias-based routing** — clients ask for `model: smart`; a YAML config resolves it to `{provider, model}`. Change the YAML, traffic shifts on next request (no restart).
- **Distributed rate limiting** — imports `github.com/kevinreber/bucketd/client`. Per-alias and per-provider limits enforced across gateway instances.
- **Circuit breaker per provider** — `sync.RWMutex` fast-path for state checks; opens on repeated failures, fails fast while open, probes recovery via half-open.
- **Fallback chain** — declarative `gateway.yaml` list per primary provider. Anthropic → OpenAI → Gemini → Ollama.
- **Cost tracking** — every request emits a `cost.Event` to a buffered channel; a single writer goroutine batches INSERTs to Postgres every 1s.
- **Exact + semantic cache** — SHA-256 of canonicalized request first, then pgvector cosine similarity above a threshold.
- **Hot config reload** — `fsnotify` watcher + `atomic.Value` swap. Lock-free reads on the hot path.
- **Observability** — Prometheus metrics for request counts, latency histograms, breaker states, cache hit rates, cost totals; structured JSON logs via `log/slog`.

## Not yet

Empty scaffold as of Jul 4, 2026. See the sprint doc for the 6-phase build plan (Jul 4 – Aug 15, 2026).

## License

MIT. See [LICENSE](LICENSE).
