# syntax=docker/dockerfile:1.6
#
# Multi-stage build for llm-gateway, same shape as bucketd's:
#   1. `builder` — has the Go toolchain; compiles a static binary
#   2. `final`   — distroless base, nothing but the binary
#
# Distroless because there is no shell, no package manager, and no
# userland to exploit. A gateway holds every provider API key the
# deployment owns, so the blast radius of code execution inside this
# container is "all of the org's LLM spend" rather than "one service".

# ---- builder ----
FROM golang:1.26-alpine AS builder

WORKDIR /src

# Copy the module files alone first so the dependency-download layer is
# reused across the iterations that do not change them.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Static binary: no cgo, so nothing links against a glibc the distroless
# image does not have. -trimpath strips filesystem paths for
# reproducibility; -s -w drops the symbol and DWARF tables.
ARG VERSION=dev
RUN CGO_ENABLED=0 GOOS=linux go build \
    -trimpath \
    -ldflags="-s -w -X main.version=${VERSION}" \
    -o /out/gateway \
    ./cmd/gateway

# ---- final ----
FROM gcr.io/distroless/static:nonroot

# ADDR serves clients. ADMIN_ADDR serves /metrics and /admin/*, and is
# deliberately not published in fly.toml — see the comment there. Both
# are overridable at runtime.
ENV ADDR=:8080
ENV ADMIN_ADDR=:8081
EXPOSE 8080
EXPOSE 8081

# UID 65532, provided by the distroless image.
USER nonroot:nonroot

COPY --from=builder /out/gateway /usr/local/bin/gateway

ENTRYPOINT ["/usr/local/bin/gateway"]
