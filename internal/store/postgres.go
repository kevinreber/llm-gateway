// Package store is the Postgres persistence layer. Today that means the
// costs table; Phase 5 adds the pgvector-backed semantic cache alongside
// it on the same pool.
package store

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"sort"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/kevinreber/llm-gateway/internal/cost"
	"github.com/kevinreber/llm-gateway/migrations"
)

// Postgres owns the connection pool. Safe for concurrent use.
type Postgres struct {
	pool *pgxpool.Pool
}

// Open parses the DSN, opens a pool, and verifies connectivity before
// returning. Failing here rather than on first query means a bad
// DATABASE_URL surfaces at startup instead of on a user's request.
//
// ctx bounds the connectivity check only. It is deliberately NOT the
// pool's own context: pgxpool hands the context it is constructed with
// to a background goroutine that pre-warms idle connections, so passing
// a startup deadline here would cancel that pre-warm as soon as the
// deadline was cleaned up. That is invisible today because the pool
// defaults to zero minimum connections, and would quietly break the
// moment someone set pool_min_conns in the DSN.
func Open(ctx context.Context, dsn string) (*Postgres, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("parse DATABASE_URL: %w", err)
	}
	pool, err := pgxpool.NewWithConfig(context.WithoutCancel(ctx), cfg)
	if err != nil {
		return nil, fmt.Errorf("open pool: %w", err)
	}
	// Ping honours ctx, so this is what actually enforces the startup
	// deadline — pgxpool connects lazily and NewWithConfig does not
	// touch the network.
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping: %w", err)
	}
	return &Postgres{pool: pool}, nil
}

// Close releases the pool.
func (p *Postgres) Close() { p.pool.Close() }

// Migrate applies any embedded migrations this database has not seen,
// in version order, each in its own transaction.
//
// Concurrency: several gateway replicas can boot at once, so the ledger
// row is inserted with ON CONFLICT DO NOTHING and the migration only
// runs if that insert claimed the version. A replica that loses the race
// skips the file rather than replaying a CREATE TABLE.
//
// The claim must stay inside the same transaction as the migration body.
// That is what makes all three interleavings correct: if the winner
// commits, the loser's insert conflicts and affects zero rows; if the
// winner aborts, the loser's insert succeeds and it runs the migration;
// and if they race, Postgres blocks the loser's ON CONFLICT insert on
// the winner's uncommitted row until it resolves, so the loser always
// observes a settled outcome instead of racing on CREATE TABLE. Moving
// the claim out of the transaction breaks all of that.
func (p *Postgres) Migrate(ctx context.Context) error {
	if _, err := p.pool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS schema_migrations (
			version     TEXT PRIMARY KEY,
			applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`); err != nil {
		return fmt.Errorf("create schema_migrations: %w", err)
	}

	names, err := migrationNames()
	if err != nil {
		return err
	}

	for _, name := range names {
		version := strings.TrimSuffix(name, ".up.sql")
		body, err := migrations.FS.ReadFile(name)
		if err != nil {
			return fmt.Errorf("read migration %s: %w", name, err)
		}
		if err := p.applyOnce(ctx, version, string(body)); err != nil {
			return fmt.Errorf("apply migration %s: %w", name, err)
		}
	}
	return nil
}

func (p *Postgres) applyOnce(ctx context.Context, version, body string) error {
	tx, err := p.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	tag, err := tx.Exec(ctx,
		`INSERT INTO schema_migrations (version) VALUES ($1) ON CONFLICT DO NOTHING`,
		version)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		// Already applied, or claimed by a replica racing us.
		return nil
	}
	if _, err := tx.Exec(ctx, body); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func migrationNames() ([]string, error) {
	entries, err := fs.Glob(migrations.FS, "*.up.sql")
	if err != nil {
		return nil, fmt.Errorf("list migrations: %w", err)
	}
	if len(entries) == 0 {
		return nil, errors.New("no migrations embedded")
	}
	sort.Strings(entries)
	return entries, nil
}

// costColumns must stay in the same order as the values built in
// InsertCosts — CopyFrom matches positionally, not by name.
var costColumns = []string{
	"ts", "provider", "model", "alias", "input_tok", "output_tok", "cost_cents",
}

// InsertCosts implements cost.Sink using COPY, which is substantially
// cheaper than a multi-row INSERT once batches reach a few dozen rows.
func (p *Postgres) InsertCosts(ctx context.Context, batch []cost.Event) error {
	if len(batch) == 0 {
		return nil
	}

	rows := make([][]any, len(batch))
	for i, e := range batch {
		// An empty alias means the caller named a concrete model. Store
		// NULL rather than '' so "requests that used an alias" is a
		// plain IS NOT NULL and the partial index applies.
		var alias any
		if e.Alias != "" {
			alias = e.Alias
		}
		rows[i] = []any{
			e.TS, e.Provider, e.Model, alias,
			e.InputTokens, e.OutputTokens, e.CostCents,
		}
	}

	n, err := p.pool.CopyFrom(ctx,
		pgx.Identifier{"costs"}, costColumns, pgx.CopyFromRows(rows))
	if err != nil {
		return fmt.Errorf("copy costs: %w", err)
	}
	if int(n) != len(batch) {
		return fmt.Errorf("copy costs: wrote %d of %d rows", n, len(batch))
	}
	return nil
}
