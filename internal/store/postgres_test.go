package store_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/kevinreber/llm-gateway/internal/cost"
	"github.com/kevinreber/llm-gateway/internal/store"
)

// openTestDB connects to the database named by TEST_DATABASE_URL and
// starts each test from a clean schema.
//
// The test skips rather than fails when the variable is unset, so
// `go test ./...` stays runnable on a laptop with no Postgres. CI sets
// the variable against a service container, where these must run.
func openTestDB(t *testing.T) *store.Postgres {
	t.Helper()

	dsn := os.Getenv("TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("TEST_DATABASE_URL not set; skipping Postgres integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	pg, err := store.Open(ctx, dsn)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(pg.Close)

	if err := pg.DropAllForTest(ctx); err != nil {
		t.Fatalf("reset schema: %v", err)
	}
	if err := pg.Migrate(ctx); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return pg
}

func TestMigrate_IsIdempotent(t *testing.T) {
	pg := openTestDB(t)
	ctx := context.Background()

	// openTestDB already migrated once; a second and third pass must be
	// no-ops rather than re-running CREATE TABLE. This is the path every
	// restart and every replica boot takes.
	for i := 0; i < 2; i++ {
		if err := pg.Migrate(ctx); err != nil {
			t.Fatalf("re-migrate %d: %v", i, err)
		}
	}

	n, err := pg.CountCostsForTest(ctx)
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 0 {
		t.Errorf("costs table has %d rows on a fresh schema, want 0", n)
	}
}

func TestInsertCosts_RoundTrip(t *testing.T) {
	pg := openTestDB(t)
	ctx := context.Background()

	now := time.Now().UTC().Truncate(time.Microsecond)
	batch := []cost.Event{
		{
			TS: now, Provider: "anthropic", Model: "claude-sonnet-5",
			Alias: "smart", InputTokens: 1000, OutputTokens: 500, CostCents: 1.05,
		},
		{
			// No alias: the caller named a concrete model. Must land as
			// NULL, not empty string.
			TS: now, Provider: "anthropic", Model: "claude-haiku-4-5",
			InputTokens: 10, OutputTokens: 5, CostCents: 0.0035,
		},
	}

	if err := pg.InsertCosts(ctx, batch); err != nil {
		t.Fatalf("InsertCosts: %v", err)
	}

	n, err := pg.CountCostsForTest(ctx)
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 2 {
		t.Fatalf("wrote %d rows, want 2", n)
	}

	nullAliases, err := pg.CountNullAliasForTest(ctx)
	if err != nil {
		t.Fatalf("count null aliases: %v", err)
	}
	if nullAliases != 1 {
		t.Errorf("rows with NULL alias = %d, want 1", nullAliases)
	}

	// cost_cents is NUMERIC(12,6): a sub-hundredth-of-a-cent Haiku
	// request must survive the round trip instead of rounding to zero.
	total, err := pg.SumCostCentsForTest(ctx)
	if err != nil {
		t.Fatalf("sum: %v", err)
	}
	if want := 1.05350; total < want-1e-6 || total > want+1e-6 {
		t.Errorf("SUM(cost_cents) = %v, want %v", total, want)
	}
}

func TestInsertCosts_EmptyBatchIsNoop(t *testing.T) {
	pg := openTestDB(t)
	ctx := context.Background()

	if err := pg.InsertCosts(ctx, nil); err != nil {
		t.Fatalf("InsertCosts(nil): %v", err)
	}
	if err := pg.InsertCosts(ctx, []cost.Event{}); err != nil {
		t.Fatalf("InsertCosts(empty): %v", err)
	}
}

func TestInsertCosts_LargeBatch(t *testing.T) {
	pg := openTestDB(t)
	ctx := context.Background()

	// The writer flushes at 100 events; exercise a batch past that so
	// the CopyFrom path is covered at its real working size.
	const n = 250
	batch := make([]cost.Event, n)
	for i := range batch {
		batch[i] = cost.Event{
			TS: time.Now().UTC(), Provider: "anthropic",
			Model: "claude-sonnet-5", Alias: "smart",
			InputTokens: i, OutputTokens: i, CostCents: 0.001,
		}
	}
	if err := pg.InsertCosts(ctx, batch); err != nil {
		t.Fatalf("InsertCosts: %v", err)
	}

	got, err := pg.CountCostsForTest(ctx)
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if got != n {
		t.Errorf("wrote %d rows, want %d", got, n)
	}
}
