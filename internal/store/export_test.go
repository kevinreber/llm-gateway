package store

import "context"

// Test-only helpers. This file compiles only under `go test`, so these
// destructive and introspective queries can never be reached from a
// production build.

// DropAllForTest returns the database to an empty schema so each test
// starts from a known state.
func (p *Postgres) DropAllForTest(ctx context.Context) error {
	_, err := p.pool.Exec(ctx, `
		DROP TABLE IF EXISTS costs;
		DROP TABLE IF EXISTS schema_migrations;`)
	return err
}

// CountCostsForTest returns the number of rows in the costs table.
func (p *Postgres) CountCostsForTest(ctx context.Context) (int, error) {
	var n int
	err := p.pool.QueryRow(ctx, `SELECT COUNT(*) FROM costs`).Scan(&n)
	return n, err
}

// CountNullAliasForTest returns how many cost rows have no alias.
func (p *Postgres) CountNullAliasForTest(ctx context.Context) (int, error) {
	var n int
	err := p.pool.QueryRow(ctx, `SELECT COUNT(*) FROM costs WHERE alias IS NULL`).Scan(&n)
	return n, err
}

// SumCostCentsForTest totals the cost column, as a billing report would.
func (p *Postgres) SumCostCentsForTest(ctx context.Context) (float64, error) {
	var total float64
	err := p.pool.QueryRow(ctx,
		`SELECT COALESCE(SUM(cost_cents), 0)::float8 FROM costs`).Scan(&total)
	return total, err
}
