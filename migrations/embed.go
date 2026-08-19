// Package migrations embeds the SQL schema so a built binary carries its
// own migrations and can bring an empty database up to date at startup —
// no sidecar container, no volume mount, no separate migration step in
// the deploy.
//
// The .sql files stay at the repo root (rather than under internal/store)
// so they are the obvious first thing a reader finds when asking "what
// does the schema look like".
package migrations

import "embed"

// FS holds every migration, named <version>_<description>.up.sql.
// Versions are applied in lexical order, so keep the numeric prefix
// zero-padded.
//
//go:embed *.sql
var FS embed.FS
