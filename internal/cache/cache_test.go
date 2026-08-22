package cache

import (
	"context"
	"strings"
	"testing"

	"github.com/kevinreber/llm-gateway/internal/provider"
)

func req(mods ...func(*provider.Request)) *provider.Request {
	r := &provider.Request{
		Model:    "claude-sonnet-5",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hello"}},
	}
	for _, m := range mods {
		m(r)
	}
	return r
}

func TestKey_IsStableAcrossCalls(t *testing.T) {
	if a, b := Key("anthropic", "m", req()), Key("anthropic", "m", req()); a != b {
		t.Errorf("same request produced different keys:\n%s\n%s", a, b)
	}
}

func TestKey_IgnoresTheClientsRequestModelField(t *testing.T) {
	// The key is built from the *resolved* destination. Two aliases
	// resolving to the same model must share entries, which is only
	// true if the name the client typed is not in the hash.
	viaAlias := Key("anthropic", "claude-sonnet-5", req(func(r *provider.Request) { r.Model = "smart" }))
	viaDirect := Key("anthropic", "claude-sonnet-5", req(func(r *provider.Request) { r.Model = "claude-sonnet-5" }))

	if viaAlias != viaDirect {
		t.Error("the client's model field leaked into the key; two aliases for one model would not share entries")
	}
}

func TestKey_ChangesWithEveryInputThatChangesTheOutput(t *testing.T) {
	base := Key("anthropic", "claude-sonnet-5", req())

	cases := []struct {
		name     string
		provider string
		model    string
		req      *provider.Request
	}{
		{"provider", "openai", "claude-sonnet-5", req()},
		{"model", "anthropic", "claude-opus-5", req()},
		{"message text", "anthropic", "claude-sonnet-5",
			req(func(r *provider.Request) { r.Messages[0].Content = "hello!" })},
		{"role", "anthropic", "claude-sonnet-5",
			req(func(r *provider.Request) { r.Messages[0].Role = provider.RoleAssistant })},
		{"extra message", "anthropic", "claude-sonnet-5",
			req(func(r *provider.Request) {
				r.Messages = append(r.Messages, provider.Message{Role: provider.RoleUser, Content: "more"})
			})},
		{"system prompt", "anthropic", "claude-sonnet-5",
			req(func(r *provider.Request) { r.System = "be terse" })},
		{"max tokens", "anthropic", "claude-sonnet-5",
			req(func(r *provider.Request) { r.MaxTokens = 100 })},
		{"temperature", "anthropic", "claude-sonnet-5",
			req(func(r *provider.Request) { r.Temperature = 0.7 })},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Key(tc.provider, tc.model, tc.req); got == base {
				t.Errorf("changing %s did not change the key", tc.name)
			}
		})
	}
}

func TestKey_DoesNotNormalizeMessageWhitespace(t *testing.T) {
	// Collapsing whitespace inside a prompt would make two genuinely
	// different inputs share an entry. Leading whitespace is
	// load-bearing often enough that discarding it would be the cache
	// changing the answer.
	padded := Key("anthropic", "m", req(func(r *provider.Request) { r.Messages[0].Content = " hello" }))
	plain := Key("anthropic", "m", req())

	if padded == plain {
		t.Error("leading whitespace was normalized away")
	}
}

func TestKey_IsNamespacedAndHexEncoded(t *testing.T) {
	k := Key("anthropic", "m", req())

	if !strings.HasPrefix(k, "llmgw:exact:") {
		t.Errorf("key = %q, want the llmgw:exact: namespace", k)
	}
	hexPart := strings.TrimPrefix(k, "llmgw:exact:")
	if len(hexPart) != 64 {
		t.Errorf("hash = %d chars, want 64 (sha256 hex)", len(hexPart))
	}
	if strings.Trim(hexPart, "0123456789abcdef") != "" {
		t.Errorf("hash = %q, want hex only", hexPart)
	}
}

func TestDisabled_AlwaysMissesAndNeverErrors(t *testing.T) {
	var c Cache = Disabled{}
	ctx := context.Background()

	if err := c.Set(ctx, "k", &provider.Response{Content: "x"}, 0); err != nil {
		t.Errorf("Set: %v", err)
	}
	resp, hit, err := c.Get(ctx, "k")
	if err != nil || hit || resp != nil {
		t.Errorf("Get = (%v, %v, %v), want a clean miss", resp, hit, err)
	}
	if err := c.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}
