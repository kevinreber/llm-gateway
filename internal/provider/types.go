// Package provider defines the abstraction over upstream LLM APIs
// (Anthropic, OpenAI, Gemini, Ollama) and implements per-provider clients.
//
// The Provider interface is intentionally small: Do executes a single
// non-streaming completion, and Health/Supports/Name give the gateway
// enough surface to route, monitor, and fall back.
package provider

// Role names used across all provider APIs.
const (
	RoleUser      = "user"
	RoleAssistant = "assistant"
)

// Message is one turn in a chat-style completion request.
// All providers accept the same {role, content} shape at the top of their
// message array; anything provider-specific (Anthropic's system block,
// OpenAI's tool_choice, etc.) lives on the outer Request.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Request is the provider-agnostic completion input. Per-provider
// implementations translate this into their native wire format.
//
// System is separated from Messages because Anthropic requires it as a
// top-level field, not a message. OpenAI accepts a system message; we
// still accept System here and prepend/inject it in the OpenAI client.
type Request struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	System      string    `json:"system,omitempty"`
}

// Response is the provider-agnostic completion output.
//
// Content is the concatenated text response. For Anthropic the wire
// format is a list of content blocks ({type: text, text: "..."}); we
// concatenate them so callers don't need to reason about block structure.
// Once we support tool-use pass-through (out of scope for v0.1.0), this
// abstraction will grow.
type Response struct {
	Content    string `json:"content"`
	Usage      Usage  `json:"usage"`
	StopReason string `json:"stop_reason"`
	Model      string `json:"model"`
}

// Usage is the token count metadata returned by every provider.
// Both Anthropic and OpenAI use snake_case on the wire.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}
