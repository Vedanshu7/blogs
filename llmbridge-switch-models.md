# Stop Rewriting LLM Code — llmbridge Gives Go One Interface for All of It

[![Go Reference](https://pkg.go.dev/badge/github.com/Vedanshu7/llmbridge.svg)](https://pkg.go.dev/github.com/Vedanshu7/llmbridge)
[![CI](https://github.com/Vedanshu7/llmbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/Vedanshu7/llmbridge/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/Vedanshu7/llmbridge)](https://goreportcard.com/report/github.com/Vedanshu7/llmbridge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Vedanshu7/llmbridge/blob/main/LICENSE)

**GitHub:** [github.com/Vedanshu7/llmbridge](https://github.com/Vedanshu7/llmbridge) · **Docs:** [pkg.go.dev/github.com/Vedanshu7/llmbridge](https://pkg.go.dev/github.com/Vedanshu7/llmbridge)

![Abstract blue background with connected technology-related icons](https://plus.unsplash.com/premium_photo-1764691292167-7263c8b464c0?fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.0)
*Photo by [Getty Images](https://unsplash.com/@gettyimages) on [Unsplash](https://unsplash.com/photos/abstract-blue-background-with-connected-technology-related-icons-1IJnGLTv7SU)*

A new model drops. GPT-4o. Claude 4. Gemini 2.0. Whatever it is — faster, smarter, cheaper. You want it.

Then you open your codebase.

```go
// scattered across a dozen files
req := openai.ChatCompletionRequest{
    Model: "gpt-3.5-turbo",
    Messages: []openai.ChatCompletionMessage{
        {Role: "user", Content: prompt},
    },
}
resp, err := client.CreateChatCompletion(ctx, req)
if err != nil {
    var apiErr *openai.APIError
    if errors.As(err, &apiErr) && apiErr.HTTPStatusCode == 429 {
        // rate limit handling...
    }
}
output := resp.Choices[0].Message.Content
```

Switching providers is never just changing the model string. It means different SDKs, different request shapes, different response shapes, different streaming APIs, different error types. If this is spread across your codebase, "upgrading to the latest model" is a multi-day refactor — not an afternoon.

But that's only the beginning of the problem. Once you're past the migration, you still don't have retries, failover, caching, spend limits, guardrails, or observability. Every team ends up bolting those on themselves, in different ways, on top of different SDKs.

**[llmbridge](https://github.com/Vedanshu7/llmbridge)** is a Go library that solves all of this at once: a single unified interface across every major LLM provider, with production infrastructure built in.


## One Interface. Every Provider. Zero Rewrites.

Write your code once against the llmbridge interface:

```go
import (
    "github.com/Vedanshu7/llmbridge"
    "github.com/Vedanshu7/llmbridge/llms/openai"
)

provider := openai.New("gpt-3.5-turbo", os.Getenv("OPENAI_API_KEY"))

resp, err := provider.Complete(ctx, llmbridge.Request{
    System:   "You are a helpful assistant.",
    Messages: []llmbridge.Message{
        {Role: "user", Content: userInput},
    },
})
fmt.Println(resp.Content)
```

Now a new model drops. You change **one line** — nothing else:

```go
// OpenAI → Anthropic Claude 4
provider := anthropic.New("claude-opus-4-7", os.Getenv("ANTHROPIC_API_KEY"))

// → Google Gemini
provider := gemini.New("gemini-2.0-flash", os.Getenv("GEMINI_API_KEY"))

// → Groq (fast inference)
provider := compatible.NewGroq("llama-3.3-70b-versatile", os.Getenv("GROQ_API_KEY"))

// → Local Ollama (no API key, no cost)
provider := compatible.NewOllama("llama3.2")
```

The `Complete` call, the error handling, the streaming code — all untouched. Every supported provider speaks the same interface.

**Supported providers:** OpenAI · Anthropic · Google Gemini · AWS Bedrock · Azure OpenAI · Cohere · Groq · Ollama · LM Studio · Together AI · xAI/Grok · any OpenAI-compatible endpoint.


## Streaming — Same API, Every Provider

Streaming is usually the hardest part of a migration. Every provider does it differently under the hood. llmbridge normalises all of it into a single channel-based API:

```go
ch, err := provider.Stream(ctx, llmbridge.Request{
    Messages: []llmbridge.Message{
        {Role: "user", Content: "Explain quantum computing simply."},
    },
})
if err != nil {
    return err
}
for delta := range ch {
    if delta.Err != nil { return delta.Err }
    if delta.Done { break }
    fmt.Print(delta.Content)
}
```

Swap `provider` from OpenAI to Anthropic to Ollama — this loop never changes.


## Typed Errors Across Every Provider

Each SDK ships its own error struct. llmbridge maps all of them into a common typed hierarchy so you write error-handling logic once:

```go
import "github.com/Vedanshu7/llmbridge/exceptions"

resp, err := provider.Complete(ctx, req)
if err != nil {
    var rlErr  *exceptions.RateLimitError
    var cwErr  *exceptions.ContextWindowExceededError
    var authErr *exceptions.AuthenticationError

    switch {
    case errors.As(err, &rlErr):
        time.Sleep(5 * time.Second)
    case errors.As(err, &cwErr):
        // context full — route to a larger model
    case errors.As(err, &authErr):
        log.Fatalf("bad API key for %s", authErr.LLMProvider)
    }
}
```

Write this once. It works regardless of which provider returned the error.


## Multi-Provider Router — Failover, Retries, Circuit Breaker

Not ready to commit to the new model yet? Run multiple providers with automatic failover:

```go
router := llmbridge.NewRouter(
    []llmbridge.Provider{
        anthropic.New("claude-opus-4-7", os.Getenv("ANTHROPIC_API_KEY")), // primary
        openai.New("gpt-4o", os.Getenv("OPENAI_API_KEY")),                // fallback
    },
    llmbridge.WithStrategy(llmbridge.PriorityOrder),
    llmbridge.WithRetryPolicy(llmbridge.DefaultRetryPolicy),
    llmbridge.WithCircuitBreaker(5, 30*time.Second),
    llmbridge.WithContextWindowFallback(true), // auto-reroute when context is exceeded
)

resp, err := router.Complete(ctx, req)
```

If the primary is down, rate-limited, or trips the circuit breaker, requests automatically fall through to the next provider. Six routing strategies are available: `PriorityOrder`, `RoundRobin`, `LeastLatency`, `LeastBusy`, `CostBased`, and `Weighted`.


## Caching — Including Semantic Cache

Repeated or near-identical requests don't need to hit the API every time. llmbridge ships four cache backends:

```go
import "github.com/Vedanshu7/llmbridge/caching"

// Exact-match in-memory cache
cache := caching.NewInMemoryCache()

// Semantic cache — hits on meaning, not just exact strings
// "What's the weather in Paris?" will hit the cache for "Paris weather today?"
embedder := openai.New("text-embedding-3-small", key)
sc := caching.NewSemanticCache(cache, embedder, 0.95) // cosine similarity threshold
```

Also supports disk and Redis backends for distributed or persistent caching.


## Budget & Spend Tracking

Set per-key or per-team spend limits with alert callbacks before costs spiral:

```go
import "github.com/Vedanshu7/llmbridge/budget"

tracker := budget.NewTracker()
tracker.SetLimit("prod-key", 100.00)  // $100 hard limit
tracker.OnAlert(func(key string, spend float64) {
    log.Printf("[ALERT] key %s has spent $%.2f", key, spend)
})

cost, _ := llmbridge.CompletionCost(resp) // per-model pricing tables built in
if err := tracker.Record("prod-key", cost); err != nil {
    // budget.ErrBudgetExceeded — stop before the bill arrives
}
```

Cost tables for every supported model are built-in — `CompletionCost` works without any configuration.


## Guardrails — Safety Before the Request Leaves Your Server

Validate inputs and outputs before they ever reach the model:

```go
import "github.com/Vedanshu7/llmbridge/guardrails"

engine, _ := guardrails.NewEngine(
    guardrails.MaxInputLength(50_000),
    guardrails.BlockPIIPatterns(),        // catch emails, phone numbers, SSNs
    guardrails.BlockPromptInjection(),    // detect jailbreak attempts
)

if err := engine.Check(req); err != nil {
    // reject before sending — save money, stay compliant
}
```


## Middleware — Composable Cross-Cutting Concerns

The middleware chain works exactly like HTTP middleware. Add logging, auth, tracing, or rate limiting without touching business logic:

```go
func Logger(log *slog.Logger) llmbridge.Middleware {
    return func(ctx context.Context, req llmbridge.Request, next llmbridge.Handler) (*llmbridge.Response, error) {
        start := time.Now()
        resp, err := next(ctx, req)
        log.Info("llm call", "latency", time.Since(start), "tokens", resp.Usage.TotalTokens, "err", err)
        return resp, err
    }
}

provider := llmbridge.Chain(
    openai.New("gpt-4o", key),
    Logger(slog.Default()),
)
```


## OpenAI-Compatible Proxy Server

If you have multiple services or teams, run llmbridge as a shared proxy. Any client that speaks the OpenAI API — in any language — can point at it and get routing, caching, auth, and observability for free:

```go
import (
    "github.com/Vedanshu7/llmbridge/proxy"
    "github.com/Vedanshu7/llmbridge/llms/anthropic"
)

backend := anthropic.New("claude-opus-4-7", os.Getenv("ANTHROPIC_API_KEY"))
srv, err := proxy.NewServerWithDB(backend, "/data/llmbridge.db")

key, _ := srv.KeyStore().GenerateAPIKey([]string{"completion"})
fmt.Println("API key:", key)

srv.Start(ctx, ":8080")
```

Or via Docker:

```bash
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY \
  -v ./config.json:/config.json:ro \
  ghcr.io/vedanshu7/llmbridge server -config /config.json
```

The proxy ships with API key management, SSO/OIDC (Google, GitHub, Microsoft), org and team management with per-team budgets, Prometheus metrics at `/metrics`, Langfuse tracing, audit logging, and secret management via AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault. There's also a web admin UI at `/admin/ui`.


## Why Go?

The LLM tooling ecosystem is heavily Python — great for prototyping, but Go is where a lot of production backend services actually live. llmbridge brings first-class LLM infrastructure into Go natively: no Python subprocess, no sidecar, no CGo.

The only external dependency is `modernc.org/sqlite`, a pure-Go SQLite driver used by the proxy's persistence layer.


## Get Started

```bash
go get github.com/Vedanshu7/llmbridge@latest
```

Requires Go 1.25+.

The next time a new model drops — or you need failover, caching, guardrails, or spend tracking — it's already there. No rewrites needed.

| Resource | Link |
|---|---|
| GitHub | [github.com/Vedanshu7/llmbridge](https://github.com/Vedanshu7/llmbridge) |
| Go Docs | [pkg.go.dev/github.com/Vedanshu7/llmbridge](https://pkg.go.dev/github.com/Vedanshu7/llmbridge) |
| Go Report Card | [goreportcard.com/report/github.com/Vedanshu7/llmbridge](https://goreportcard.com/report/github.com/Vedanshu7/llmbridge) |
| Issues / Feature Requests | [github.com/Vedanshu7/llmbridge/issues](https://github.com/Vedanshu7/llmbridge/issues) |
| Contributing Guide | [github.com/Vedanshu7/llmbridge/blob/main/CONTRIBUTING.md](https://github.com/Vedanshu7/llmbridge/blob/main/CONTRIBUTING.md) |
