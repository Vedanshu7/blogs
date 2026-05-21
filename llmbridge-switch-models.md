# Stop Rewriting LLM Code — llmbridge Gives Go One Interface for All of It

[![Go Reference](https://pkg.go.dev/badge/github.com/Vedanshu7/llmbridge.svg)](https://pkg.go.dev/github.com/Vedanshu7/llmbridge)
[![CI](https://github.com/Vedanshu7/llmbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/Vedanshu7/llmbridge/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/Vedanshu7/llmbridge)](https://goreportcard.com/report/github.com/Vedanshu7/llmbridge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Vedanshu7/llmbridge/blob/main/LICENSE)

**GitHub:** [github.com/Vedanshu7/llmbridge](https://github.com/Vedanshu7/llmbridge) · **Docs:** [pkg.go.dev/github.com/Vedanshu7/llmbridge](https://pkg.go.dev/github.com/Vedanshu7/llmbridge)

![Abstract blue background with connected technology-related icons](https://plus.unsplash.com/premium_photo-1764691292167-7263c8b464c0?fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.0)
*Photo by [Getty Images](https://unsplash.com/@gettyimages) on [Unsplash](https://unsplash.com/photos/abstract-blue-background-with-connected-technology-related-icons-1IJnGLTv7SU)*


## The Problem Nobody Talks About

A new model drops. GPT-4o. Claude 4. Gemini 2.0. Whatever it is — faster, smarter, cheaper. You want it.

Then you open your codebase and find this scattered across a dozen files:

```go
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
        time.Sleep(5 * time.Second)
        // retry...
    }
}
output := resp.Choices[0].Message.Content
```

Switching providers is never just changing the model string. It means:

- A different SDK with different import paths and initialization
- Different request structs (`ChatCompletionRequest` vs `MessagesRequest` vs `GenerateContentRequest`)
- Different response shapes (`resp.Choices[0].Message.Content` vs `resp.Content[0].Text`)
- Different streaming implementations — each provider has its own SSE format
- Different error types — you can't `errors.As` an OpenAI error into an Anthropic error
- Different token counting, different cost tables, different rate limit behavior

If any of this is spread across your codebase, "upgrading to the latest model" is a multi-day refactor — not an afternoon.

But the migration is only the first layer of the problem. Once you're past it, you still don't have:

- **Failover** — what happens when OpenAI has an outage at 2am?
- **Caching** — why pay for the same response twice?
- **Spend tracking** — how do you know when a key or team blows past budget?
- **Guardrails** — who's checking for PII or prompt injection before the request hits the model?
- **Observability** — can you explain to your CTO what your LLM spend looked like last week?

Every team ends up bolting these on themselves, differently, on top of different SDKs. It's the kind of accidental infrastructure that makes codebases painful to work in six months later.

**[llmbridge](https://github.com/Vedanshu7/llmbridge)** is a Go library that solves all of this at once: a single unified interface across every major LLM provider, with production-grade infrastructure built in from day one.


## One Interface. Every Provider. Zero Rewrites.

Write your code once against the `llmbridge.Provider` interface:

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
if err != nil {
    return "", err
}
fmt.Println(resp.Content)
```

Now a new model drops. Change **one line**:

```go
// Switch to Anthropic Claude 4
provider := anthropic.New("claude-opus-4-7", os.Getenv("ANTHROPIC_API_KEY"))

// Switch to Google Gemini
provider := gemini.New("gemini-2.0-flash", os.Getenv("GEMINI_API_KEY"))

// Switch to Groq for fast inference
provider := compatible.NewGroq("llama-3.3-70b-versatile", os.Getenv("GROQ_API_KEY"))

// Switch to a local Ollama model — no API key, no cost
provider := compatible.NewOllama("llama3.2")

// Switch to AWS Bedrock for compliance reasons
provider := bedrock.New("anthropic.claude-3-sonnet-20240229-v1:0", "us-east-1")
```

Everything below that line — the `Complete` call, the error handling, the streaming logic, the cost tracking — stays exactly the same. Every provider speaks the same interface.

**Supported providers out of the box:**

| Provider | Package |
|---|---|
| OpenAI | `llms/openai` |
| Anthropic Claude | `llms/anthropic` |
| Google Gemini | `llms/gemini` |
| AWS Bedrock | `llms/bedrock` |
| Azure OpenAI | `llms/azure` |
| Cohere Command | `llms/cohere` |
| Groq | `llms/compatible` |
| Ollama (local) | `llms/compatible` |
| LM Studio | `llms/compatible` |
| Together AI | `llms/compatible` |
| xAI / Grok | `llms/compatible` |
| Any OpenAI-compatible endpoint | `llms/compatible` |


## Streaming — Same API, Every Provider

Streaming is usually the most painful part of any LLM migration. OpenAI uses server-sent events with `delta.content`. Anthropic uses a different event schema. Gemini has its own format. Under the hood they are all different, but llmbridge normalizes all of it into one clean channel-based API:

```go
ch, err := provider.Stream(ctx, llmbridge.Request{
    Messages: []llmbridge.Message{
        {Role: "user", Content: "Write a short story about a lighthouse keeper."},
    },
})
if err != nil {
    return err
}

for delta := range ch {
    if delta.Err != nil {
        return delta.Err
    }
    if delta.Done {
        fmt.Printf("\n\nTotal tokens used: %d\n", delta.Usage.TotalTokens)
        break
    }
    fmt.Print(delta.Content)
}
```

Swap `provider` from OpenAI to Anthropic to Gemini to Ollama — this loop never changes. The token usage is normalized too, so you can track costs consistently regardless of provider.


## Tool Use / Function Calling

Tool use is one of the most valuable LLM capabilities, and one of the most inconsistently implemented across providers. llmbridge gives you a single `Tool` definition that works everywhere:

```go
tools := []llmbridge.Tool{
    {
        Name:        "get_weather",
        Description: "Get the current weather for a given city.",
        Parameters: llmbridge.Schema{
            Type: "object",
            Properties: map[string]llmbridge.Property{
                "city":    {Type: "string", Description: "The city name"},
                "unit":    {Type: "string", Enum: []string{"celsius", "fahrenheit"}},
            },
            Required: []string{"city"},
        },
    },
    {
        Name:        "search_web",
        Description: "Search the web for up-to-date information.",
        Parameters: llmbridge.Schema{
            Type: "object",
            Properties: map[string]llmbridge.Property{
                "query": {Type: "string", Description: "The search query"},
            },
            Required: []string{"query"},
        },
    },
}

resp, err := provider.Complete(ctx, llmbridge.Request{
    Messages: []llmbridge.Message{
        {Role: "user", Content: "What's the weather in Tokyo right now?"},
    },
    Tools: tools,
})

// If the model wants to call a tool:
if len(resp.ToolCalls) > 0 {
    for _, call := range resp.ToolCalls {
        fmt.Printf("Tool: %s, Args: %s\n", call.Name, call.Arguments)
        // execute the tool, feed result back
    }
}
```

Define your tools once. They work identically whether the model is GPT-4o, Claude, Gemini, or a local Llama. No translating between OpenAI's `function_call` format and Anthropic's `tool_use` format — llmbridge handles the wire-format transformation for you.

You can also use the `toolbuilder` package for a fluent, type-safe way to define tools:

```go
import "github.com/Vedanshu7/llmbridge/toolbuilder"

tool := toolbuilder.New("get_weather", "Get current weather for a city").
    StringParam("city", "The city name", true).
    StringEnumParam("unit", "Temperature unit", []string{"celsius", "fahrenheit"}, false).
    Build()
```


## Typed Errors Across Every Provider

Every SDK ships its own error struct and its own way of signaling rate limits, auth failures, or context overflows. With vendor SDKs, you end up writing provider-specific error handling in every service. llmbridge maps all provider errors into a common typed hierarchy:

```go
import "github.com/Vedanshu7/llmbridge/exceptions"

resp, err := provider.Complete(ctx, req)
if err != nil {
    var authErr *exceptions.AuthenticationError
    var rlErr   *exceptions.RateLimitError
    var cwErr   *exceptions.ContextWindowExceededError
    var cpErr   *exceptions.ContentPolicyViolationError
    var toErr   *exceptions.TimeoutError

    switch {
    case errors.As(err, &authErr):
        // authErr.LLMProvider tells you which provider rejected the key
        log.Fatalf("invalid API key for provider: %s", authErr.LLMProvider)

    case errors.As(err, &rlErr):
        // back off and retry
        time.Sleep(5 * time.Second)

    case errors.As(err, &cwErr):
        // the prompt is too long for this model
        // route to a provider with a larger context window
        log.Printf("context exceeded (%d tokens), switching model", cwErr.TokenCount)

    case errors.As(err, &cpErr):
        // content policy violation — log and reject
        log.Printf("content policy blocked by %s", cpErr.LLMProvider)

    case errors.As(err, &toErr):
        // network or inference timeout
        log.Printf("request timed out after %s", toErr.Duration)
    }
}
```

Write this error-handling block once. It works identically regardless of which provider is behind `provider`. When you add a new provider to your stack, nothing in this code changes.

Full list of typed errors: `AuthenticationError` · `RateLimitError` · `TimeoutError` · `ContextWindowExceededError` · `ContentPolicyViolationError` · `BudgetExceededError` · `InternalServerError` · `ServiceUnavailableError` · and more.


## Multi-Provider Router — Failover, Retries, Circuit Breaker

Real production workloads need more than one provider. OpenAI goes down. Anthropic throttles you during a traffic spike. You want to test a new model without risking availability. The llmbridge router handles all of this:

```go
router := llmbridge.NewRouter(
    []llmbridge.Provider{
        anthropic.New("claude-opus-4-7", os.Getenv("ANTHROPIC_API_KEY")),
        openai.New("gpt-4o", os.Getenv("OPENAI_API_KEY")),
        compatible.NewGroq("llama-3.3-70b-versatile", os.Getenv("GROQ_API_KEY")),
    },
    llmbridge.WithStrategy(llmbridge.PriorityOrder),     // try in order, fall through on failure
    llmbridge.WithRetryPolicy(llmbridge.DefaultRetryPolicy),
    llmbridge.WithCircuitBreaker(5, 30*time.Second),     // trip after 5 failures, reset after 30s
    llmbridge.WithContextWindowFallback(true),           // auto-reroute if context is exceeded
)

resp, err := router.Complete(ctx, req)
```

If the primary provider is down, rate-limited, or trips the circuit breaker, the router falls through to the next automatically — no retries in your application code needed.

**Six routing strategies:**

| Strategy | When to use |
|---|---|
| `PriorityOrder` | Always try your preferred provider first, fall back on failure |
| `RoundRobin` | Distribute load evenly across providers |
| `LeastLatency` | Route to whichever provider responded fastest recently |
| `LeastBusy` | Route to the provider with fewest in-flight requests |
| `CostBased` | Always route to the cheapest provider that can handle the request |
| `Weighted` | Custom traffic split — e.g. 70% OpenAI, 30% Anthropic |

The `WithContextWindowFallback` option is especially useful during model transitions: if a user's conversation exceeds the primary model's context limit, the router automatically re-routes to a provider with a larger window rather than returning an error.


## Caching — Exact Match and Semantic

LLM API calls are expensive. If your application asks the same or similar questions repeatedly — a documentation bot, a Q&A system, a product search assistant — you're paying for the same tokens over and over. llmbridge ships four cache backends with no extra infrastructure required for the common cases:

```go
import "github.com/Vedanshu7/llmbridge/caching"

// In-memory — zero setup, great for development or single-instance apps
memCache := caching.NewInMemoryCache()

// Disk cache — survives restarts, good for batch jobs
diskCache := caching.NewDiskCache("/tmp/llmbridge-cache")

// Redis — distributed, share cache across multiple instances
redisCache := caching.NewRedisCache("redis://localhost:6379")
```

The most powerful option is the semantic cache, which uses vector similarity to match requests that mean the same thing even if the exact wording differs:

```go
// Semantic cache — hits on meaning, not just exact text
embedder := openai.New("text-embedding-3-small", key)
sc := caching.NewSemanticCache(memCache, embedder, 0.95)

// First call — cache miss, hits the API
resp1, _ := sc.Complete(ctx, req("What is the capital of France?"))

// Second call — cache hit, returns instantly at zero cost
// "Which city is the capital of France?" is close enough (cosine > 0.95)
resp2, _ := sc.Complete(ctx, req("Which city is the capital of France?"))
```

The cosine similarity threshold (0.95 in the example) controls how aggressive the matching is. A lower threshold caches more aggressively; a higher one is more conservative. Any `llmbridge.Provider` can be wrapped with a cache — including the router.


## Budget & Spend Tracking

LLM costs compound fast, especially across multiple teams or API keys. llmbridge has a budget tracker with per-key and per-org limits built in:

```go
import "github.com/Vedanshu7/llmbridge/budget"

tracker := budget.NewTracker()

// Set limits at any granularity
tracker.SetLimit("team-frontend", 50.00)    // $50/month for the frontend team
tracker.SetLimit("team-backend", 200.00)    // $200/month for the backend team
tracker.SetLimit("prod-key", 1000.00)       // hard cap on the production key

// Alert callback fires before the limit is hit — give yourself warning
tracker.OnAlert(func(key string, spend float64) {
    // send a Slack message, page on-call, update a dashboard
    log.Printf("[BUDGET ALERT] %s has spent $%.2f", key, spend)
})

// After each completion, record the cost
cost, err := llmbridge.CompletionCost(resp)
if err == nil {
    if err := tracker.Record("team-backend", cost); err != nil {
        // budget.ErrBudgetExceeded — hard stop
        return fmt.Errorf("budget exceeded: %w", err)
    }
}
```

Cost tables for every supported model are built-in and kept up to date — `CompletionCost` works without any configuration on your part. You don't have to maintain a pricing spreadsheet.


## Guardrails — Safety at the Edge

Every LLM-powered feature is a potential attack surface. Users submit prompts you didn't design for. Responses contain information they shouldn't. Guardrails let you define rules that run before requests are sent and before responses are returned — without touching your business logic:

```go
import "github.com/Vedanshu7/llmbridge/guardrails"

engine, _ := guardrails.NewEngine(
    guardrails.MaxInputLength(50_000),       // reject oversized inputs
    guardrails.MaxOutputLength(10_000),      // cap response length
    guardrails.BlockPIIPatterns(),           // detect emails, phone numbers, SSNs, credit cards
    guardrails.BlockPromptInjection(),       // catch "ignore previous instructions" style attacks
    guardrails.BlockKeywords([]string{       // domain-specific blocklist
        "confidential", "internal only",
    }),
)

// Check input before sending
if err := engine.CheckInput(req); err != nil {
    return fmt.Errorf("input rejected by guardrails: %w", err)
}

resp, err := provider.Complete(ctx, req)
if err != nil {
    return err
}

// Check output before returning to the user
if err := engine.CheckOutput(resp); err != nil {
    return fmt.Errorf("output rejected by guardrails: %w", err)
}
```

Guardrails run entirely locally — there's no additional network hop. Catching a bad request before it reaches the model saves you the cost of the token round-trip too.


## Middleware — Composable Cross-Cutting Concerns

The middleware chain works exactly like HTTP middleware. You stack behaviors on top of a provider without modifying it, and any number of middleware layers can be composed together:

```go
// Structured latency + token logging
func Logger(log *slog.Logger) llmbridge.Middleware {
    return func(ctx context.Context, req llmbridge.Request, next llmbridge.Handler) (*llmbridge.Response, error) {
        start := time.Now()
        resp, err := next(ctx, req)
        if err == nil {
            log.Info("llm call",
                "provider", resp.Provider,
                "model", resp.Model,
                "latency_ms", time.Since(start).Milliseconds(),
                "prompt_tokens", resp.Usage.PromptTokens,
                "completion_tokens", resp.Usage.CompletionTokens,
            )
        }
        return resp, err
    }
}

// Automatic retry on transient errors
func Retry(maxAttempts int) llmbridge.Middleware {
    return func(ctx context.Context, req llmbridge.Request, next llmbridge.Handler) (*llmbridge.Response, error) {
        var err error
        for i := 0; i < maxAttempts; i++ {
            var resp *llmbridge.Response
            resp, err = next(ctx, req)
            if err == nil {
                return resp, nil
            }
            var rlErr *exceptions.RateLimitError
            if !errors.As(err, &rlErr) {
                break // don't retry non-transient errors
            }
            time.Sleep(time.Duration(i+1) * 2 * time.Second)
        }
        return nil, err
    }
}

// Stack them together
provider := llmbridge.Chain(
    openai.New("gpt-4o", key),
    Logger(slog.Default()),
    Retry(3),
)
```

Because the middleware interface is just a function wrapping the `Handler` type, you can build anything: request enrichment, response transformation, A/B testing, custom rate limiting, Datadog tracing — without the provider knowing about any of it.


## Embeddings

Embeddings are first-class in llmbridge. If your provider supports them, you get a consistent `Embed` method with no extra setup:

```go
import "github.com/Vedanshu7/llmbridge/llms/openai"

embedder := openai.New("text-embedding-3-small", key)

vecs, err := embedder.Embed(ctx, []string{
    "llmbridge is a Go library for LLM providers",
    "how do I switch between GPT and Claude?",
    "what is semantic caching?",
})
if err != nil {
    return err
}

// vecs[0], vecs[1], vecs[2] are []float64 — store in pgvector, Pinecone, Weaviate, etc.
fmt.Printf("embedding dimensions: %d\n", len(vecs[0]))
```

The same `Embed` interface is available on any provider that supports embeddings. You can swap between `text-embedding-3-small` and `text-embedding-3-large` — or move to a different provider's embedding model — without changing downstream code.


## Cost Estimation

Before you send a request, you can estimate what it will cost. After you get a response, you can calculate exactly what it cost. Both use the same built-in pricing tables:

```go
// Estimate cost before sending (based on token count)
estimated, err := llmbridge.EstimateRequestCost("gpt-4o", req)
if err == nil {
    fmt.Printf("estimated cost: $%.6f\n", estimated)
}

// Exact cost after the response (uses actual usage from the API)
resp, _ := provider.Complete(ctx, req)
actual, err := llmbridge.CompletionCost(resp)
if err == nil {
    fmt.Printf("actual cost: $%.6f  (prompt: %d tokens, completion: %d tokens)\n",
        actual, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
}
```

Pricing tables cover every supported model and are updated as providers change their prices. You never have to maintain a rate card yourself.


## OpenAI-Compatible Proxy Server

For teams with multiple services or multiple languages, llmbridge can run as a shared infrastructure proxy. Any client that speaks the OpenAI API — Python, Node, Ruby, a LangChain app, a LlamaIndex pipeline, a curl command — points at llmbridge instead, and immediately gets routing, caching, auth, and observability without a single code change on the client side.

```go
import (
    "github.com/Vedanshu7/llmbridge/proxy"
    "github.com/Vedanshu7/llmbridge/llms/anthropic"
)

backend := anthropic.New("claude-opus-4-7", os.Getenv("ANTHROPIC_API_KEY"))
srv, err := proxy.NewServerWithDB(backend, "/data/llmbridge.db")
if err != nil {
    log.Fatal(err)
}

// Generate a key for a client
key, _ := srv.KeyStore().GenerateAPIKey([]string{"completion", "embeddings"})
fmt.Println("Client API key:", key)

srv.Start(ctx, ":8080")
```

Clients connect like any OpenAI client, just pointed at your proxy:

```python
# Python client — no code change except the base_url
import openai
client = openai.OpenAI(base_url="http://your-proxy:8080/v1", api_key="llmb-your-key")
response = client.chat.completions.create(model="claude-opus-4-7", messages=[...])
```

Or via Docker with a config file:

```bash
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY \
  -v ./config.json:/config.json:ro \
  ghcr.io/vedanshu7/llmbridge server -config /config.json
```

A minimal `config.json` to get started:

```json
{
  "listen_addr": ":8080",
  "jwt_secret": "change-me-in-production",
  "admin_keys": ["llmb-your-admin-key"],
  "models": [
    {"name": "claude-opus-4-7", "provider": "anthropic", "model": "claude-opus-4-7"},
    {"name": "gpt-4o",          "provider": "openai",    "model": "gpt-4o"}
  ],
  "router": {"strategy": "round_robin", "retries": 2},
  "guardrails": {
    "max_input_length": 100000,
    "block_pii": true,
    "block_prompt_injection": true
  }
}
```

The proxy ships with:

- **API key management** — generate, revoke, and scope keys per client or team
- **SSO/OIDC** — Google, GitHub, and Microsoft login out of the box
- **Org and team management** — per-team budgets and rate limits
- **Prometheus metrics** at `/metrics` — ready to scrape into Grafana
- **Langfuse tracing** — full request/response traces with latency and cost
- **Audit logging** — every request recorded with key, model, tokens, and cost
- **Secret management** — pull API keys from AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault
- **Web admin UI** at `/admin/ui` — manage keys, models, and orgs without writing code


## Why Go?

The LLM tooling ecosystem is overwhelmingly Python. That's fine for notebooks and experimentation. But most production backend infrastructure — APIs, microservices, CLI tools, data pipelines — runs on Go. Until now, Go developers had to either wrap Python processes, use thin OpenAI-only SDKs, or write their own provider abstractions.

llmbridge is the library that should have existed. It brings production-grade LLM infrastructure natively into Go:

- **No Python subprocess** — everything runs in-process
- **No sidecar** — no separate service to deploy or maintain
- **No CGo** — pure Go, cross-compiles to any target
- **One external dependency** — `modernc.org/sqlite` (pure Go, no CGo) for the proxy's persistence layer
- **Idiomatic Go** — channels for streaming, `errors.As` for error handling, interfaces for extensibility

If you want to add a new provider that llmbridge doesn't support yet, you implement one interface:

```go
type Provider struct { /* your fields */ }

func (p *Provider) Name() string { return "myprovider" }
func (p *Provider) Complete(ctx context.Context, req types.Request) (*types.Response, error) {
    // translate req to your provider's wire format, call their API, translate back
}
```

Optionally implement `base.Streamer` for streaming, `base.EmbedProvider` for embeddings, or `base.ImageGenerator` for image generation. Then open a PR — the contribution guide is in [CONTRIBUTING.md](https://github.com/Vedanshu7/llmbridge/blob/main/CONTRIBUTING.md).


## Get Started

```bash
go get github.com/Vedanshu7/llmbridge@latest
```

Requires Go 1.25+. Here's a complete working example from zero to streaming response:

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/Vedanshu7/llmbridge"
    "github.com/Vedanshu7/llmbridge/llms/openai"
)

func main() {
    ctx := context.Background()

    // swap this one line to change providers
    provider := openai.New("gpt-4o-mini", os.Getenv("OPENAI_API_KEY"))

    ch, err := provider.Stream(ctx, llmbridge.Request{
        System: "You are a helpful assistant.",
        Messages: []llmbridge.Message{
            {Role: "user", Content: "What makes Go a good language for backend services?"},
        },
    })
    if err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }

    for delta := range ch {
        if delta.Err != nil {
            fmt.Fprintln(os.Stderr, delta.Err)
            os.Exit(1)
        }
        if delta.Done {
            break
        }
        fmt.Print(delta.Content)
    }
    fmt.Println()
}
```

The next time a new model drops — or your primary provider has an outage — you're one line away from the fix. No migrations, no rewrites, no weekend incidents.

| Resource | Link |
|---|---|
| GitHub | [github.com/Vedanshu7/llmbridge](https://github.com/Vedanshu7/llmbridge) |
| Go Docs | [pkg.go.dev/github.com/Vedanshu7/llmbridge](https://pkg.go.dev/github.com/Vedanshu7/llmbridge) |
| Go Report Card | [goreportcard.com/report/github.com/Vedanshu7/llmbridge](https://goreportcard.com/report/github.com/Vedanshu7/llmbridge) |
| Issues / Feature Requests | [github.com/Vedanshu7/llmbridge/issues](https://github.com/Vedanshu7/llmbridge/issues) |
| Contributing Guide | [github.com/Vedanshu7/llmbridge/blob/main/CONTRIBUTING.md](https://github.com/Vedanshu7/llmbridge/blob/main/CONTRIBUTING.md) |
