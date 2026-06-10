# LLMs Were Trained to Guess. Here's How to Build Systems That Don't.

*The honest technical answer: hallucination is a math problem, not a behavior problem. Here's what that means for how you actually fix it.*

![Abstract glowing sphere of interconnected dots and lines, evoking a neural network](figs/llm-hallucinations.jpg)
*Photo by [Growtika](https://unsplash.com/@growtika) on [Unsplash](https://unsplash.com/photos/nGoCBxiaRO0)*


You ask an LLM for the CEO of a company. It gives you a name — confidently, fluently, with no hedging. The name is wrong. It doesn't exist. The model didn't know, and it didn't tell you it didn't know. It just filled in the gap with something plausible.

That's a hallucination. And understanding why it happens requires understanding what an LLM actually is at the mathematical level — because the fix is not "better prompting" alone, and it's not "use a smarter model" alone. The fix is architectural, and it requires a layered approach.


## What a Hallucination Actually Is

The term "hallucination" implies the model is confused or imagining things. That anthropomorphizes the problem in a way that leads you toward the wrong solutions.

The accurate description: **a hallucination is a statistically plausible token sequence that is factually wrong.**

The model didn't try to deceive you. It doesn't have a concept of deception or of truth. It has weight matrices. When you send a prompt, those matrices calculate a probability distribution over every possible next token. The model samples from that distribution. If the most probable next token happens to be factually incorrect, the model has no internal alarm that fires. It just outputs the token.

The confusion between "plausible" and "true" is the entire problem.


## Why It Happens: The Technical Root Causes

There are three distinct mechanisms that produce hallucinations, and you need to understand all three to apply the right mitigations.

### 1. The Architecture Is Probabilistic by Design

Every large language model is built on the Transformer architecture. At inference time, the model processes your prompt through layers of attention mechanisms and feed-forward networks, and at the end of each forward pass, it outputs a probability distribution over its vocabulary.

The math is roughly: given everything I've seen up to this point, what token comes next?

This is not retrieval. There is no database lookup. The model is not "looking up" the CEO of a company — it's computing which token sequence has historically followed contexts like this one in its training data. If the training data contained many sentences that used a particular name near the word "CEO" for a particular company, that name will have high probability. If the training data was sparse, wrong, or outdated, the model still has to output *something*. The probability distribution doesn't have a "null" option. It doesn't have an "I have no idea" token.

The model will always output a completion. The question is only how confident that completion's probability distribution is — not whether the output is grounded in fact.

### 2. Creativity and Hallucination Live in the Same Neurons

This is the uncomfortable one. Recent interpretability research into feed-forward layers has found that the circuits responsible for hallucination — the ones that produce plausible-but-invented content — are largely the same circuits responsible for logical reasoning and creative abstraction.

The model's ability to write poetry, summarize a novel, and solve a math problem all draw on the same capacity to generalize beyond what it has literally seen. That generalization is also what produces a plausible-sounding wrong CEO name.

You cannot surgically disable hallucination without disabling useful reasoning. Any intervention that prevents the model from "filling in" gaps will also reduce its ability to summarize, infer, and synthesize. This is why you can't solve the problem at the architecture level without tradeoffs — and why mitigation has to happen at the system level around the model, not inside it.

### 3. The Model Was Trained to Guess

The third mechanism is the most fixable — and the most overlooked.

Standard LLM training (and benchmark evaluation) uses a scoring scheme that strongly incentivizes guessing. A correct answer gets full credit. An incorrect answer gets zero. Saying "I don't know" also gets zero.

Given those rewards, the optimal strategy is always to guess. A random guess might be correct and earn points. Abstaining guarantees zero points. This is mathematically identical to the problem a student faces on a multiple-choice exam with no penalty for wrong answers.

The result: models are trained to be confidently wrong rather than honestly uncertain. The model's token-level probabilities are actually a reasonable signal of its internal uncertainty — but the training process optimized away from surfacing that uncertainty in the output.

This means the behavior is not a fundamental limitation of the architecture. It's a consequence of how the model was trained. Change the incentives, and you change the behavior.


## When Hallucinations Are Most Likely

Knowing the mechanism tells you which situations are highest risk:

**Rare or niche facts.** If a topic appeared rarely in training data, the model's probability distribution over relevant tokens is broad and uncertain. The model will still output confidently.

**Names, numbers, and dates.** These are the highest-risk categories. Language patterns provide strong priors for *types* of answers (a name, a year, a statistic) but weak priors for specific correct values. The model knows the shape of an answer, not the answer.

**Recent events.** Training cutoffs mean the model has no signal for post-cutoff events. But language patterns still provide strong generalization pressure, so the model fills the gap.

**Long documents and multi-hop reasoning.** As context length grows, attention mechanisms spread their weights across more tokens. Factual grounding weakens. The model is more likely to generate statements that are internally consistent with the conversation but detached from any specific true fact.

**Highly specific technical or legal claims.** The more precise the claim required, the more likely the model is to produce a plausible-sounding approximation rather than an accurate specific.


## Measure Before You Mitigate

Before applying any fix, establish a baseline. If you don't measure your current hallucination rate, you have no way to know whether your mitigations are actually working — or whether a prompt change introduced a regression.

Two frameworks do this well.

**Ragas** is built for RAG pipeline evaluation. It tracks three metrics that directly map to hallucination risk:

- **Faithfulness**: does the answer contain only claims derivable from the retrieved context? Low faithfulness means the model is supplementing retrieved facts with its own weights.
- **Answer Relevance**: does the answer actually address the question?
- **Context Recall**: did the retrieval step find the documents that contain the correct answer?

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_recall
from datasets import Dataset

# Collect outputs from your RAG pipeline across a test set
# Note: Ragas 0.2+ renamed "ground_truth" to "reference" — check your installed version
samples = {
    "question":     ["What year was Amazon founded?"],
    "answer":       ["Amazon was founded in 1994."],           # LLM output
    "contexts":     [["Amazon was founded on July 5, 1994."]],  # retrieved docs
    "ground_truth": ["1994"],  # use "reference" if on Ragas >= 0.2
}

dataset = Dataset.from_dict(samples)
results = evaluate(dataset, metrics=[faithfulness, answer_relevance, context_recall])
print(results)
# faithfulness: 1.0 — answer contains only claims from context
# answer_relevance: 0.95 — answer addresses the question
```

A faithfulness score below 0.8 on your test set is a direct signal that your retrieval and grounding setup isn't working — the model is adding things not in the source.

**DeepEval** takes a unit-testing approach. You write assertions against hallucination metrics the same way you'd write pytest tests for application logic:

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, FaithfulnessMetric

def test_answer_is_grounded():
    test_case = LLMTestCase(
        input="Who founded Amazon?",
        actual_output="Amazon was founded by Jeff Bezos in 1994.",
        context=["Amazon was founded on July 5, 1994 by Jeff Bezos in Bellevue, Washington."],
    )
    assert_test(test_case, [
        HallucinationMetric(threshold=0.1),  # fail if > 10% hallucinated
        FaithfulnessMetric(threshold=0.8),
    ])
```

Run DeepEval in CI on a curated test set every time you change your prompt, swap your model, or update your retrieval pipeline. You'll catch regressions immediately rather than after users report them.

The practical split: **DeepEval for regression testing in CI, Ragas for ongoing RAG pipeline evaluation in production.**


## How to Actually Fix It

There is no single solution. Hallucination prevention is a layered system problem. Each layer below addresses a different root cause.

### Layer 1: Ground the Model with RAG

The most impactful single change for most applications is Retrieval-Augmented Generation. Instead of asking the model to recall facts from its weights, you retrieve the relevant facts first and inject them into the context window.

The model's job changes from "recall and generate" to "read this text and answer based on it." This is a fundamentally easier task that produces far fewer hallucinations — because the model can produce a high-probability output without inventing anything.

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# One-time setup: index your documents into a vector store
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory="./db")

# At query time: retrieve relevant chunks first, then generate
docs = vectorstore.similarity_search(user_query, k=4)
context = "\n\n".join([d.page_content for d in docs])

prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't have that information."

Context:
{context}

Question: {user_query}"""
```

The system instruction matters as much as the retrieval. "Use only the context below" and "If the answer isn't in the context, say so" do real work. Without explicit constraints, the model blends retrieved context with its own weights — you get a mix of grounded and invented content.

RAG attacks the first root cause directly: it removes the need for the model to rely on its probabilistic weight representations of facts.

### Layer 2: Lower the Temperature

Temperature controls how the model samples from its output probability distribution. At temperature 1.0, it samples proportionally — low-probability tokens have a real chance. At temperature 0, it always selects the most probable next token.

For factual tasks, temperature 0 is almost always correct. It eliminates the random variation that produces low-probability tokens. It doesn't eliminate hallucinations from high-probability wrong answers, but it removes the entire class of hallucinations caused by sampling noise.

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    temperature=0,  # always pick the most probable token — no creative guessing
    messages=[{"role": "user", "content": query}],
)
```

One caveat: on Opus 4.7 and 4.8, sampling parameters including `temperature` are not accepted when `thinking` is enabled — the two are mutually exclusive. Use `temperature=0` for deterministic factual tasks where you don't need reasoning traces, and use adaptive thinking (`thinking={"type": "adaptive"}`) when you need the model to reason through a complex problem. Don't try to combine both.

### Layer 3: Force Structured Outputs

Free-form text generation gives the model maximum freedom to hallucinate. If your use case has a known output shape — a classification label, a structured record, a yes/no decision — constrain the output to that shape.

The cleanest way to do this with the Anthropic SDK is to define the output as a tool the model must call, with a JSON schema enforcing the fields:

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=256,
    tools=[{
        "name": "emit_decision",
        "description": "Emit a structured classification decision.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["billing", "technical", "general", "escalate"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"},
            },
            "required": ["category", "confidence", "reasoning"],
        },
    }],
    tool_choice={"type": "tool", "name": "emit_decision"},
    messages=[{"role": "user", "content": customer_message}],
)

decision = response.content[0].input
if decision["confidence"] < 0.7:
    route_to_human(customer_message)  # low confidence → human review
```

When the model is forced into a constrained output space, there is no room to hallucinate extra fields. The `enum` on `category` means it cannot invent a new category. The `confidence` field, when you actually use it as a threshold, lets you catch the uncertain cases before they reach a user.

This attacks the second root cause: instead of asking the model to generalize into open-ended text, you define the output space and make hallucination outside that space structurally impossible.

### Layer 4: Chain-of-Thought Before the Answer

Ask the model to reason before it concludes. "Think step-by-step" is not just a performance booster — it forces the model to surface its intermediate reasoning, which makes contradictions and unsupported leaps visible and catchable.

```python
system = """You are a factual assistant.

Before giving any answer:
1. List the specific facts you are confident about
2. Identify any gaps in your knowledge
3. State your conclusion with appropriate uncertainty

If any required fact is missing or uncertain, your conclusion must reflect that uncertainty.
Never present a guess as a fact."""
```

Chain-of-thought works because the model's output at each step becomes part of the context for the next step. If the model writes "I am not certain about the founding date," that uncertainty is now in-context and shifts the probability distribution for the final answer tokens — it literally makes the model less likely to produce a confidently wrong conclusion.

### Layer 5: Multi-Agent Verification

One model generates, a second evaluates. This structural pattern catches hallucinations that the first model doesn't catch itself.

```python
import anthropic

client = anthropic.Anthropic()

def generate_and_verify(question: str, context: str) -> dict:
    # First model: generate an answer grounded in context
    generation = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": f"Answer using only the context below.\n\nContext: {context}\n\nQuestion: {question}"
        }],
    )
    # Extract the text block (skip thinking blocks)
    answer = next(b.text for b in generation.content if b.type == "text")

    # Second model: verify every claim in the answer against the context
    verification = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        thinking={"type": "adaptive"},
        tools=[{
            "name": "verdict",
            "input_schema": {
                "type": "object",
                "properties": {
                    "grounded": {"type": "boolean"},
                    "issues":   {"type": "string"},
                },
                "required": ["grounded", "issues"],
            },
        }],
        tool_choice={"type": "tool", "name": "verdict"},
        messages=[{
            "role": "user",
            "content": (
                "Does this answer contain only claims directly supported by the context? "
                "Flag any claim not found in the context.\n\n"
                f"Context: {context}\n\nAnswer: {answer}"
            ),
        }],
    )

    verdict = next(b for b in verification.content if b.type == "tool_use").input
    return {
        "answer":   answer if verdict["grounded"] else None,
        "grounded": verdict["grounded"],
        "issues":   verdict["issues"],
    }
```

The cost is two API calls instead of one. For high-stakes outputs — medical, legal, financial — that cost is cheap relative to the cost of a wrong answer reaching a user.

### Layer 6: Rewire the Penalty Structure

This is the least-used lever and potentially the most impactful.

The [arXiv paper "Why Language Models Hallucinate"](https://arxiv.org/abs/2509.04664) formalizes what the benchmark paradox looks like mathematically: under binary scoring, abstaining and guessing wrong are both penalized equally, so the model always guesses. The optimal strategy changes entirely if you introduce a penalty for confident wrong answers that exceeds the penalty for honest abstention.

You can approximate this today without retraining anything. In your system prompt, make the cost structure explicit — and crucially, flip the framing away from numeric thresholds (which the model can't self-assess precisely) toward qualitative honesty signals:

```
RESPONSE POLICY

Before answering, assess how confident you actually are in the specific claim.

- Highly confident — you have clear recall or it's in the provided context:
  Answer directly.

- Somewhat confident but not certain:
  Answer, and explicitly flag which parts you're less sure about.

- Uncertain about key facts:
  State what you know, name the specific gap, and suggest where to verify.

- Guessing:
  Do not answer. Say exactly what information is missing.

You will be penalized more for presenting a guess as a fact than for admitting
you don't know. Honest uncertainty is always the correct output when you lack
confidence. Fabricating a confident-sounding answer is the worst possible outcome.
```

This is prompt-level behavior modification, not architecture-level retraining — but it works because it shifts the model's in-context optimization target. The model's reasoning about "what is the best response here" is influenced by the stated evaluation criteria. Explicitly telling the model that confident fabrication is more penalized than abstention changes the probability landscape of its output.

For teams training or fine-tuning their own models, the paper's implication is direct: use a scoring scheme that grants partial credit for calibrated abstention and heavily penalizes confident wrong answers. The paper notes that smaller models trained this way outperform larger models in reliability — a small model that only speaks when it's confident is more useful in production than a large model that always answers.

### Layer 7: Constrained Decoding with Guidance or Outlines

For applications where the valid output vocabulary is fully known — a classification label, a regex pattern, a fixed set of values — constrained decoding goes further than JSON schema enforcement. It restricts the model's token sampling *during* the forward pass, making it physically impossible to produce out-of-vocabulary tokens.

[Guidance](https://github.com/guidance-ai/guidance) and [Outlines](https://github.com/dottxt-ai/outlines) both implement this. The key distinction from API-level schema validation is timing: constrained decoding prevents bad tokens from being generated at all, whereas schema enforcement catches them after generation.

```python
from outlines import models, generate

# Load a local model
model = models.transformers("mistralai/Mistral-7B-v0.1")

# Force output to always match an ISO date — no other output is possible
date_generator = generate.regex(model, r"\d{4}-\d{2}-\d{2}")
date = date_generator("When was the Eiffel Tower completed?")
# Output will always be a valid YYYY-MM-DD string — hallucinating a non-date is impossible
```

This approach requires controlling the inference stack, so it's most useful for open-source models you're running locally. For hosted APIs, you're limited to schema enforcement at the API boundary (Layer 3). That said, the two approaches solve slightly different problems — constrained decoding is near-zero-cost at inference time, while API schema enforcement has no model-infrastructure requirements.

### Layer 8: Fine-Tune for High-Stakes Domains

For medicine, law, finance, or any domain where factual precision matters and errors carry real consequences, prompt engineering and retrieval alone may not be enough. The base model's weight distribution over domain-specific terminology, case references, and technical concepts may be sparse, forcing the model to generalize across domains in ways that produce plausible-but-wrong outputs.

Fine-tuning on a curated, high-quality dataset of domain-specific facts does two things: it shifts the model's probability distribution toward correct in-domain token sequences, and it reduces reliance on cross-domain generalization.

The practical constraint is data quality. A fine-tuned model trained on noisy or incorrect domain data will hallucinate domain-specific wrong answers with *higher* confidence than the base model. The bar for fine-tuning data is higher than for most ML tasks — if you're not certain your training data is correct, stay with RAG until you are.


## Putting It Together

These layers aren't mutually exclusive. A production system should combine them based on the cost of a wrong answer. The full stack looks like this:

```
User query
    │
    ▼
[Measure] Ragas / DeepEval baseline before any change
    │
    ▼
[RAG] Retrieve relevant documents, inject into context
    │
    ▼
[Temperature 0] Remove sampling noise for factual tasks
    │
    ▼
[Structured output] Constrain response to a known schema
    │
    ▼
[CoT prompt] Require step-by-step reasoning before conclusion
    │
    ▼
[Confidence threshold] Route uncertain outputs to human review
    │
    ▼
[Verifier] Second model checks every claim against source documents
    │
    ▼
Response
```

You don't need all eight layers for every application. A customer support bot that classifies tickets needs RAG, structured output, and a confidence threshold. A medical information tool needs all eight. Match the layers to the cost of a wrong answer — not to how impressive the stack sounds.


## What Doesn't Work

A few common non-solutions worth naming:

**"Just use a smarter model."** Larger models hallucinate less frequently on common knowledge, but they hallucinate with *greater confidence* on niche knowledge. A more capable model that doesn't know something is still going to fill in the gap — it just sounds more authoritative doing it.

**"Tell the model not to make things up."** Instructions like "do not hallucinate" in a system prompt have marginal effect. The model isn't choosing to hallucinate. It's outputting the most probable next token. A general instruction doesn't change the underlying probability distribution for specific factual gaps.

**"Use a model with a larger context window."** More context doesn't prevent hallucination — it can amplify it. As context length grows, attention spreads, and factual grounding weakens. Long-context models still need RAG and verification.


## Closing Thoughts

Hallucinations are not a bug the LLM vendors are going to fix in the next release. They're a consequence of how language models work. The architecture produces plausible token sequences, not verified facts. The training process rewards confident outputs. The output space is unbounded.

The layered approach above — RAG for grounding, constrained outputs for precision, multi-agent verification for critical paths, measurement in CI so regressions don't slip through — doesn't eliminate hallucinations. Nothing does, at the current state of the art. But applied together, they reduce the probability of a hallucination reaching a user substantially, and they make the remaining uncertain outputs visible rather than silent.

The most important mental model shift is this: treat the LLM as a reasoning engine with unreliable recall, not as a database. Give it the facts. Constrain its outputs. Verify its conclusions. The model's reasoning ability is genuinely impressive — the factual recall is not. Build your system around that asymmetry.


*If you've run into a specific hallucination pattern that none of the above layers caught, I'd be curious — leave a note in the comments. The failure modes are as varied as the applications, and the edge cases are where you learn the most.*
