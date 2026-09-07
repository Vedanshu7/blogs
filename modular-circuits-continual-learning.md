# Growing a Model Instead of Retraining It

*A research agenda for models that gain one capability at a time, without touching the other billion parameters.*

![Macro photograph of a green printed circuit board, with specialised chips and components soldered onto a shared base](figs/modular-circuits-continual-learning.jpg)

*Photo by [Ludovico Ceroseis](https://unsplash.com/@ludovico_06) on [Unsplash](https://unsplash.com/photos/AEboEnOlpLc)*

You ship a model. Three weeks later the world moves: a new API version, a regulation that did not exist, a product line nobody had named yet. Your model knows none of it, and every option in front of you is bad. You can stuff the context window and pay that tax on every single request. You can fine-tune and hope you did not quietly break something you were not measuring. Or you can wait for the next base model and inherit somebody else's training cutoff.

All three options share an assumption so ingrained that it rarely gets stated: the model is one indivisible artefact. Change anything and you change everything. Learning a new fact and forgetting how to write Python are the same operation applied to the same weights.

Biology does not work this way. An organism does not rewrite its entire genome to express a new protein. The base structure stays remarkably stable while specialised machinery gets built on top of it. What would it take to give a neural network that property?


## The Idea

The proposal is a loop rather than a training run. A meta-model watches the base model operating on live data. When the base model fails in a way that looks systematic rather than random, the meta-model characterises what is missing, builds a small specialised circuit for it, trains only that circuit, proves the circuit actually causes the behaviour you wanted, and then wires it into the base model.

![Diagram of the circuit growth loop: live data flows into a meta-model that names the missing capability, which selects or grows a circuit, trains it with base weights frozen, then passes it through causal validation which either rejects the circuit for a retry or verifies it for integration into the base model](figs/circuit-growth-loop.png)

The base model is never fully retrained. It accumulates capabilities the way a codebase accumulates modules, and the interesting property is that each addition is independently verifiable before it lands.

The genetic analogy is useful right up to the point where it misleads you. Genes are not tidy modules with clean interfaces; expression is contextual, and the same gene participates in unrelated processes depending on what else is active. That turns out to be a fair description of neural networks too, and it is exactly where this proposal gets difficult. Hold that thought.


## Why This Is Not Just Mixture of Experts

This is the first objection any informed reader will raise, and it deserves a direct answer, because most of the individual pieces here already exist.

**Mixture of Experts** gives you specialised subnetworks and a router that picks between them. But the experts and the router are trained jointly, up front, as part of one optimisation. No expert is ever discovered after the fact, and you cannot add expert number 65 to a trained model because you noticed a gap on Tuesday.

**LoRA and adapters** already solve selective learning. You freeze the base and train a small number of new parameters, which is precisely step three of the loop. What they do not do is decide anything. A human picks the task, curates the data, chooses the rank, and decides when to merge. The adapter is a tool, not an agent.

**Progressive networks** grow architecture as new tasks arrive, which covers dynamic growth. But growth is scheduled by the experimenter, one column per task, with the task boundaries known in advance. The network never notices it needs a new column.

**Model editing** methods such as ROME locate and rewrite specific factual associations, which is genuinely close in spirit. The scope is narrower though: editing a known fact you can already name, rather than discovering that a capability is absent and constructing it.

**Continual learning** research targets catastrophic forgetting, and techniques like elastic weight consolidation directly address the stability problem. It is the defensive half of this agenda. It tells you how to avoid losing what you had; it does not tell you how to acquire something new autonomously.

**Mechanistic interpretability** has produced real circuit discovery, including automated methods. But the discipline is descriptive. It finds circuits that already exist and explains them. It does not build new ones to order.

So the honest position is this: six of the seven sub-problems have serious prior work, and any implementation should lean on it heavily. The claim that survives is narrow and worth stating plainly. **Nobody has closed the loop.** Discovery, construction, validation, and integration have not been joined into a cycle that runs without a human at each step.


## The Seven Problems

### Circuit discovery

Given a model and a failure, determine which components are implicated. Automated circuit discovery exists but is expensive, and it typically answers "which parts implement this behaviour that works" rather than "which parts should implement this behaviour that does not exist yet". The negative case is much harder than the positive one.

### Circuit generation

When existing capacity is insufficient, create new capacity. The open question is representational: what is the right unit? A LoRA update, a handful of neurons, a full adapter block, a sparse set of edges? The choice constrains everything downstream, particularly validation.

### Selective learning

Largely solved. Freeze the base, train the new parameters, keep the gradient confined. The unsolved part is the decision of what to train, not the mechanics of training it.

### Validation

Determine whether the circuit causes the behaviour rather than correlating with it. This is the crux, and it gets its own section below.

### Integration

Connect a verified circuit so it fires when relevant and stays quiet otherwise. This is a routing problem, and it is where MoE research has the most to offer, except the router now has to accommodate experts that did not exist when it was trained.

### Stability

Guarantee that adding capability N+1 does not degrade capabilities 1 through N. The regression suite grows without bound, and every integration is a potential regression across everything the model already does.

### Dynamic growth

Decide when to add parameters rather than reuse existing ones, and when to stop. Unbounded growth is not a capability, it is a memory leak with better marketing.


## Validation Is the Real Bottleneck

Training a small circuit on a narrow task is cheap and mostly a solved engineering problem. Proving that the circuit does what you think is neither.

The temptation is to validate behaviourally: run an eval set, watch the score go up, ship it. This is not sufficient, and the reason is familiar to anyone who has debugged a flaky test. A score improving tells you something changed. It does not tell you the new circuit is responsible, that it generalises past your eval, or that the improvement is not the model getting better at your particular test rather than the underlying capability.

The stronger test is causal and comes straight from interpretability practice. Ablate the circuit and confirm the capability disappears. Patch its activations into a clean run and confirm the capability appears. If the behaviour survives removing the circuit, you did not build what you thought you built.

> **Pro tip:** if you take one design decision from this piece, make it that validation is causal, not correlational. A pipeline that integrates on eval scores alone will accumulate circuits that do nothing, and you will not find out for months.

The cost structure here is uncomfortable. Causal validation is expensive per circuit, and it has to run before every integration. It is entirely possible that validation dominates the total cost of the system, at which point the economics against full retraining get much less obvious.


## Where This Probably Breaks

**Circuits may not be separable.** Superposition means networks represent more features than they have neurons, with features sharing components. If capabilities are smeared across the same parameters rather than living in clean modules, the premise of independently addable circuits weakens considerably. This is the genome analogy failing exactly where it was most appealing.

**Integration is where interference hides.** A circuit that validates perfectly in isolation can still fire when it should not, or shift the distribution of activations feeding something unrelated. Isolated validation does not compose.

**The meta-model has to be very good.** Something must reliably characterise absent capabilities from observed failures. That is arguably a harder problem than the one being solved, and there is a real risk of an infinite regress where the meta-model needs a meta-model.

**The economics might not work.** The pitch is that this beats full retraining. If validation is costly enough, and integration regression testing grows with the number of installed circuits, that advantage could evaporate at exactly the scale where you need it.


## Closing Thoughts

I find this direction interesting precisely because it sits at an intersection rather than inside one field. Continual learning has the stability tools. Interpretability has the discovery and validation tools. Modular architectures have the routing tools. Agentic systems have the autonomy. Each field has been solving its piece, and nobody has been paid to join them into a cycle.

The most useful next step is not to build the whole loop. It is to attack the weakest link on a problem small enough to be legible: take a toy model, remove one narrow capability, and see whether an automated process can detect the absence, build a replacement, and prove causally that the replacement works. If that fails on a toy problem, the full agenda is not worth attempting. If it succeeds, every remaining problem is an engineering problem.

That is the experiment I would run first, and I would want it to fail fast if it is going to.


## Further Reading

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) (adapters)
- [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)
- [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (ROME)
- [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796) (EWC)
- [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593)
- [Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997)
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)

*This is a research direction, not a result. Nothing described here has been built or measured, and I would genuinely like to be told which part breaks first. If you are working on something adjacent, or you think the separability assumption is fatal, leave a comment.*
