# Where this probe sits — an atlas of LLM methods, the demand we answer, and the product it implies

A map for orientation (written 2026-07-02, after the sequence pivot). Goal: locate our small probe precisely
in the landscape of modern LM methods, name the literature/demand it answers, and paint the scaled-up product
it *implies* — while staying firmly on the theory / small-experiment side (no LM-scale benchmark bakeoffs).

---

## 1. The atlas — three (mostly independent) axes

Modern LM design mixes choices along three axes. Most papers move on ONE axis. Our probe is a specific point
on all three at once, and that intersection is nearly empty.

### Axis A — sequence mixing: *how do tokens exchange information?*
This is the axis everyone argues about. Ordered by the expressivity↔efficiency trade:

| family | cost | can it do content-based **selection/recall**? | maintainability |
|---|---|---|---|
| **Dense softmax attention** (GPT/Llama) | O(n²) | **yes** — this is the whole point of attention | poor: an edit touches everything |
| **Linear attention / SSM** (Mamba, RWKV, RetNet) | O(n), O(1) state | **weak** — the documented "recall gap" (Zoology/Based) | cheap *append*, all-or-nothing state |
| **Sparse / structured attention** (sliding-window, Longformer, BigBird, block-sparse) | O(n·w) | yes (it's still softmax) | local *by construction* — but suffers the reachability problem |

The tension: dense = expressive but not maintainable; linear = maintainable(-ish) but not expressive; sparse =
local (a step toward maintainable) but then **"locality does not imply reachability"** (2606.02680) — a windowed
model can't get information across the sequence without help.

### Axis B — computation depth: *how deep, and is it tied?*
| family | what it is |
|---|---|
| Feedforward stack | fixed L distinct layers (standard transformer) |
| Weight-tied / looped / Universal Transformer | one block applied repeatedly |
| **Deep Equilibrium Model (DEQ)** | weight-tied to the `L→∞` limit: solve `z = f(z)` to a **fixed point** ← *our substrate* |

A DEQ is the infinite-depth limit of a looped transformer. You don't pick a depth; you iterate a solver until
the representation stops moving.

### Axis C — the serving / maintenance layer (systems, orthogonal to A/B)
How you *keep* the computation as the input changes: KV caching, prefill, cache reuse (CacheBlend, PIE),
streaming with attention sinks (StreamingLLM). This is where "in-context editing is expensive" actually bites.

### Our point in the atlas
**Sparse (windowed) attention [A] × Equilibrium/DEQ [B], analyzed for edit-maintenance [C].**
i.e. *an equilibrium transformer with local attention, studied for how cheaply and exactly it can be maintained
under edits.* Each axis alone is populated; the **triple intersection is open** (novelty scan, 2026-07-02).

---

## 2. The demands & literature we answer

We are not inventing a need; four existing threads each leave a specific hole, and one quantity (σ_min) plugs
all four.

1. **In-context editing / KV-cache invalidation.** A mid-context edit invalidates the suffix cache; CacheBlend/PIE
   reuse a *heuristically chosen*, **lossy, uncertified** subset. → *Demand: a principled, exact, bounded
   recompute.* We answer with the σ_min-certified ξ-ball: recompute exactly that, the rest is provably unchanged.

2. **The reachability problem in sparse attention** ("Locality Does Not Imply Reachability", 2606.02680). In a
   *feedforward* block-sparse model, being inside a window doesn't guarantee information reaches you; they patch
   it with hand-placed "boundary repair" edges and give **no quantitative bound**. → *Demand: long-range from
   local structure, with a guarantee.* We answer: **equilibrium restores reachability** (the `L→∞` that closes
   the gaps) and **σ_min supplies the screening length** they lack.

3. **The recall-vs-efficiency tension** (Mamba / Zoology / Based). Linear models are efficient but can't recall;
   hybrids bolt attention back on *to recover recall*. → *Demand: maintainability WITHOUT surrendering recall.*
   We answer by detaching maintainability from linearity — see §3.

4. **NCA / self-organization.** Growing-NCA, Graph-NCA regenerate after damage but **heuristically, no reach
   theorem**. → *Demand: a certified regeneration reach.* Our edit=damage, warm-start-re-solve=regeneration, and
   σ_min = the first regeneration-reach guarantee for an NCA.

**The one idea underneath all four:** `σ_min(I − J)` is the **screening length** of an edit — how far a
perturbation propagates before it dies. For a *linear* map it reduces to Mamba's `ρ` (edit decay `= ρ(A)`); for
a *nonlinear equilibrium* it is `σ_min(I − J)`. So we **generalize the linear maintenance mechanism to the
selection-capable (softmax) regime** — the same maintenance, minus the linearity tax.

---

## 3. The one-sentence thesis (and the honest cost)

> The "maintainable class" of sequence models was assumed to require **linearity**; we show it only requires
> **good conditioning** of `(I − J)`, so a nonlinear, recall-capable equilibrium can be a member of it — paying
> in solve iterations, not in expressivity.

Honest cost: our maintenance is **not** O(1) like Mamba's; we re-solve the ξ-ball. Defensible comparisons:
- vs **Mamba**: same maintainable class, but we can *select* (it can't); and Mamba's cheap update is *append*-only
  (its state reuse is all-or-nothing — a mid-context edit is actually worse than a KV cache).
- vs **dense transformer**: we can select *and* our edit is σ_min-local (a bounded ball) instead of full-suffix.

We never claim "cheaper locality than Mamba."

---

## 4. The scaled-up product it implies (motivation, NOT a deliverable)

Imagine the mechanism scaled into a real system — this is the story that *motivates* the paper; we do not build it.

**Product: a self-maintaining context for an edit-heavy assistant** (IDE code model, agent scratchpad, RAG worker).
The context is a *living document* that is edited constantly. Instead of re-running the model over the whole file
(or the whole suffix after the cursor) on every keystroke/edit, the system:

1. holds an **equilibrium representation** of the document;
2. on an edit, computes the **σ_min-certified ξ-ball** of positions the edit can affect (a bounded local
   neighborhood, walked along the actual attention-support graph);
3. **re-solves only that ball** from a warm start (the old equilibrium); everything outside is *provably*
   unchanged, so it stays cached;
4. → edit latency scales with the **local edit size**, not the document length.

The **certificate is the product feature.** CacheBlend/PIE *hope* the un-recomputed part is fine (lossy); we can
*guarantee* it (exact, up to the σ_min bound). That matters where a silent cache-corruption is unacceptable —
code, legal, medical, agent state.

Positioning against what exists:
- **StreamingLLM** (sink + window): append-only, discards the middle.
- **CacheBlend / PIE**: recompute a heuristic subset, lossy/uncertified.
- **This**: recompute the **certified-exact ball**; the rest is provably unchanged.

Honest scope for the product too: it's **not** a general decode speedup (append is already optimal via KV cache; a
DEQ adds per-token solve overhead and stores O(n·d) state at inference). The win is the **edit-heavy / local-readout**
regime, and — via the multi-scale/coarse-node arm (C4) — long-range-relevant edits reach the far end in O(log n)
instead of full recompute. CALIBRATION (2026-07-08): only the *qualitative* characterization (edits act on a
causal downstream cone + low-rank long-range highways) transfers to an ordinary transformer, and only as a
gradient-expensive diagnostic. The *cheap certificate* does not: it needs the fixed-point residual `‖f(z)−z‖`
as a self-check (a feedforward stack has none) and a well-defined `σ_min(I−J)` resolvent (no fixed point → no
resolvent). The equilibrium is what makes it a theorem, not incidental packaging.

---

## 5. Why we can stay on the theory / small-experiment side (and must)

The contribution is a **mechanism + a characterization**, both provable at toy scale. Nothing here needs an
LM-scale benchmark:

- **C1 (reach)** — done: sliding-window equilibrium relays recall past any matched finite unroll (MQAR, toy).
- **C2 (maintainability)** — the theorem: an edit's `|Δz|` decays with distance, screening length `ξ` set by
  σ_min, `ξ ≤ Faber bound`. This is the load-bearing result and it is a *measurement against a bound*, not a
  bakeoff.
- **C3 (tradeoff)** — window `w` dials solve-iterations vs edit-reach: a Pareto curve.
- **C4 (multi-scale)** — O(log n) coarse nodes carry a long-range edit to the end; matches the empirical
  attention-sink structure of real models.

**What we deliberately DON'T do** (and shouldn't, on our compute and for our venue): no WikiText/perplexity
bakeoffs, no scaling laws, no "beat Mamba on Long-Range-Arena." Those would be a *different, systems* paper. Our
claims are isolated on the **minimal task that exposes the mechanism** (MQAR), where the σ_min theorem can be
measured cleanly. Venue fit: **TMLR** (rigor/characterization, no novelty/SOTA gate); LoG secondary. See
`project_venue_strategy`.

The theory *is* the paper: the σ_min screening-length certificate, its identity as the nonlinear generalization
of ρ, the reach theorem, and the reachability gap it closes over 2606.02680. The experiments are minimal
illustrations that the theorem holds — not a leaderboard.

---

## 6. On-ramps to code this yourself (the parts that are *your* strength)

The transformer-engineering surface is the intimidating part, but the load-bearing pieces of this project are
**small, self-contained, and math-side** — exactly where your footing is strongest. Good pieces to own:

- **`spectrum()`** (ρ, σ_min via the dense Jacobian) — pure linear algebra, *is* the theory. Natural extension:
  compute the full Faber/Chebyshev bound and overlay it on the measured decay (that's half of C2).
- **`gen_mqar`** — the task generator; fully comprehensible, no transformer magic.
- **The C2 edit-locality measurement** — perturb one value token, measure `|Δz|` vs sequence distance, fit `ξ`.
  Small, deeply theory-connected, and the heart of the paper. A great one to write end-to-end yourself; I can
  hand you a scaffold with the core left blank.
- **The plots** — reach curve, decay-vs-distance, the Pareto curve. You own the narrative visualization.

The heavy/annoying parts (the DEQ solver, IFT gradients, contraction control, speed tuning) are the wheels we
correctly don't reinvent (torchdeq) — leave those to the agent; spend your on-the-ground time on the math-side
pieces above, which is what Geng's advice is really pointing at.
