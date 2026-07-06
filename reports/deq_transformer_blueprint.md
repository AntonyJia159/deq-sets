# DEQ-Transformer edit-locality — experiment blueprint & honest scoping

## Working title & abstract (2026-07-03 draft)
**Title (lead):** *Conditioning, Not Contraction: Certifying Local Edits in Deep Equilibrium Transformers.*
Alts: *Editing Needs an Equilibrium: A Conditioning Certificate for Local In-Context Maintenance* /
*Characterizing Edit-Locality in Deep Equilibrium Transformers* (TMLR-neutral).

**Abstract (draft).** Mid-context edits are the operation modern sequence models handle worst: a dense
transformer must invalidate its entire suffix cache, and linear-state models (Mamba/SSMs) are worse still,
admitting only all-or-nothing reuse of their fused state. We ask when an edit's influence is provably
*local* — confined to a bounded neighborhood, so the remainder is reused exactly — and show the governing
quantity is the **conditioning** of the fixed-point map, σ_min(I−J), not its contraction rate ρ. This
certificate is intrinsically an **equilibrium** object: (I−J)⁻¹ is the linearization of z=f(z), with no
feedforward analogue — and in the peaked, recall-capable ρ>1 regime, the feedforward Neumann truncation
Σ Jᵏ *diverges* while the resolvent still exists and edits stay local. The maintenance operation is
correspondingly equilibrium-only: a **warm-start local re-solve** from the previous fixed point, exact under
a σ_min uniqueness condition and reach-adaptive in cost. The certificate splits by attention direction into
two classical regimes — a **causal** face governed by a product–Lyapunov (transfer-operator) rate whose linear
special case is the transition-matrix product of a linear recurrence (the object underlying linear-state models
such as SSMs), and a **bidirectional** face governed by
Faber/Demko–Moss–Smith resolvent decay — unifying the maintenance mechanisms of graph propagation, SSMs, and
equilibrium transformers under one invariant, and **detaching editability from the linearity incumbents
require to obtain it**. On associative-recall probes, sliding-window equilibria relay recall beyond any
matched finite unroll; edit-response decays with distance within the σ_min certificate across a conditioning
sweep, staying local even at ρ>1; and warm-start re-solve is exact and adaptive. We present this as a
*characterization* — the equilibrium is where edit-reach is a theorem and upper-bounds any finite transformer
— with in-context editing (code, RAG, agent state) as the motivating application.
*(Gap: the bidirectional-face experiments the abstract headlines are pending; done = C1 reach, causal-face
C2, full theory + lit review.)*

---

Design doc for the sequence direction (Geng meeting 2026-07-02). Substrate is validated: on a first
probe (`seq_recall_probe.py`) a softmax-attention DEQ solves MQAR associative recall (acc 1.00) that a
linear-attention / SSM cell cannot (0.25), and its edits are warm-start-exact — but **edit-locality is
over the attention graph, not sequence distance**. So sparse / sliding-window attention is the
load-bearing choice: it makes the attention graph metric-local, so the edit-response resolvent decays
with distance (Demko–Moss–Smith / Faber), giving a *certified* reach.

`Mamba : us  ::  InstantGNN : us` — the linear incumbent gets cheap incrementality from linearity; we
detach it (get it from σ_min conditioning) and buy a more expressive nonlinear member of the same
maintainable class.

### The thesis in one table — detaching maintainability from linearity
The maintainable class was *assumed* to require linearity. It doesn't; it requires good conditioning.

|                              | selection / recall (expressive) | cheap edit-maintenance          |
|------------------------------|:-------------------------------:|:-------------------------------:|
| Linear SSM / Mamba           | ✗                               | ✓ (O(1), from **linearity**)    |
| Dense softmax transformer    | ✓                               | ✗ (mid-context edit ⇒ full-suffix invalidation) |
| **Sparse softmax DEQ (us)**  | ✓                               | ✓* (σ_min-local, from **sparsity**) |

Only the last row has both. Cheap locality is *not* the prize — Mamba already has it and hasn't displaced
attention, precisely because people reach for a transformer for the **recall/selection** the linear model
gave up to get that locality (documented: the MQAR gap in Zoology/Based; hybrids Based/Griffin/Jamba add
attention back *to recover recall*). We are not re-selling locality; we offer **maintainability without
surrendering recall** — the combination neither incumbent has.

**Sharper still — Mamba's ✓ is *append*-cheap, not *edit*-cheap.** SSM-state reuse is documented as
**all-or-nothing**: a fused recurrent state can be reused only if the *entire* prefix matches; it supports
no partial / incremental / segmented reuse. So Mamba is cheap for **append** (roll the state forward) but a
**mid-context edit is *worse* than a transformer** — it invalidates the single fused state and forces a
full-suffix recompute with no per-position granularity to fall back on (a KV cache at least keeps per-token
K,V). So the **edit** sub-regime we target is *open for both incumbents*, not a crowded lane. (Mamba's
constant-state locality is nonetheless commercially hot: Mamba-3 @ ICLR'26, NVIDIA Nemotron-3 hybrids,
Amazon Mamba2-primed hybrids, SGLang/vLLM hybrid paging — but the applications are *streaming/append*
real-time inference, not mid-context editing.)

**Why σ_min is literally the generalization of Mamba's ρ (the theorem, not the analogy).** For a *linear*
map `A`, an edit propagates as `Σ_k A^k = (I−A)⁻¹`, spatial decay set by `ρ(A)` — that *is* Mamba's
cheap-locality mechanism. For a *nonlinear equilibrium* `f`, the edit propagates through the fixed-point
Jacobian: `(I−J)⁻¹`, decay set by **σ_min(I−J)**; when `f` is linear, `J=A` and σ_min reduces to the ρ
story. So σ_min(I−J) is the maintainability mechanism for **any** map, and Mamba's ρ is its linear special
case. We didn't invent a new maintenance; we generalized the linear one to cover the selection-capable
regime — *that* is "detach maintainability from linearity."

**One number, two readings.** In *any* linear recurrence, driving the transition spectrum toward the unit
circle (λ→1) lengthens the memory horizon `1/(1−ρ)`. Read *backward*, the identical number is the
**screening length of a perturbation**: an edit's influence survives exactly as far as memory does —
**long memory ≡ far-reaching edits; remembering is the inability to locally forget a change.** The
state-space literature notes the *forward* half (the memory kernel controls input-perturbation response
amplitude; Cirone et al 2024); the *maintenance* half is unpriced, for a structural reason: a fused
recurrent state has no mid-context edit operation to price — you need a per-position state you can re-solve
locally, the sparse-attention equilibrium. Paper line: *"the rate that sets a linear recurrence's memory
horizon is the rate that sets its edit-reach — one quantity, two directions; maintenance is the direction
never priced."* This duality is the σ_min screening length in the linear special case. (Keep this framing at
the level of a generic linear recurrence — do NOT invoke the discretization / selection-as-Δ / HiPPO lineage
of any specific model; that apparatus is orthogonal to edit-locality and only blurs the story.)

**The honest cost (`*`).** Our maintenance is not O(1) like a linear recurrence's — we re-solve the
equilibrium in the σ_min-certified ξ-ball. Defensible claims only: vs a **linear-state model (SSM)**, same
maintainable class but we can *select* (cost = solve iterations, not O(1); and their fused state is
all-or-nothing, so a mid-context edit is worse for them); vs the **dense transformer**, we can select *and*
the edit is σ_min-local (bounded ξ-ball) instead of a full-suffix recompute. Never "cheaper locality than an
SSM."

### The unification we contribute — one face is linear (ρ), the other is nonlinear (σ_min)
The right diagram is NOT "graph propagation ≈ sequence propagation" — that linear equivalence is being
worked *right now* (Message-Passing State-Space Models 2505.18728; Message-Passing→Linearized Graph
Sequence Models 2605.12358; GNN-as-graph-resolvent `(I−αÃ)⁻¹`, 2101.11859). **We cite that as the *linear
face*, we do not claim it.** The unification we add is one axis up:

> **{ linear graph propagation, linear sequence propagation } = ONE face** — the `ρ` / linear-resolvent
> special case, same theorem on two topologies (chain vs general graph; Demko–Moss–Smith is topology-
> agnostic). **{ nonlinear DEQ } = the GENERAL face** — `σ_min(I−J)`, which *reduces* to `1−ρ` in the linear
> limit and *dissociates* from ρ in the nonlinear one (ρ>1 yet edit-local).

What's genuinely unclaimed (novelty scan 2026-07-02) is the pair: **(a)** reading the resolvent as *certified
edit-locality via σ_min conditioning* — not an *imposed* decay mask (RetNet/KMS `γ^|i−j|`, Mamba-2 SSD
off-diagonal decay are decay *put in*), and not *forward* propagation (the GNN resolvent) — but decay
*derived* as a reach guarantee; **(b)** doing it on a nonlinear equilibrium where ρ and σ_min come apart.
Linear models cannot produce (b) and no one has measured σ_min-screening on a nonlinear DEQ. Cautions so we
don't overclaim: do **not** claim first to notice graph≈sequence (active), nor first to use the resolvent in
SSMs (Mamba-2 SSD writes the decay matrix explicitly), nor first to control a DEQ's Jacobian spectrum (Bai
2106.14342 — but that's ρ-for-stability, not σ_min-for-edit-reach). The sliver is the σ_min *edit-locality*
reading on the *nonlinear* equilibrium.

### Identity: this is an attention-based NCA with *certified* regeneration
Exact correspondence, not metaphor: local sliding-window attention = NCA local update rule; equilibrium
= NCA `t→∞`; **edit → warm-start local re-solve = damage → regeneration**; bidirectional window = the
spatial, no-time-arrow regime of regeneration. So the maintenance model *is* an attention-based Graph-NCA
on the attention graph. Novel identity + community (self-organization / NCA): Growing-NCA (Mordvintsev),
Graph-NCA (Grattarola), and ZJ's own NCA↔DEQ equivalence (self-cite, 2501.03573) regenerate *heuristically*
with no reach theorem — our σ_min contribution is **the first regeneration-reach guarantee for an NCA**.
Preferred framing: *"a self-organizing attention field that provably regenerates locally after edits."*

### Margolus / staggered-block correspondence (ZJ, 2026-07-03) — a note, not a claim
**Block** attention (non-overlapping partition) = a Margolus-neighborhood block CA (Toffoli–Margolus 1987):
a single block layer can't cross a block seam, and **two interleaved layers staggered by half a block** =
the Margolus double-step that mixes across seams (one block/two layers). Payoffs: (i) this reframes
2606.02680's ad-hoc **"boundary repair"** (hand-added seam edges) as a clumsy rediscovery of the Margolus
stagger — and **the equilibrium subsumes both** (the fixed point crosses seams regardless of partition:
converge, don't repair). (ii) It slots our two sparsity choices onto two CA conventions: **sliding-window ↔
overlapping-neighborhood CA** (overlap buys mixing, costs redundancy) vs **staggered-block ↔ Margolus**
(no overlap, mixing deferred to the alternation, cheaper). (iii) DEQ realization: make the cell the
composition `f = (partition B) ∘ (partition A)` so one iteration = one Margolus double-step; the **σ_min
certificate is unchanged** — `(I−J)` just has a staggered-block sparsity instead of banded (topology-agnostic
DMS). **Caveat (don't overreach):** Margolus is prized for *reversible* CA (bijective block rule); that half
does **not** transfer — our edit-locality is from σ_min *conditioning*, not reversibility. Cite the
neighborhood/mixing structure only. Scope: a *remark* + possible future-work arm (staggered-block DEQ), NOT
a plot; experiments stay sliding-window.

### The four lands — one operator, four projections (a framing figure, not a claim of unification)
The central object is **a local operator on a graph, iterated/inverted to a fixed point, whose edit-response
is the resolvent `(I−J)⁻¹` with decay governed by `σ_min(I−J)`.** Each "land" is a *projection* of it:

| land | substrate | topology | linear? | depth | edit-decay governed by |
|---|---|---|---|---|---|
| **Graph** (GNN / InstantGNN) | message passing | general graph | linear (incumbents) | iterated → fixed | `ρ(A)` — graph resolvent `(I−αÃ)⁻¹` |
| **SSM** (Mamba / S4) | linear recurrence | 1-D chain | linear | scan (unrolled ∞) | `ρ(A)` = `1−σ_min` on a chain |
| **Transformer** | attention | attention graph | nonlinear | finite `L` | truncated Neumann `Σ_{k≤L} Jᵏ` |
| **NCA** (Neural CA) | local rule | lattice / graph | nonlinear | `t→∞` | fixed point — *heuristic, no bound* |
| **Ours** | sparse-attn **equilibrium** | attention graph | **nonlinear** | **∞ (DEQ)** | **`σ_min(I−J)` — certified** |

Read the map as: **linearize** → Graph/SSM (`ρ`); **truncate depth** → Transformer (Neumann); **drop the
certificate** → NCA (heuristic regeneration). The edges between lands are the known correspondences we cite,
not invent: SSM↔Graph (chain = 1-D graph; the linear unifiers 2505.18728 / 2605.12358 / 2101.11859),
Transformer↔SSM (Mamba-2 state-space duality: linear attention ≡ SSM), Transformer↔Graph (attention *is* a
directed graph), Transformer↔NCA (our iterated-attention identity), Graph↔NCA (Graph-NCA, Grattarola). The
one **empty cell everything points at** — nonlinear + equilibrium + *certified* — is ours, and `σ_min(I−J)`
is the master invariant that lives at the center and reduces to `1−ρ` on every linear edge. Honest scope: the
figure is **exposition**; the linear unification among the outer lands is prior art. Our contribution is the
center cell, not the map.

### Two regimes, two attention directions
- **Decode / generation → CAUSAL** window (`i` attends `[i−w, i]`): relay is forward-only; this is the
  C1 expressivity test.
- **Edit / maintenance → BIDIRECTIONAL** window (`i` attends `[i−w, i+w]`): the document is fully present,
  you edit and re-settle in both directions. This matches code (defs after uses, edits hit callers above
  and below) and RAG chunk-swaps, and it is the NCA damage-regeneration regime. C2/C4 use this.

### Applicability of the (I−J) theory to ORDINARY transformers
`(I−J)δz=δf` needs a fixed point, so the resolvent-decay reach is *exact only for equilibrium models*. But
a feedforward L-layer transformer propagates an edit as a product of L layer-Jacobians = the first L terms
of the Neumann series `(I−J)⁻¹=Σ_k J^k` (paths of length ≤ L). So a finite transformer is a **truncation**
of the resolvent, the DEQ is the `L→∞` limit, and our reach **upper-bounds a finite transformer's reach**
too (conservative, sound). Practical insight transfers to any transformer via attention-reachability
(combinatorial support) + σ_min (quantitative decay). DEQ = the clean setting where reach is a theorem.

### Why DEQ is load-bearing, not incidental — the resolvent exists only at equilibrium (sharpened)
The certificate `σ_min(I−J)` and the resolvent `(I−J)⁻¹` are *the linearization of a fixed-point equation* —
they exist **only** because you perturbed `z=f(z)`. A feedforward net has no `z=f(z)`, hence no `(I−J)` to
invert; it has only the **truncation** `Σ_{k≤L} Jᵏ`. The decisive case is **ρ>1** (which peaked, recall-capable
attention drives): there the Neumann truncation *diverges* — the feedforward/unroll picture doesn't just
approximate poorly, it **breaks** — yet `(I−J)⁻¹` still exists (σ_min>0) and the edit is still local. So in
exactly the regime that makes the model expressive, the maintenance object is **well-defined only at
equilibrium and computable only by solving (Anderson/Broyden), not by any finite unroll**. This is the sharp
answer to "why not a deep feedforward transformer, or Mamba?":

| model | the object | what it is |
|---|---|---|
| feedforward transformer | `Σ_{k≤L} Jᵏ` | truncation — no inverse; **diverges at ρ>1** |
| Mamba / causal | `(I−N)⁻¹`, `N` nilpotent | *one-shot* resolvent = the forward scan → product-Lyapunov |
| **bidirectional DEQ (ours)** | `(I−J)⁻¹`, `J` two-sided | **genuine iterative resolvent** → σ_min/Faber |

**Mamba is not an equilibrium — it is a *degenerate (nilpotent) one*.** Its scan `h_t=A_th_{t-1}+B_tx_t` is
`(I−N)⁻¹h` with `N` strictly-lower (nilpotent), so the "solve" terminates in one forward sweep — no iteration,
no contraction condition. That is the causal special case, not a bidirectional equilibrium. Editing means
re-settling in **both** directions (a change hits readers below *and* callers above), which needs the genuine
two-sided resolvent; the feedforward truncation and Mamba's one-directional nilpotent version are its two
*shadows* that cannot. (Even the causal resolvent hides a per-token self-consistency solve inside each
`(I−D_i)⁻¹` — a token's own equilibrium — that a fixed-depth causal net never performs.) One-line thesis:
**the maintenance certificate is a property of the fixed-point resolvent; genuine two-sided editing requires
the full equilibrium, and in the ρ>1 regime nothing else even yields a finite object.**

### Gradient mode: phantom is a *truncation*, not a failure — and it mirrors the reach cliff
The stability probe found phantom (1-step) gradient caps MQAR recall at ~0.45 while exact IFT hits 1.00.
This is **not** "phantom can't train peaked transformers" — phantom (Geng et al. 2021) and the original
DEQ-Transformer (Bai et al. 2019) train fine. Phantom is a **truncated-Neumann** approximation of the
adjoint `(I−J)⁻¹`, and MQAR's retrieval signal lives in the *deep* Neumann terms (the multi-hop relay), so
a 1-step phantom under-credits exactly the peaking pathway. It is the **backward/adjoint mirror of the
forward finite-unroll reach cliff**: *tasks where equilibrium beats unroll (forward) are the same tasks
where IFT beats phantom (backward)* — one criterion, two sides of the map. Well-mixed natural language is
fine with phantom; a sharp relay (MQAR) needs IFT. **Use `ift=True` + Anderson for the recall-critical
runs**; `grad=5` blowing up (ρ→8) is undamped tuning (τ<1 unset), not fundamental.

### Prior art we BUILD ON (do not claim these as ours)
The bidirectional/infilling machinery already exists and speaking the field's language is an asset (Geng's
"worth the exposure"), so we cite it and sit on top:
- **Infilling / dual-mode LMs.** FIM (Bavarian et al. 2022) reorders `[prefix][suffix][middle]` to get
  tri-source ("ante / post / edited region") conditioning *cheaply* with a plain causal model and no
  after-`t` positional embeddings — shipped in StarCoder/CodeLlama/DeepSeek-Coder. Prefix-LM / UniLM
  (Dong et al. 2019, switchable masks), GLM (Du et al. 2022, AR blank-infill), XLNet, and diffusion LMs
  (SEDD, LLaDA) generate bidirectionally / any-order. A **bidirectional-window DEQ is an infilling model** —
  the NCA regeneration regime restated. **Unclaimed by all of them: a *certified* reach for the re-settle.**
  Novelty stays pinned on the σ_min certificate, not on the bidirectionality.
- **Incremental / self-adjusting computation** (Acar) and InstantGNN's affected-subgraph propagation — the
  substrate for the support-graph re-solve below.
- **Attention-as-a-directed-graph** is an established lens we adopt, not invent: an attention mask = a
  directed information-flow graph over positions (edge src→tgt = tgt reads src in one layer); stacking
  layers gives a **reachability closure** `R_ℓ(t)` (positions reachable by walking edges backward).
  *Attention Flows* (2009.07053), *Lost in Transmission* (2505.08140), FlowTracer (2606.10646). Empirical
  properties of these graphs, all relevant to us: (i) **>90% sparse**, head-specific/content-adaptive
  (MInference 2407.02490, SampleAttention 2406.15486) → the realized support graph is far sparser than the
  window (helps the support-graph re-solve); (ii) **not purely metric-local — O(1) global hubs = attention
  sinks**, provably necessary for some tasks (2603.11487; "Spike/Sparse/Sink" 2603.05498) → real attention
  is *local structure + bounded locality-breaking hubs*, which is **exactly the C4 multi-scale design** (so
  C4 matches how attention is already shaped, not an add-on); (iii) **induction heads** (prefix-match+copy)
  are the documented long-range motif our MQAR relay rides; (iv) caveat: **rank collapse** thins the
  effective graph in deep layers.
- **C1's closest precedent — "Locality Does Not Imply Reachability" (2606.02680).** Feedforward
  block-sparse causal attention: being *inside* a local window does NOT guarantee information reaches you
  (block-boundary bottlenecks), which they fix with ad-hoc "**boundary repair**" (hand-added edges at block
  edges). This is C1 stated as a problem, in a finite transformer. **Our two-part gap over it:** (a)
  **equilibrium restores reachability** without hand-placed boundary edges (the K→∞ that closes the gaps a
  finite block-sparse stack leaves open), and (b) they give **no quantitative decay bound** — σ_min/Faber
  supplies the *screening length* they lack. Must-read-in-full before writing C1; cite as the precedent
  that makes the reach question legible to reviewers.

---

## Positioning — what we upgrade over KV-cache incumbents

- **Standard causal transformer.** Append is exact & cheap (causal immutability of past K,V). A
  **mid-context edit** invalidates the entire suffix cache (every later token attended to the edit).
- **CacheBlend / PIE.** Reuse the cache, recompute a *heuristically chosen* subset of tokens.
  **Lossy and uncertified.**
- **Us.** The Faber/DMS screening length is a **theorem** — a rigorous **upper bound** on which
  positions an edit can affect. Recompute exactly the certified ξ-ball, *exactly*; the rest is provably
  unchanged. **Heuristic → theorem; lossy → exact.**

### Why a *loose* bound is the *right* guarantee
For maintenance you want a **sound (conservative)** bound, not a tight one. Recomputing a *superset* of
the affected positions is **correct** (just wasteful); *under*-recomputing (CacheBlend's risk) is
**wrong**. The Faber bound is empirically loose — actual reach ≪ bound, especially near singularity
(`sigma_min_law`: near-singular ξ_pred 3.4–4.0 vs ξ_meas 0.7–0.9; worst-case ≠ typical) — but its
looseness costs **compute, never correctness**. And its tightness is governed by κ: **tight when
well-conditioned** (r≈0.9) and conservative near singularity. So keeping κ small makes the certificate
both exact and efficient. One-line pitch: **replace an uncertified lossy heuristic with a sound, exact,
conditioning-tightened certificate of edit reach.**

---

## Two efficiency components (certify with σ_min, execute on the support graph)

**Warm start (state the mechanism precisely).** Seed the solver at the *old* equilibrium `z*`. An edit is a
local perturbation to `h0`; Anderson/Broyden from `z*` converges in `O(ξ)` iterations, and the residual only
lifts *inside* the ξ-ball — outside it `f(z*)≈z*`, so those coordinates arrive already-converged and cost
nothing. That is the concrete "downstream tokens are not recomputed from scratch."

**Warm start's claim-status (characterization vs proposal — keep this line bright).** Warm-start is the one
*operational* channel that is genuinely equilibrium-exclusive: a feedforward net has no "resume" — after an
edit the affected positions pay all `L` layers again, a **fixed** cost regardless of how small the edit's
effect is, and its recompute set is an `L`-layer light cone. The equilibrium object instead has (1)
**adaptive cost** — iterations ∝ how far the solution moved (∝ ξ), not ∝ depth; (2) **exactness** —
path-independence, warm==cold to ~1e-7 (measured), where feedforward partial-recompute heuristics are lossy;
(3) **depth-independent reach** — the ball is ξ (conditioning), not `L`. HOW we report it decides which
paper we're writing: we measure **solver iterations vs edit distance + the warm/cold iteration ratio**
(architecture-internal property measurement → characterization); we never report wall-clock against
optimized serving stacks (→ systems proposal, owing benchmarks we don't run). Paper sentence: *"the object
we characterize additionally possesses an exact, reach-adaptive maintenance channel unavailable to any
finite feedforward network; whether this makes equilibrium LMs practical at scale is a systems question we
do not answer."* Counterweights stay attached: warm-start needs the O(n·d) equilibrium state stored (≈ a KV
cache — no memory win), and per-token decode overhead remains; the channel is for **edits**, not generation.

**Support-graph incremental re-solve (promoted from "later" to a named component).** The support of the
attention matrix (nonzero-weight indices) *is* a sparse dependency graph. Re-solving only over the region
reachable from the edit along that graph, freezing the rest, is **self-adjusting computation** (Acar) applied
to a fixed point — the sequence analog of InstantGNN's affected-subgraph propagation. This is **complementary
to σ_min, not redundant**: σ_min gives the *a-priori worst-case* ξ-ball (lets you budget/certify before
touching anything); the realized support is the *actual* frontier you propagate along (usually ≪ the ball,
matching the empirically-loose-bound finding). **Certify with σ_min; execute on the support graph.** Because
the support is data-dependent it is typically far sparser than a fixed sliding window.

**Positional edits — relative PE is *necessary but not sufficient*; sparsity is load-bearing.** Absolute PE:
an insert shifts every downstream position → global invalidation. Relative PE (RoPE) only shifts the offset
of each **straddling** edge (`i<p<j` for an insert at `p`); with *dense* attention there are `O(n)` straddling
long edges (begin↔end) → still effectively global. Only **sparsity** localizes it: a window `w` confines
straddling edges to a **width-`w` band** around the cut (far regions shift uniformly, so their attention is
byte-identical). So an insertion has **three regimes**: (i) within `w` of the cut — direct perturbation,
recompute the band; (ii) begin↔end of a block wider than `w` — perturbed only *through the equilibrium relay*,
σ_min-decayed within ξ; (iii) a truly distant generation point — `O(distance)` fundamental, amortized only by
the C4 multi-scale coarse channel. Relative PE handles the uniform-shift bookkeeping, sparsity confines the
direct hit, σ_min + multiscale handle propagation. **Make relative PE the default for the insert/delete (v2)
story** — but the localization comes from sparsity, not from the PE choice.

---

## Honest scope — what this is NOT (do not overclaim)

- **NOT a decode/generation speedup.** The causal KV cache is already optimal for append; a DEQ
  re-solves per token (overhead), and at inference it stores O(n·d) equilibrium state ≈ a cache — the
  famous "O(1) memory" is a **training** property, not an inference one. **Drop any decode-speed claim.**
- **NOT cheap for long-range-relevant edits in a pure sliding-window model.** An edit that must reach a
  far generation point costs O(distance) — fundamental (information must travel), not fixable by any
  method. Partially resolved by the multi-scale arm (C4).
- **The genuine regime** = edit-heavy / local-readout: code editing (re-predict near the cursor — PIE's
  setting), agent scratchpad revision, RAG chunk swaps re-read locally. The *characterization* (recompute
  exactly the certified ξ-ball) also transfers to standard transformers; the DEQ is the clean setting
  where it is provable.

---

## Claims to test

- **C1 (reach).** A sliding-window softmax DEQ solves **cross-window** recall (gap `G ≫ w`) via
  equilibrium propagation, *beyond* a matched `K`-step unroll of the same cell (reach capped at `K·w`).
- **C2 (maintainability).** A value edit is warm-start-exact (warm==cold); `|Δz|` **decays with sequence
  distance** (metric-local, unlike dense attention), screening length `ξ` set by conditioning;
  `ξ ≤ Faber bound` (sound), tight when well-conditioned.
- **C3 (tradeoff).** Window `w` dials solve-iterations (mixing) vs edit-reach `ξ` (locality) — a Pareto
  curve. (Small `w`: slow solve, local edits. Large `w`: fast solve, global edits.)
- **C4 (multi-scale resolution).** Adding `O(log n)` coarse / global nodes lets a *long-range-relevant*
  edit reach the generation point in `O(log n)` via the coarse channel (local ball + `O(log n)` coarse
  updates) instead of full-suffix recompute — at the cost of the coarse nodes being bounded (`O(log n)`)
  locality-breaking hubs. This is the concrete answer to "how does the signal reach the end without a
  whole recomputation."

---

## Architecture — the DEQ cell

Input injection (graph-independent): `h0 = Emb(tokens) + PosAbs`. **Absolute** positions for v1
(substitution edits don't shift positions); RoPE / relative is the insert/delete (PIE-regime) follow-up.

Equilibrium cell `z ← f(z)`:
```
q_i, k_i = Wq z_i, Wk z_i          # RAW — must PEAK for retrieval (spectral-norming q/k kills recall)
v_i      = Wv_n z_i                 # Wv spectral-normed (bounds the map -> contraction)
mask     = causal sliding window: i attends to [i-w, i]      # banded (dense attn + banded mask, L<=256)
a_ij     = softmax_j(q_i·k_j/sqrt(d)) over window            # linear variant: elu-kernel, no softmax
agg_i    = sum_j a_ij v_j
z_i'     = h0_i + s · Wo_n agg_i    # Wo spectral-normed, s = s_max*sigmoid(.), capped
```
- **Multi-head H = 2–4** (cross-window relay likely needs a "carry" head + a "read" head).
- **Contraction control:** bounded window degree (≤ w, convex weights) + SN(Wv, Wo) + s_max cap +
  non-finite-step guard + Anderson solve. **Monitor ρ(J) and σ_min(I−J)** — the peaking↔contraction
  tension is the likeliest failure at longer relays, so watch it.
- **Spectral-norm hoisted once per solve** (the 5× bug from the MQAR probe).
- **Two solvers:** equilibrium (Anderson to tol) and finite `K`-step unroll of the same cell (the C1
  control).
- **Multi-scale variant (C4):** add `M` global nodes attended by all and attending all (sink-style), or
  a log-dilated skeleton, on top of the window.

---

## Task — controlled-gap MQAR

```
[ k* v* ][ G distractors: other (k,v) pairs + filler ][ q* ]   ->  predict v*
```
- Distractor pairs force *selection* (peaking); the gap `G` forces a *relay* of ⌈G/w⌉ windows through
  the equilibrium. Sweep `G` vs `w`.
- Vectorized generator (no per-example Python loops — the CPU-bound bug from the probe). Disjoint id
  ranges for keys / values / filler; CE at query positions only.

---

## Sweep & deliverables

- **Plot 1 (C1):** recall vs `G/w`, curves {eq-softmax, `K`-unroll K=2/4/8, eq-linear}. Unroll should
  cliff at `G≈K·w`; equilibrium extends past it; linear fails throughout. Crossover = where equilibrium
  earns expressivity (nothing for `G<w`, matching the graph null; real reach for `G≫w`).
- **Plot 2 (C2):** `|Δz|` vs forward sequence distance from a value edit → fit `ξ`; overlay the Faber
  bound; vary `w`. Expect metric-local decay (unlike dense), `ξ ≤ bound`, `ξ` growing with `w`.
- **Plot 3 (C3):** solve-iterations vs `ξ` across `w` — the mixing↔locality Pareto curve.
- **Plot 4 (C4):** #positions that must be recomputed to make a long-range-relevant edit reach the end,
  **with vs without** the coarse channel — expect O(suffix) → O(local + log n).

---

## Pitfalls (learned the hard way)

- q/k **raw**; only Wv/Wo normed — else both models tie at chance (the earlier 0.25/0.25 bug).
- Verify **both** compared models actually fit their achievable task (no degenerate R²≈0 "wins").
- Hoist spectral-norm out of the solve loop; vectorize data gen; guard non-finite steps; Anderson solve;
  log ρ(J), σ_min. Sliding window = dense attention + banded causal mask at L ≤ 256 (no custom kernel).

---

## Compute & decision value

Toy-scale, forward-only edit probe, minutes/condition on the RTX 4050. Build order: **Plot 1 first**
(make-or-break for "local + equilibrium = long range"); if it survives, Plots 2–4.
- **C1+C2+C3(+C4) hold** → expressive + long-range-via-equilibrium + maintainable-with-a-certified-dial
  → the paper spine.
- **Equilibrium ≯ unroll on the relay** → equilibrium buys nothing here either → fall back to
  maintenance/characterization framing only.
- **Sliding-window can't relay recall** → local+equilibrium hope fails → pivot to AI4Science.
