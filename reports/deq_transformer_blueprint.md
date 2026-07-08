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

**The KV-cache interface (what our object *is*, in serving terms).** The cached object is not the embeddings
(`h0` is the *input* injection — caching it restarts the solve from scratch); it is the **equilibrium state
`z*` itself** (or its projections `Wk z*, Wv z*`). The dictionary, standard-transformer ↔ equilibrium:
- **Cache contents:** per-layer K,V over all past tokens, `O(L·n·d)` ↔ a single `z*`, **`O(n·d)`** — weight-tying
  collapses the depth axis, so the equilibrium cache is a factor `L` smaller.
- **Append (decode):** reuse the cache exactly ↔ *causal face* — prefix equilibria provably do not depend on
  the appended token, so append = solve one new position against a frozen cached `z*`.
- **Edit:** cache invalid downstream, reused *heuristically* and lossily (CacheBlend/PIE) ↔ cache provably valid
  **outside the ξ-ball**, re-solved inside and **warm-started from the cache itself** — the sound version of
  what CacheBlend approximates: *certified partial invalidation, the invalidation region is a theorem not a
  guess.* This is Geng's scenario B closed into a loop.
- **Dual use:** in standard practice the cache is only a speed trick; here `z*` is *one object with two reads* —
  it **is** the KV cache (decode) and it **is** the warm start (edit).
Honest counterweights unchanged: `O(n·d)` beats `O(L·n·d)` in memory but decode still pays solver iterations
per appended token (no throughput claim vs optimized serving stacks); and on the **bidirectional face there is
no free append** — a new tail token perturbs its own ξ-ball *backward*, the correct price of two-sided semantics.

**Practical bidirectional architectures use exactly our two sparsity patterns (cite, don't rebuild).** Production
bidirectional models are sparse in one of two ways, both already in this blueprint: (a) **alternating local
windows** — Swin (vision) is literally the Margolus construction (W-MSA ↔ SW-MSA shifted windows cross seams by
the stagger); ModernBERT interleaves sliding-window layers with a periodic global layer; (b) **local band +
designated global/hub tokens** — Longformer, BigBird = our C4 multiscale, and it matches the attention-sink
finding (real models grow `O(1)` hubs). The certificate is topology-agnostic, so it applies to both unchanged
(staggered blocks = a different J sparsity; hubs = C4). Notably the *infilling* workload is served today mostly
by **causal + rearrangement (FIM)** — because bidirectional training is the harder path (our six-round blocker
saga is the small-scale echo). Staggered-block substrate = good future work, **not** a second experiment for
this paper (a full pipeline rerun for a footnote); one demonstrated topology + the two measured faces suffices.

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
story** — but the localization comes from sparsity, not from the PE choice. So an insert's "exponential shadow
on both sides" is **not** a wide recompute but exactly the σ_min-screened ξ-ball (+ an O(n) index-bookkeeping
pass with zero recompute) — the certificate *prices* it rather than an uncontrolled cost — provided PE is
relative; under absolute PE the shadow really is global. **CORRECTION (measured, `posw_ablation.py`):** the
substrate is **not** relative-PE despite the rel-bias — `h0 = emb + posw[:L]` still adds a *learned absolute* PE,
and zeroing it collapses recall at gap>0 (→0.4–0.5) with `‖posw‖` *growing* with gap (2.84→11.15). **RESOLVED
by the pure-relative retrain (`curriculum_bidir_noposw.py`, bidirnp00–40):** trained *without* posw from the
start, the relay works — recall 1.000/0.987/0.912/0.937/0.819 at gaps 0–40 (vs 0.997–0.938 with posw). So
absolute PE is a **crutch, not a wall** (the ablation measured the trained-with-posw model's learned reliance,
not a class necessity). Insert/delete unblocked on the bidirnp substrate; anchor-token contingency recorded in
findings digest §11 (BVP boundary value / Haviv causal-implicit-anchor / attention-sink / NCA-seed) as an
optional booster, not needed. See findings digest §11.

---

## Honest scope — what this is NOT (do not overclaim)

- **NOT a decode/generation speedup.** The causal KV cache is already optimal for append; a DEQ
  re-solves per token (overhead), and at inference it stores O(n·d) equilibrium state ≈ a cache — the
  famous "O(1) memory" is a **training** property, not an inference one. **Drop any decode-speed claim.**
- **NOT cheap for long-range-relevant edits in a pure sliding-window model.** An edit that must reach a
  far generation point costs O(distance) — fundamental (information must travel), not fixable by any
  method. Partially resolved by the multi-scale arm (C4).
- **THE CENTRAL HONESTY — edit-locality is DUAL to forgetting, so it is self-defeating in the causal/decode
  regime; the maintenance win lives ONLY in the bidirectional local-readout niche.** The *same* σ_min gives
  the screening length ξ (edit-locality) AND the memory horizon (one-number-two-readings): small σ_min = long
  ξ = long memory; large σ_min = short ξ = short memory. Therefore **a causal LM that is doing its job (long
  memory — it carries bindings to the cursor) has a large ξ by necessity — edits are NOT local**, and a causal
  model whose edits ARE local is one that has forgotten (useless for recall). *Edit-locality in the causal
  regime is exactly the property you do not want.* The queried-value edit's must-carry to the cursor is not a
  DEQ flaw and not a win — it is the memory horizon, priced. **How the ξ-ball interfaces with must-carry:** the
  certificate is worst-case (smallest σ_min direction), so the ξ-ball **always contains** the carry direction
  → soundness holds (it correctly says "this edit can reach the cursor"). But in the useful causal regime that
  ball ≈ the whole suffix, so the certificate is **sound but VACUOUS** (no compression, "recompute everything
  downstream") for any carry-exciting edit. It only compresses for edits that miss the carry subspace (filler,
  unqueried-value) — and "cheaply maintaining the part of the context that doesn't matter" is a weak pitch in a
  generation setting. NUANCE (keeps it from being too self-flagellating): reach is **anisotropic** — the causal
  product-Lyapunov form has geometric-mean coupling ≈1 along the low-dim carry (full-suffix reach) and <1
  transverse (short reach); the scalar σ_min just reports the worst (carry) direction. But the demotion stands.
  **What the CAUSAL face is FOR in the paper:** it is the *proof ground* (product-Lyapunov / the RNN-Lyapunov
  BPTT bridge) and the regime where we *characterize the must-carry limitation* — **not** where we claim a
  maintenance proposal. Characterization, not proposal, in causal-land.
- **THE CAUSAL CLAW-BACK — not local, but adaptively priced (a 3-tier ladder; see findings digest §10).** The
  central honesty says causal edits aren't *local*; it does NOT say they aren't *adaptively priced*. (1) **Tier 1
  — directional certificate** (a-priori, certified; the recommended small experiment = the product-form debt):
  project the known-before-solving δh onto the low-rank **carry subspace** (top singular dirs of the per-hop
  transfer product) → transverse part gets a certified short-ξ⊥ bound, carry part is a rank-r long-range update;
  the vacuous scalar ball refines to *ξ⊥-ball + rank-r carry*, and low-carry edits get an a-priori containment
  verdict. (2) **Tier 2 — a-posteriori** (done): `‖z−z*‖ ≤ resid/σ_min` certifies *any* partial recompute
  (stop at `resid/σ_min<tol`). Lineage note (ZJ): this is the **conditioning upgrade of the collage/Banach
  a-posteriori bound** `‖z−z*‖ ≤ ‖f(z)−z‖/(1−c)` — tighter when contraction holds (σ_min ≥ 1−c), valid at
  ρ>1 where collage is void, at the price of locality (first-order, empirically validated). Future work:
  "conditioned collage training" — train on the certified error resid/σ_min instead of the raw collage
  residual (which certifies nothing near-singular); check novelty vs Bai's Jacobian regularization +
  phantom-gradient line with Geng. (3) **Tier 3 — emergent metering** (measured; **REVISED by
  C2m**: a *bidirectional* property — clean output-sensitive law there (Spearman ~0.9), weak/mode-confounded
  causally, absent near-singular; causal face's reliable instruments are tiers 1–2 only; the coarse 3-class
  cost ordering still holds causally as a step, not a law); feedforward/cold = input-sensitive flat toll
  (confirmed); tier 3 sits inside the tier-2 bound so the composite stays sound. **Deferred-billing reading (ZJ):** most
  edits are a quiet build-up of relationships stored locally, awaiting a future trigger; metering bills you
  **when the reader arrives and excites the carry**, not at write time — the write→trigger iter-gradient
  (single-digit→moderate→heavy) is the desired billing curve, emergent. Harness hook (one paragraph): δh known
  pre-solve + carry basis precomputed → O(r·d) cost-prediction *before* paying (small proj → patch ξ⊥-ball, keep
  cache; large → schedule full re-solve). Caveats: **first-order** certificate on a *finite* perturbation (needs
  C2-style validation, not just derivation); rank choice + carry-subspace stability across contexts = the
  attack surface.
- **The genuine regime for a maintenance WIN = bidirectional + local-readout + readers-present.** ξ-ball
  compression needs σ_min bounded away from 0 in *all* directions (no long-range carry) — i.e. the model is
  *not* a long-memory autoregressor. That is the bidirectional niche we identified: code editing (re-predict
  near the cursor — PIE's setting), agent scratchpad revision, RAG chunk swaps re-read locally — no single
  cursor everything must flow to, relevance spatially bounded, and (crucially) the readers/queries are
  **present AND ATTENDABLE in the context at solve time** so the relay can be query-aware and selectively
  forget. CORRECTION (2026-07-07): present is not enough — our trained bidir substrate keeps queries READONLY
  (context can't attend to them), so its relay *cannot* be query-aware and its measured must-carry ≈ causal
  (far/near 0.061 vs 0.068); the reader-set principle predicted this (invisible readers force carry). Whether a
  QUERY-VISIBLE substrate trains (readonly off + window curriculum, untested combination) and actually drops
  irrelevant-edit transport = the bidirqv retrain + C2t. REMAINING CAVEAT regardless: for the edit-now /
  query-later workload the future readers are unknown → a must-carry-like burden returns even with visibility.
  So the *cleanest* claimable regime is
  bidirectional local-readout where the relevant readers are already in context. CALIBRATION (2026-07-08,
  correcting a looser earlier claim): the *cheap certificate* does **not** transfer to a standard transformer.
  The deployable bound is the a-posteriori `‖z−z*‖ ≤ ‖f(z)−z‖/σ_min` — its power is that `‖f(z)−z‖` is a
  self-check computable from the candidate alone (one `f`-eval). A feedforward stack has **no fixed point**, so
  no free residual, no `(I−J)⁻¹`, no `σ_min(I−J)` — to check a partial recompute you'd have to recompute it.
  The clean single-rate `ρ(G)`/Stein form additionally needs weight-tying (autonomy) and windowed attention
  (the block-tridiagonal structure); Llama has neither (per-layer weights → non-autonomous product; full
  attention → dense resolvent, only the low-rank-carry story survives). What *does* transfer is the
  **qualitative** picture — causal downstream cone + low-rank long-range highways — measurable on any
  transformer as a (gradient-expensive) diagnostic, but not a cheap certificate. So the equilibrium is
  **load-bearing**: it is what creates the residual self-check and the well-defined resolvent. Certified KV
  maintenance genuinely pends a DEQ-style model (or DEQ-ifying the cache with a few tied refinement iters).

---

## Claims to test

- **C1 (reach).** A sliding-window softmax DEQ solves **cross-window** recall (gap `G ≫ w`) via
  equilibrium propagation, *beyond* a matched `K`-step unroll of the same cell (reach capped at `K·w`).
- **C2 (maintainability — BOTH FACES MEASURED).** A value edit is warm-start-exact (warm==cold); `|Δz|` **decays
  with sequence distance** (metric-local, unlike dense attention), screening length `ξ` set by conditioning;
  `ξ ≤ Faber bound` (sound), tight when well-conditioned.
  - *Causal face* (`c2_edit_locality.py`, curr ckpts): envelope holds on filler edits (hop-binned, 5–16×
    conservative); 3-tier taxonomy monotone; **must-carry** discovered (a causal relay can't see future
    queries → transports all bindings); warm-start exact where unique, branch-tracking where not.
  - *Bidirectional face* (`c2_bidir.py`, bidir ckpts — **the σ_min/conditioning face**; NOT the Faber-FOV
    regime, see certified-bounds note below): substrate
    trained via **window curriculum** (w=2→4→10 forms the binding hop that full-width bidirectional masks
    suppress — every cold config stuck at the one-layer ceiling 0.38; see probe logs) then gap curriculum
    with re-banded queries; recall 1.0/0.997/0.987/0.995/0.938 at gaps 0–40 with **ρ<1 throughout**
    (0.43→0.87; causal needed ρ up to 8.4) and σ_min spanning 0.246→0.016. Measured: **envelope OK** on
    filler everywhere measurable (ξ 0.29/0.41/0.51 hops vs proxy 4.6/6.8/6.3; ~10× conservative; **now
    certified** via Route A below, measured ~100× inside); ξ grows
    as σ_min falls; response genuinely **two-sided** (left-mass up to 0.44 vs 0 causal); ν(J) **orders** the
    faces but does NOT license Faber on either (ν_bidir 0.21–0.31 vs ν_causal 0.32–0.71 — less vs strongly
    non-normal, most extreme at gap 40: 0.21 vs 0.71; **0∈W(I−J) on BOTH**);
    **CERTIFIED BOUNDS (2026-07-08):** proxy √κ is a *heuristic* (Hermitian theorem misapplied). Route A
    (normality-free DMS via normal equations) *certifies*: 42–222 hops, measured ~100× inside → envelope is a
    theorem. Route B (sharp Faber-on-FOV+Crouzeix) **abstains on both faces** — +1∈W(J) everywhere (ρ≈0.74 but
    Re W(J)≈3.1, w/ρ≈4×): huge numerical range despite ρ<1, sharp rate defeated by non-normality. This
    **corrects "proper Faber regime" → "σ_min governs; Faber-FOV rate not certifiable here"**, and is a **4th
    clever prediction (Route-B splits faces) dying on measurement** while the crude κ object survives.
    **TIGHT a-priori certificate (2026-07-08, `c2_weighted_cert.py`, digest §4c):** Route A is
    sound-but-VACUOUS (222 hops on 5 windows). Tight object = **block-transfer rate ρ(G)** (G=block-Jacobi of
    reblocked resolvent): ρ(G) 0.33/0.42/0.85, 10–100× tighter, tracks the exact resolvent. Transient growth
    (‖G‖ 5–22) defeats a norm bound; **adapted (Stein/Lyapunov) norm** gives ‖Gᵏ‖≤√κ(P)rᵏ → **certified ~1 hop
    @ const 26–57** (bidir16), rate tight, non-normality quarantined into √κ(P). **LINEAGE SHIFT: off
    approximation-theory (DMS/Faber), onto dynamical-systems/Lyapunov = the causal face's own lineage.** Causal
    G nilpotent (ρ=0) = exact product-Lyapunov corner; bidir ρ(G)∈(0,1) = geometric. **Two faces = two regimes
    of one operator G**; Faber demoted to loose floor. ρ(G)<1 = SPATIAL contraction of the resolvent iteration,
    NOT ρ(J) (to 8.4) — "conditioning not contraction" sharpened into two axes (ρ(G) reach; σ_min error).
    ~~must-carry dissolves~~ **CORRECTED: must-carry PERSISTS on this
    substrate** (irrelevant far/near 0.061 vs causal 0.068 — nearly identical; READONLY_Q makes queries
    invisible to the context, so query-awareness is architecturally impossible here and the reader-set
    principle predicts exactly this; the old "dissolves" contrast used a stale v2-era causal ξ≈27. True
    query-awareness needs the QUERY-VISIBLE substrate → bidirqv retrain + C2t); **maintenance channel
    quantified**: warm 4 vs cold 14–22 evals on filler (3.5–5.5×), warm≈cold
    on relevant edits (cost ∝ how far the solution moves). At σ_min=0.016 one seq near-multistable → filler
    gated "not measurable: approaching the uniqueness boundary" (honest degradation, same as causal face).
  - **The reader-set principle (the correct general statement of must-carry — supersedes "causal carries,
    bidirectional doesn't").** Three edit tiers have three *logical statuses*, not three magnitudes: (i)
    **queried-value** edits — transport to the cursor is **information-theoretically necessary in both faces**
    (if the answer changes it must arrive; measured ridge far/near ~0.09–0.10 both faces); (ii) **filler** —
    never carried, either face (the fair envelope witness); (iii) **unqueried-value** — the tier where faces
    diverge, and the divergence is **impossibility vs. observed capability, not two guarantees**: causally,
    carrying is *forced* (the relay cannot condition on future queries → must keep every binding — an
    availability argument, architecture-level, theorem-flavored), whereas bidirectionally selective forgetting
    is *permitted* and our trained model *exercises* it — **emergent, not certified** (a carry-everything
    bidirectional model scores identically on recall; nothing in the loss demands selectivity). The deep,
    general statement: **selectivity is possible exactly w.r.t. readers *present in the context at solve time*;
    unknown/future readers force carry in *any* architecture.** Causal attention is the special case where all
    readers are structurally unknown (future by construction). CONSEQUENCE for the maintenance workload
    (edit-now, query-later): future readers are unknown even to a bidirectional model → a must-carry-like
    burden returns; our C2-bidir measured selectivity only because the queries sit *in* the solved context.
    DIVISION OF LABOR (the load-bearing framing): the σ_min/Faber **envelope upper-bounds *every* edit class,
    selective or not — that is the certificate (a guarantee)**; must-carry vs. query-awareness only describes
    *where inside that sound envelope* the trained map places transport — that is measured mechanism. Same
    structure as loose-but-sound recompute: certify with the envelope, observe the realized footprint is
    usually far smaller.
- **C3 (tradeoff — DISSOLVES; result in, `c3_mixing_locality.py`).** Predicted a Pareto (small `w`: slow
  solve, local edits; large `w`: fast solve, global edits). Measured (window sweep, ξ fit in positions):
  solve-iters **fall** with `w` (98→40) but ξ_positions is **flat and window-independent** (~3–5 tokens,
  sub-window throughout; w=8/12/20 → 4.95/3.10/3.81, no trend) — **not** ξ∝w. So the mixing↔locality tradeoff
  **dissolves at equilibrium**: `w` is a *solve-speed* dial, edit-reach is set by σ_min (conditioning), not the
  window. This is C1's reach-decoupling seen from the maintenance side — a *confirmatory corollary* of C1 + the
  σ_min thesis (discriminates our framework from the naive finite-depth `w·K` Pareto), **not** a standalone
  surprise. CAVEAT: flatness may partly reflect our contraction control (SN + s_max cap) pinning learned σ_min
  into a similar range across `w`; the honest claim is "reach tracked conditioning, and conditioning didn't
  move much with `w`," not "the window is powerless." Don't oversell.
- **C2d (directional certificate — MEASURED, `c2d_directional.py`; result in, discharges the product-form
  debt).** Far-reach map F_p from the exact-resolvent oracle; pred_far = ‖F_p·δh‖ **pre-solve**. Validated on
  curr16/24/40: **V1** linear-response profiles match measured nonlinear responses (log-log corr 0.98/0.98/0.93);
  **V2** the 3-tier taxonomy is predicted *a-priori* from δh alone, quantitatively (3 orders of magnitude class
  separation); **V3** zero false containments everywhere (the work-saving verdict is safe); as a *quantitative
  bound* it under-predicts carry-exciting edits ~2–7× at near-singular conditioning → classifier + predictor,
  not certified upper bound there; **V4** carry rank ~**8 of 64**, stable across conditioning — rank-r update,
  not full-suffix; **V5** coarse w-window T-product **reconstructs the exact resolvent block at 1.5e-15**
  (re-blocking theorem operationalized) while the scalar norm-product bound's slack explodes **7.5×→764×** over
  hops (per-hop norms all >1 = scalar vacuous, directional decays): **at trained conditioning, direction is the
  entire content of the causal certificate.** Completes the claw-back ladder (findings digest §10).
- **C2i (edit-interference — maps the linear-regime validity boundary).** Response to *paired* edits vs the sum
  of single-edit responses, as a function of separation. Where they diverge = where nonlinear attention
  re-routing breaks first-order superposition = the validity boundary of the whole directional-certificate
  story. Cheap, same C2 machinery, one new loop. Also grounds multi-token edits (stack-and-project).
- **C2t (deferred billing / trigger — measures the lazy-vs-eager evaluation split; the reader-set principle as
  an iteration ledger).** KEY TRICK: retargeting a query token (substituting which key it asks) IS the "reader
  arrives" event — just another substitution edit the harness already supports. Protocol per ckpt/seq:
  **lazy path** (bidir): (1) edit an *unqueried* value → warm re-solve → `iters_write` (predict: cheap,
  contained — the query-aware relay stores it locally); (2) retarget a query to the edited key → warm re-solve →
  `iters_trigger` (predict: expensive, gap-dependent — the binding must travel NOW; the transport ridge should
  appear in the response profile *at trigger time, not write time*). **Eager control:** retarget first (reader
  present), then edit → cost lands at write (`iters_write_eager` large). **Baseline:** retarget a query to an
  *unedited* key (pure reader-arrival cost). **RESULT (MEASURED — NEGATIVE, reported straight):** the
  lazy-evaluation prediction **did not hold**. Clean signal = **write-cost(edit unqueried value) is
  reader-INDEPENDENT on all three substrates** (causal 7.6≈7.8, readonly 12.9≈13.0, query-visible 15.3≈15.7,
  lazy≈eager) → the relay carries unqueried bindings regardless of reader presence → **must-carry is
  empirically robust, and selective forgetting did NOT emerge even where architecturally permitted
  (query-visible).** = "emergent not certified" coming back negative (nothing in the recall loss rewards
  selectivity; qv also trained worse). Reader visibility made the query-retarget *more* expensive (readonly
  trigger ~4 vs qv ~11 — visibility couples the reader into the context equilibrium) = the opposite of a
  laziness win. Path-independence held cleanly (totals conserved, final states 1e-7 = warm-start exactness).
  MEASUREMENT LESSON: dz@reader-position is confounded (retargeting changes the query token's own embedding =
  direct state change, not transport). **SURVIVES:** tier-3 metering (cost ∝ realized ‖Δz‖); the "two faces =
  eager vs lazy evaluation" framing is **demoted** to "lazy is permitted but not incentivized — these trained
  models are all eager." Bonus finding: bidirqv substrate = a **visibility↔trainability tension** (recall
  0.94→0.63 at gap 40 vs readonly).
- **C2m (emergent metering — MEASURED, `c2m_metering.py`; the law is FACE-DEPENDENT, inverting the
  load-bearing assignment).** Real edits + synthetic carry/transverse perturbations; cold solves = flat toll;
  ‖R·δh‖ = pre-solve forecast. **Bidirectional face gets the clean output-metering law** (Spearman(n_warm,‖Δz‖)
  0.90/0.92/0.89, partial corr w/ input norm ≈ 0); **causal face weak and mode-confounded** (0.67/0.65 with
  negative partials — carry-aligned movement = slow modes = disproportionate cost; **absent** at near-singular
  curr40, 0.09). Mechanism = ν a third time: uniform mode rates (near-normal) → magnitude metering; wild mode
  rates (non-normal, per-hop norms to 25) → magnitude under-determines cost. **Faces differ in proof family,
  trainability, and billing legibility.** Universal winner: the forecast ‖R·δh‖→‖Δz‖ at Spearman 0.96–1.00 on
  both faces at all conditioning. Flat toll confirmed (cold ≈ constant, Spearman ≈ 0). Caveats: 18–36% floor
  points (coarse at the small end); slope-vs-σ_min inconclusive. Figure: `checkpoints/c2m_records.npz`.
- **C2ν (semi-causal dial — SLATED, not run; the missing control for every "ν governs X" claim).** Our
  ν-governs results (proof family, trainability, billing legibility) are confounded with causality itself
  (ν only varies through the mask). Control: an **asymmetric band** [i−w, i+βw], β ∈ {0, 0.25, 0.5, 1} —
  ν moves continuously with β; metering legibility (C2m rerun) then either tracks ν continuously (ν is the
  governing variable) or jumps at β>0 (strict triangularity is topologically special; ν was a proxy). Either
  outcome sharpens the claim. Streaming/ASR bounded-lookahead attention = the practical semi-causal mirror.
  Cost: curriculum retrain + C2m pass per β.
- **C4 (multi-scale resolution).** Adding `O(log n)` coarse / global nodes lets a *long-range-relevant*
  edit reach the generation point in `O(log n)` via the coarse channel (local ball + `O(log n)` coarse
  updates) instead of full-suffix recompute — at the cost of the coarse nodes being bounded (`O(log n)`)
  locality-breaking hubs. This is the concrete answer to "how does the signal reach the end without a
  whole recomputation."
- **C-insert (insert/delete — v2 spine, NOT this paper; now UNBLOCKED).** Aligned-frame reduction: an
  insert/delete under relative PE + banded attention = a width-`w` multi-site substitution at the cut (theory,
  §11). The viability premise is **verified**: the pure-relative substrate exists and relays (bidirnp00–40,
  recall 0.82–1.00). Remaining: insert-type `apply_edit` + alignment bookkeeping, measured vs the
  "band at the cut" prediction.

---

### Scoped plan — the end-to-end pipeline + a hub/spoke ground-truth test (PLANNED, 2026-07-08)

Motivation: every result so far is a *separate* measurement-against-a-bound on **MQAR**, whose dependency
structure is **planted and near-uniform**. Two honest gaps: (i) the a-priori reach ball, the residual bound
(§4 of Note #11), and the reader-set machinery have never been run as **one integrated loop**; (ii) the
**reader-set / dependency-graph** claim — our most distinctive angle — has never been tested against a task
with *heterogeneous, non-planted* dependencies (real hubs vs spokes). Both are fixable **without** leaving the
theory/small-experiment side (synthetic tasks, known ground truth, toy DEQ, no LM-scale bakeoff). Decision:
do **C5 regardless** (cheap, and it is the harness-level evidence the paper implicitly promises); do **C6**
because the reader-set/lanes idea is the headline and is currently untested — but keep it a **ground-truth
synthetic**, never real text.

- **C5 (end-to-end certified-recompute pipeline — PLANNED; runs on the CURRENT curr/bidir checkpoints, no new
  model).** Materialize tiers 1+2 + warm-start as a single validated artifact. Loop per (ckpt, seq, edit):
  (1) apply an edit; (2) compute the **a-priori** reach ball from ρ(G)/eff_rate (Note #11 §§1–3) walked along
  the attention-support graph; (3) **partial re-solve only that ball** (freeze the complement); (4)
  **residual-certify** the frozen region via `‖z−z*‖ ≤ ‖f(z)−z‖/σ_min` (§4); (5) **verify against a full
  re-solve** that the frozen region really is within the certified tolerance; (6) report. Metrics:
  **containment** (true changed set ⊆ certified ball → target **zero false containments**, the C2d-V3 safety
  bar); **exactness** (‖partial − full‖ on the frozen region ≤ certified tol, i.e. the residual bound holds
  empirically); **recompute fraction** |ball|/L; and the **Kantorovich numbers** h=βLη and R₋ from a probed L
  (show h ≤ ½ fires so the rigorous version — not just the linear bound — is in force).
  **EFFICIENCY PROBE (iterations as the compute proxy — no wall-clock, per the ledger's "count evals, don't
  trust wall-clock" rule):** log (a) `iters_full` / f-evals of a cold full re-solve (baseline), (b)
  `iters_partial` / f-evals of the warm partial-ball re-solve, and (c) the honest **compute proxy** = iters ×
  active tokens, since one f-eval over a ball of `b` tokens costs O(b·w) not O(L·w). Headline number:
  `iters_full·L` vs `iters_partial·|ball|` (this is where the recompute fraction |ball|/L *becomes* a compute
  claim, not just a set-size claim). Report the iteration ratio AND the compute-proxy ratio; both must beat 1
  for the certificate to pay for itself, and the gap between them exposes any warm-start iteration penalty.
  Deliverable = "the machinery works as a *system* AND saves measurable compute," replacing three isolated
  bound-checks with one closed loop.
  **INGREDIENT / ABLATION LADDER (toggle each, measure on the compute proxy — "which piece earns its keep"):**
  (0) cold full re-solve (baseline); (1) + ball restriction (solve only the reach ball); (2) + warm start (old
  z* as the init); (3) **+ Woodbury low-rank PREDICTOR as a "warmer-than-warm" prior** — the fun one: the edit
  is a low-rank δJ=UVᵀ (footprint from the weights, carry rank ~8 of 64 per C2d-V4), so the Woodbury update
  `M_new⁻¹ = M⁻¹ + M⁻¹U(I−VᵀM⁻¹U)⁻¹VᵀM⁻¹` gives a **first-order-exact prediction of the new equilibrium**
  (exact if f affine) to initialize from, *strictly better than copying the old z*. Then **residual-certify the
  prediction (§4) and iterate only if the residual exceeds tol** — often it doesn't, so the solve is *skipped*.
  This is the Kantorovich predict→certify→correct loop: Woodbury = predictor, partial re-solve = corrector,
  residual bound = the "do I even need to correct?" test. Ingredients for the update: U,V from the weights,
  M⁻¹U from the *cached* warm resolvent action, an r×r reduced solve (cheap, dense). (4) **+ adaptive per-WINDOW
  early-stop** — freeze a window once its certified per-token error $[M⁻¹\,\text{res}]_i=\sum_j[M⁻¹]_{ij}\text{res}_j$
  (the resolvent row · residual field = the reach envelope, already bounded) drops below tol; the ball
  *shrinks* as outer windows converge. **OWED CHECK:** this makes the iteration masked/Gauss–Seidel-style —
  verify it still converges to the *same* fixed point under our diagonal-dominance/contraction conditions
  (expected, not assumed). Do it per-window (contiguous), NOT per-token (see hardware note). Each rung is one
  toggle on the same loop; the ladder tells us whether Woodbury+gating actually beat plain warm-start-on-a-ball.
  **HARDWARE-ALIGNMENT NOTE (design-level, NOT a systems/wall-clock claim; goal: don't build something absurd
  at the kernel level).** The hazard is fine-grained per-token dynamic sparsity (irregular gather/scatter,
  warp divergence, memory-bound) — a proxy win that's a wall-clock loss. We avoid it because the framework's
  natural granularities are already GPU-aligned, and not by accident: (i) the reblocking that makes the ρ(G)
  certificate clean also makes a recompute ball a **contiguous range of windows** = a dense sub-slice → restricting
  the solve is just **the existing attention kernel on a shorter sequence**, no sparse primitive; per-window
  early-stop shrinks that contiguous slice; (ii) the carry is **low-rank, not sparse**, so Woodbury is a small
  *dense* r×r solve + tall-skinny matmuls (dense BLAS); (iii) window size `w` is a free knob → align `w·d` to
  tensor-core tile sizes so block-Jacobi blocks *are* the tiles; (iv) DEQ's O(1)-memory implicit diff +
  compute-bound matmul solve are memory-friendly where it counts. **Design rule: stay at block/window
  granularity + low-rank corrections; never make per-token gather the headline.** We report only the
  architecture-level proxy (iters × active tokens), realizable by coarse-contiguous recompute, and explicitly
  disclaim wall-clock; kernel fusion / ragged-batch scheduling = named out of scope. Caveat: this is an
  *argument* that the mechanism isn't hardware-hostile, not a profile — "checked it's not absurd," not "it's fast."

- **C6 (hub/spoke structured-dependency task — PLANNED; the ground-truth test of the reader-set machinery
  beyond planted-uniform MQAR).** Pick a synthetic where positions have **genuinely heterogeneous influence
  footprints** and the true dependency graph is **known exactly**. Shortlist (primary → dense control):
  (a) **pointer-chase / hierarchical MQAR** — keys point to keys (indirection chains), a root key is a **hub**
  read by many, chains create narrow **lanes**; (b) **bracket-matching / state-tracking** — an open bracket is
  a hub for everything until its close, nesting depth = a spread of footprints; (c) **sorting / set-op** — a
  deliberately **dense** control where dependencies are global, so lanes *should* fill (the honest negative
  case that checks the certificate reports a large ball when it must, not a false sparsity win). For each, with
  the true graph in hand, measure: **safety** (certified reader-set ⊇ true dependency set, zero false
  containment); **tightness** (|certified| / |true|); **lane sparsity** (do recompute paths zigzag in narrow
  lanes or fill the downstream cone — the sparsity hypothesis, with sorting as the built-in dense witness);
  **hub identification** (does the resolvent / σ_min-locality correctly flag hub tokens as wide-footprint and
  spokes as narrow?). Runs the *entire* moving apparatus — ρ(G) reach, σ_min residual bound, C5 loop, reader
  cones — on non-uniform structure with ground truth. Cost: a small task generator + a curriculum retrain of
  the same cell; no DEQ-LM. This is the load-bearing addition if the reader-set is a **headline** claim (you
  cannot claim a dependency-aware method and only ever show planted-uniform dependencies); if it stays a
  discussion-level *principle*, C5 alone suffices for TMLR.

  **CONCRETE GENERATORS (spec, 2026-07-08; establishment status checked).** Base MQAR is canonical (Zoology /
  Arora et al. 2023); the two hub/spoke variants below extend it — pointer-chasing is a classic task family,
  multi-hop AR appears in eval suites (MAD, mechanistic-eval work) but has **no single canonical generator**, so
  these generators are ours, grounded in established tasks. Both hand you the **true dependency graph by
  construction** — that is the whole point.
  - **Task A — multi-hop / hierarchical MQAR** (tree/DAG topology; hubs by fan-in). Bindings form a depth-`D`
    indirection DAG: level-0 keys → level-1 keys → … → value (`D` = 2–3). Vocab split into key- and value-space;
    sequence = shuffled binding token-pairs `[child : parent]` + queries at the tail; answering a query **chases
    `D` hops**. **Hub** = a node with high fan-in (referenced by many children) → wide influence footprint;
    **spoke** = a leaf. Ground-truth recompute set for editing binding `b` = its **ancestor cone** (every query
    that chases through `b`). Params: depth `D`, fan-in distribution (dials hub-ness), n_bindings, gap.
  - **Task B — pointer-chase** (chain topology; hubs by in-degree / shared tails; the cleanest lane test). A
    functional graph on `N` nodes, each with a pointer `ptr(i)`; sequence encodes the table as pairs
    `[i : ptr(i)]` + a query `(start s, hops k)`; answer = `ptr^k(s)`. **Permutation** pointers → disjoint cycles
    → each query walks a **narrow lane of exactly its k nodes** (the sparse-lanes ideal); **random-function**
    pointers → ρ-shaped graphs with **shared tails = super-hubs** (the heterogeneous case). Editing `ptr(j)`
    invalidates exactly the queries whose chain passes through `j` → a **path-shaped, sparse** recompute set —
    the direct test of "does the certified reader-set zigzag in lanes or fill the cone." Params: `N`, chain
    length `k`, permutation vs random-function (the hub dial).
  - **Dense control — sorting / set-op** (already in the shortlist): global dependencies, lanes *should* fill;
    the honest witness that the certificate reports a large ball when it must, rather than a false sparsity win.

- **On a real-text DEQ-LM (WikiText etc.) — DEFERRED, optional, robustness-only; DO NOT let it become the
  paper.** Real text = the only place you *can't* test the headline (no ground-truth dependency graph → you
  drop to self-consistency: partial re-solve ≈ full re-solve), it flips the genre (perplexity/recompute tables
  invite systems baselines — CacheBlend/ProphetKV on Llama — and wall-clock we can't provide on 6 GB), and it
  smuggles in "first build a DEQ-LM." Note DEQ-LMs are **not** hypothetical: **Bai, Kolter & Koltun 2019
  ("Deep Equilibrium Models")** trained a DEQ-Transformer (and TrellisNet) on **WikiText-103** (code:
  locuslab/deq), and the maintained modern vehicle is **`torchdeq` (Geng's library)**, whose DEQ Zoo ships
  `deq-lm` (WikiText-103, word-level).
  **SEARCH FINDINGS (2026-07-08):** (i) the torchdeq DEQ-Zoo model doc gives launch/data instructions but
  **does not advertise a released *pretrained* LM checkpoint** — availability unconfirmed (check the
  locuslab/deq repo releases / HF directly before relying on it); (ii) confirmed the cell is **"MultiHead
  Decoder Attention" = full causal attention**, so it will **not** drop into our windowed / block-tridiagonal
  structure that the **ρ(G) reach half** of the certificate needs. **USEFUL NUANCE:** a full-attention DEQ-LM
  still *has a fixed point*, so the **σ_min residual bound (§4) and the reader-set diagnostic transfer to it**
  as-is (they need only a fixed point, not windowing) — only the ρ(G) block-transfer *reach* certificate is
  windowing-bound. So IF a pretrained `deq-lm` checkpoint turns up, it could test the **residual-bound half of
  C5 on real text** for near-zero cost (no training), while the reach half stays on our windowed substrates.
  Otherwise a real-text run means training a *small windowed* DEQ-LM ourselves via torchdeq (char/byte, tiny
  corpus), self-consistency-only, framed as "scales to natural text," **not** a benchmark — and only if
  C5 + C6 land first.

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
