# Bidirectional-face run — findings to internalize for the paper

*Session 2026-07-07. Everything below is measured or lit-checked in this run; each item lists the
claim, the evidence, the honest scope, and where it lands in the paper. Read this before drafting the
C2 section. Companion files: [`deq_transformer_blueprint.md`](deq_transformer_blueprint.md) (design +
claims), [`sequence_certificate_literature.md`](sequence_certificate_literature.md) (prior art),
`09-causal-face-product-lyapunov.html` (the two-faces certificate).*

---

## 0. Headline — the spine is complete

C1 (reach), C2-causal, **C2-bidirectional**, C3 (tradeoff dissolves) are all measured. C2-bidirectional
was the last owed experiment — the one the abstract headlines (the edit regime). Both faces of the
certificate now have data.

---

## 1. The BPTT bridge equation — one formula ties us to RNN-Lyapunov theory

**Claim.** The backprop-through-time gradient
`∇_{h_t}L = Σ_{s≥t} (∏_{r=t+1}^{s} J_rᵀ) Wᵀ ∇_{o_s}L`
is the **backward twin** of our forward edit-response. Four exact connections:
1. Forward edit `(I−J)⁻¹δh` and backward gradient `(I−Jᵀ)⁻¹∇` are **transposes**; σ_min is
   transpose-invariant → **one conditioning number certifies edit-locality (forward) and gradient
   conditioning (backward)**. This is the concrete identity behind the "third face" error-amp `resid/σ_min`.
2. BPTT's finite sum = a **truncated Neumann** series of `(I−Jᵀ)⁻¹`; the DEQ/IFT gradient is the resolvent
   limit → gradient-side mirror of "feedforward is a truncation, DEQ is the resolvent" (diverges at ρ>1
   where the resolvent persists).
3. For causal/triangular J our product-Lyapunov certificate **is** this Jacobian product, read forward over
   positions instead of backward over time.
4. **Duality punchline:** vanishing gradients ⟺ edit-locality (λ<0 = short credit assignment = short reach);
   λ≈0 = long memory *and* long reach, paid in conditioning. RNN memory↔trainability tension **is** our
   reach↔maintainability tension.

**Use in paper.** This is *the* bridge equation — an RNN-dynamics reviewer recognizes it instantly and reads
our forward edit-locality as its dual. Cite **Vogt et al. 2006.14123** (builds directly on this object).
**Scope:** it's a reframing/unification, not new math; state it as such.

---

## 2. The bidirectional training blocker — a genuine, near-undocumented phenomenon

**Claim.** Bidirectional attention-only models **fail to form the two-hop bind-then-retrieve circuit** on
MQAR; they plateau at the **one-attention-layer ceiling** (recall ≈ 0.38 = optimal value-*set* guessing).
The failure is an **optimization-path** problem, not expressivity.

**Evidence (6 probe rounds, `bidir_train_probe.py`, `checkpoints/probe_bidir_round*.txt`).** Ruled out, all
stuck ~0.38: init (causal mask-swap transfer), steps (700), lr, s_max, per-head relative-position bias, **and
equilibrium itself** (unroll4-bidir stuck too, while unroll4-*causal* solved gap 0–16 at 1.0 in C1), **and
weight-tying** (untied 2/4-layer mini-BERT controls stuck). Every config landed at the one-layer ceiling =
the same 0.38 as C1's unroll2-causal.

**Mechanism (secondary but real): consensus collapse / oversmoothing.** Bidirectional row-stochastic attention
has a Perron/consensus mode the resolvent amplifies by `1/(1−s)` while contracting token-distinguishing
differences — the recall signal lives in exactly those differences. Measured by mean pairwise cosine of the
fixed-point token states (`cossim`): **healthy causal 0.033 vs stuck bidirectional 0.274**. But finite depth
(unroll4) fails identically → collapse is a *pull*, not the whole story; the binding circuit simply never
gets a gradient toward forming.

**The rescue: window curriculum (our improvisation).** Train at **w=2** (a value token's band is just its own
key + the next key → the binding hop is *forced by connectivity*; rel-bias picks "left"), with query rows
reading the full context (`QUERY_FULL`, so retrieval has a gradient at w=2). Recall snaps
**0.385 → 0.838 → 1.000** (plateau-then-abrupt transition, exactly the induction-head signature; cossim falls
0.23 → 0.09 as the circuit forms) and **survives widening w=2→4→10** (1.000 at w=10). Then re-band queries and
run the gap curriculum. Final substrate: recall **1.0/0.997/0.987/0.995/0.938** at gaps 0–40.

**Literature status (searched this run).** The two halves are known **separately**, the coupling appears
**undocumented**:
- *Forward half (rank collapse vs mask):* Dong et al. 2021 (pure attention loses rank doubly-exponentially);
  **Wu et al. 2405.18781 (NeurIPS'24) — CITE:** collapse rate monotone in graph **radius**, causal mask has one
  center node → provably *looser* bound ("the advantage of the causal mask"), local attention slows collapse.
  *No trainability analysis.* BERT oversmoothing: Shi et al. 2022.
- *Learning half (one-hop shortcut first, two-hop later):* Bietti 2306.00802 (Birth of a Transformer — global
  bigrams fast, induction head slow/top-down), 2409.10559 (provable phased dynamics), 2506.13688 (loss-plateau
  circuit stalls), 2410.19637 (distributional simplicity bias). **All causal-only.**
- **Nobody shows bidirectional masks *block* two-hop circuit formation while identical architectures succeed
  causally.** Growing *sequence-length* curricula are established (sequence-length warmup, Staged Training,
  HyenaDNA); growing the *attention band at fixed length to force a specific circuit* — connectivity-as-curriculum
  — appears to be ours.

**Scope.** **Appendix training-recipe observation**, not a headline. It *supports* the two-faces story (the
faces differ in proof family, failure mode, **and** trainability) and connects to Wu et al.'s radius theory.
Don't overclaim novelty of the mechanism — claim the *coupling observation* + the *curriculum fix*.

**Why prior transformer-iteration work didn't hit this:** vision/NCA iteration lives on smooth local features
that are happy with averaging-flavored equilibria (no binding-by-content needed); Bai/Geng DEQ-LMs were
**causal**, where the mask hands the binding hop its directionality for free.

---

## 3. Two-hop induction — the circuit we were fighting to form

The minimal associative-recall circuit needs two chained attention moves (one step can gather but not
gather-then-match): **Hop 1 (binding)** — each value attends to its previous token (its key), so its state
encodes the pair; **Hop 2 (retrieval)** — the query attends *by content* over those enriched states and reads
the value. Neither alone suffices: no hop 1 → a content query retrieves the undifferentiated value *set*
(= our 0.38 ceiling: the model knows the four values, not which key owns which). This is the interpretability
literature's induction head (previous-token head + matching head). Our whole saga in one line: **causal
training finds both hops easily; bidirectional training fell into the hop-2-only shortcut until w=2 made
"look left" nearly the only edge.**

---

## 4. C2-bidirectional results — the Faber face, measured

`c2_bidir.py` on `bidir00–40`. This is the theoretically *proper* face for the κ→ξ Faber formula.

| Finding | Evidence | Status |
|---|---|---|
| **Envelope holds** on filler | ξ 0.29/0.41/0.51 hops vs Faber 4.6/6.8/6.3 (~10× conservative, sound) | the certificate claim, on the proper face |
| **Conditioning governs reach** | filler ξ grows monotonically as σ_min falls | confirms σ_min thesis, bidir side |
| **Genuinely two-sided** | left-mass up to 0.44 (causal: 0 by construction) | the two-sided decay plot (`c2_bidir_profiles.npz`) |
| **ν justifies the per-face split** | ν_bidir 0.21–0.31 vs ν_causal 0.32–0.71 (gap40: 0.21 vs 0.71) | say "near-normal, within Faber's domain" — **not** "normal" |
| **Must-carry dissolves** | irrelevant-edit ξ 0.7–1.6 (causal up to 27), far/near ≤0.061 | query-aware relay = architecture governs transport |
| **Maintenance channel** | warm 4 vs cold 14–22 evals on filler (3.5–5.5×); warm≈cold on relevant | cost ∝ solution movement |
| **Honest edge** | at σ_min=0.016 one seq near-multistable → filler gated "not measurable" | degrades honestly, same as causal face |

**The substrate has a bonus property:** ρ stays **< 1 throughout** (0.43→0.87) while σ_min spans 15× — the
bidirectional relay meets in the middle (half the effective depth), so it lives natively in the Faber/κ domain
of validity, exactly where the two-faces theory wants it. (Causal needed ρ up to 8.4 for comparable reach.)

---

## 5. The reader-set principle — the correct general statement of must-carry

**Supersedes "causal carries, bidirectional doesn't."** Three edit tiers, three *logical statuses*:
- **Queried-value:** transport to the cursor is **information-theoretically necessary in both faces** (if the
  answer changes it must arrive). Measured ridge far/near ~0.09–0.10 both faces. Nothing forgets these, ever.
- **Filler:** never carried, either face (the fair envelope witness).
- **Unqueried-value:** the tier where faces diverge — **impossibility vs. observed capability, not two
  guarantees.** Causally, carry is *forced* (the relay can't condition on future queries → must keep every
  binding; availability argument, architecture-level, theorem-flavored). Bidirectionally, selective forgetting
  is *permitted* and our trained model *exercises* it — **emergent, not certified** (a carry-everything bidir
  model scores identically on recall; nothing in the loss demands selectivity).

**The deep statement:** *selectivity is possible exactly w.r.t. readers **present in the context at solve
time**; unknown/future readers force carry in any architecture.* Causal attention = the special case where all
readers are structurally unknown (future by construction).

**Consequence for the real workload (edit-now, query-later):** future readers are unknown even to a
bidirectional model → a must-carry-like burden **returns**; our C2-bidir measured selectivity only because the
queries sit *in* the solved context.

**Division of labor (load-bearing framing):** the σ_min/Faber **envelope upper-bounds every edit class,
selective or not — that is the certificate (a guarantee)**; must-carry vs. query-awareness only describes
*where inside that sound envelope* the trained map puts transport — that is measured mechanism. Same structure
as loose-but-sound recompute.

**How the ξ-ball interfaces with must-carry, and the central honesty (edit-locality is dual to forgetting).**
The certificate is worst-case (smallest-σ_min direction), so the ξ-ball **always contains** the carry
direction → soundness holds for a queried-value edit (it correctly says "can reach the cursor"). But the *same*
σ_min gives ξ (edit-locality) *and* the memory horizon: small σ_min = long ξ = long memory. So **a causal LM
doing its job (long memory) has a large ξ by necessity — its edits are not local**, and a causal model with
local edits is one that has *forgotten*. Edit-locality in the causal regime is exactly the property you don't
want. Interface verdict: in the useful causal regime the ξ-ball ≈ the whole suffix → the certificate is **sound
but vacuous** (no compression) for carry-exciting edits; it only compresses for filler/unqueried edits, and
"cheap maintenance of the irrelevant context" is a weak generation-time pitch. Nuance: reach is *anisotropic*
(product-Lyapunov coupling ≈1 along the low-dim carry, <1 transverse; scalar σ_min reports the carry), so it's
not literally *every* edit — but the demotion stands. **Consequence for scope:** the causal face is the *proof
ground* (product-Lyapunov / BPTT bridge) and the regime where we *characterize the must-carry limitation* — NOT
a maintenance proposal. A maintenance *win* needs σ_min bounded from 0 in all directions (no long carry) = the
**bidirectional local-readout niche with readers present in context**. Even there, edit-now/query-later brings
back unknown future readers → must-carry-like burden; the cleanest claimable regime is bidirectional
local-readout where relevant readers are already in context. **State this plainly in the paper — do not sell
causal-LM maintenance.**

---

## 6. The KV-cache interface — what our object *is* in serving terms

The cached object is **not** the embeddings (`h0` is the input injection; caching it restarts the solve). It
is the **equilibrium state `z*`** (or `Wk z*, Wv z*`).

| Standard transformer | Equilibrium analog |
|---|---|
| KV cache: per-layer K,V, `O(L·n·d)` | one `z*`, **`O(n·d)`** (weight-tying collapses depth) |
| Append reuses cache exactly | causal face: prefix equilibria independent of new token → solve one position vs frozen `z*` |
| Edit → heuristic lossy partial reuse (CacheBlend/PIE) | edit → **provably valid outside the ξ-ball**, re-solve inside, warm-start from the cache itself = **certified partial invalidation** |
| cache is a speed trick | `z*` is dual-use: **is** the cache (decode) and **is** the warm start (edit) |

**Paper line:** the equilibrium state is a KV cache whose invalidation region is a **theorem** instead of a
guess — the sound version of CacheBlend; Geng's scenario B closed into a loop. **Counterweights:** `O(n·d)`
beats `O(L·n·d)` in memory but decode still pays solver iterations per token (no throughput claim); on the
**bidirectional face there is no free append** (a tail token perturbs its own ξ-ball backward).

---

## 7. Warm-start gains: why 3.5–5.5× here vs 1.1–2× on graphs

The ratio = (global relaxation cost) / (local re-settle cost); both factors moved:
- **Denominator shrank relative to L.** Win ∝ L/ξ (fraction already converged at warm init). Sequences are
  long relative to ξ (L up to 50, ξ < one window); the graph experiments edited small graphs where the ball
  *was* much of the graph.
- **Numerator grew.** Cold solves here are genuinely expensive (14–26 evals, iteration-bound, critical slowing
  as σ_min → 0.016). Graph cells self-limited to comfortable contraction → cold was already cheap, nothing to
  save. Trained sequence models live near the conditioning edge *because reach demands it* — exactly where
  warm-starting pays.
- **Honest caveat:** the 3.5–5.5× is on *contained* (filler) edits; on relevant edits warm≈cold (~1.1×,
  resembling the old graph numbers). The right claim is **adaptive cost**, not universal speedup.

**"Cost ∝ solution movement" precisely:** warm-start iterations track `‖z*_new − z*_old‖`, not depth, not L.
Contained edit → solution barely moves → ~4 evals (cold ~20). Queried edit → answer must change, information
must travel → warm ≈ cold (the information-theoretic floor). Feedforward charges all L layers for *any* edit;
the equilibrium meters cost by what the edit actually did.

---

## 8. "ν justifies the two faces empirically" — what that means

Theory *assigns* certificates by Jacobian geometry: Faber/κ needs **near-normal** J; triangular J is maximally
non-normal (spectrum lies), where product-Lyapunov takes over. **ν = ‖JJᵀ−JᵀJ‖_F / ‖J‖²_F** is the measured
departure-from-normality. ν_bidir 0.21–0.31 vs ν_causal 0.32–0.71, gap widest (0.21 vs 0.71) at the
hardest-trained checkpoint → **each trained model sits where its assigned certificate is valid**; the face-split
is a *measured property of the Jacobians*, not a modeling assumption. Honest footnote: 0.2–0.3 ≠ 0, and the
~10× envelope margin is what absorbs the imperfection in practice.

---

## 9. Loose ends & things to remember

- **Subhomogeneous DEQs (Sittoni & Tudisco, ICML'24, 2403.00720):** we took *nothing operational* — it's an
  **honesty citation** representing the "uniqueness by architectural construction" school (nonlinear
  Perron–Frobenius, Thomson-metric contraction). We contrast: *our* uniqueness is measured **per-instance** via
  σ_min + multistability probes on unconstrained models, and that school never prices spatial edit reach — the
  boundary of our sliver.
- **Practical bidirectional design** = alternating local windows (Swin = Margolus construction; ModernBERT) or
  local band + global hubs (Longformer/BigBird = our C4). **Cite, don't build** a second substrate — a full
  pipeline rerun for a footnote. Staggered-block = good future work, ideally on a seam-sensitive task.
- **Insert/delete under relative PE** is **screened, not wide**: the "exponential shadow on both sides" *is*
  the σ_min ξ-ball (+ O(n) index bookkeeping, zero recompute), which the certificate prices. Under absolute PE
  it really is global. **Bonus:** the rel-bias we added to fix the binding blocker makes the bidir substrate
  already relative-PE → a C2-insert (v2) measurement needs no architecture change, only an insert-type
  `apply_edit`.
- **The Excel/NCA anecdote (ZJ):** linearly-interpolated Wolfram CA collapsing to gray 0.5 = consensus
  collapse; discrete CA stays interesting because thresholding is maximally non-averaging (= peaked softmax
  re-sharpening). One-line inhabitant of the NCA corner of the four-lands figure: continuous-CA gray-out =
  oversmoothing = rank collapse — one phenomenon, three fields' names.

---

## 10. Remaining (non-spine) debts before drafting

1. Full reads of the 4 must-read refs: Vogt 2006.14123, Cirone 2402.19047, Benzi–Golub decay, 2411.04400.
2. Product-form certificate computed on real J blocks (g max + geo-mean) vs the L1 exact-resolvent oracle.
3. C4 multiscale — optional/stretch.
4. C2-insert (v2) — cheap now (bidir substrate is relative-PE); optional depending on scope.
