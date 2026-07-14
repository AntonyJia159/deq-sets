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

## 4. C2-bidirectional results — the σ_min/conditioning face, measured

`c2_bidir.py` on `bidir00–40`. The face where **conditioning governs reach** — but NOT (per §4b) the face
where the sharp Faber-on-FOV *rate* is certifiable; that abstains here. The certified reach bound is the
normality-free κ-route (Route A).

| Finding | Evidence | Status |
|---|---|---|
| **Envelope holds** on filler | ξ 0.29/0.41/0.51 hops vs proxy 4.6/6.8/6.3 (~10× conservative); certified via Route A (§4b), measured ~100× inside | the certificate claim — now a theorem, not just the proxy |
| **Conditioning governs reach** | filler ξ grows monotonically as σ_min falls | confirms σ_min thesis, bidir side |
| **Genuinely two-sided** | left-mass up to 0.44 (causal: 0 by construction) | the two-sided decay plot (`c2_bidir_profiles.npz`) |
| **ν _orders_ the faces** (does NOT license Faber on either) | ν_bidir 0.21–0.31 vs ν_causal 0.32–0.71 (gap40: 0.21 vs 0.71) | bidir is less non-normal, **but** 0∈W(I−J) on both (§4b) → say "less non-normal", never "within Faber's domain" |
| ~~Must-carry dissolves~~ **CORRECTED: must-carry PERSISTS** | irrelevant far/near 0.061 (bidir40) vs 0.068 (curr40) — nearly identical; the old contrast used a stale v2-era causal ξ≈27 vs final v5 ξ 0.75–1.82 | READONLY_Q makes readers invisible to the context → no query-awareness is *possible*; reader-set principle predicted this |
| **Maintenance channel** | warm 4 vs cold 14–22 evals on filler (3.5–5.5×); warm≈cold on relevant | cost ∝ solution movement |
| **Honest edge** | at σ_min=0.016 one seq near-multistable → filler gated "not measurable" | degrades honestly, same as causal face |

**The substrate has a bonus property:** ρ stays **< 1 throughout** (0.43→0.87) while σ_min spans 15× — the
bidirectional relay meets in the middle (half the effective depth), so it lives natively in the **κ/σ_min
domain** of validity. (Causal needed ρ up to 8.4 for comparable reach.) *But note §4b:* ρ<1 does **not** put it
in the *Faber-FOV* domain — the numerical range is huge (w(J)≈3, w/ρ≈4×) despite the small spectral radius.

### 4b. Certified reach bounds — the envelope is a theorem, but only the crude one survives (2026-07-08)

`faber_xi(κ)` = proxy: the √κ Chebyshev rate is a theorem only for **Hermitian** banded M; our (I−J) is
near-normal-ISH (ν≈0.3), not Hermitian — as computed it was a *heuristic*, and ν small does not repair it.
Two genuine bounds added to `c2_bidir.py` (`route_a_xi`, `route_b_xi`), run on the real Jacobians:

| ckpt | κ | ν | Route A (certified, hops) | Route B (sharp FOV) | proxy √κ | measured filler ξ |
|---|---|---|---|---|---|---|
| bidir08 | 84.7 | 0.30 | **42.4** | abstains (∞) | 4.6 | 0.29 |
| bidir16 | 186 | 0.30 | **93.2** | abstains (∞) | 6.8 | 0.41 |
| bidir24 | 159 | 0.31 | **79.4** | abstains (∞) | 6.3 | 0.51 |
| bidir40 | 445 | 0.21 | **222** | abstains (∞) | 10.5 | <noise |

- **Route A (certified, normality-free)** — DMS via the normal equations (M⁻¹=M\*(MM\*)⁻¹; MM\* Hermitian PD,
  cond κ²). Finite, loose; measured reach ~100× inside. **"Envelope holds" is now a theorem** via the crude
  κ-route. Ordering: **certified-A ≫ proxy ≫ measured** (proxy = good *predictor*, not a bound).
- **Route B (sharp Faber-on-FOV + Crouzeix 1+√2)** — **abstains on BOTH faces.** +1∈W(J) everywhere:
  verified ρ(J)≈0.74 but Re W(J)≈3.1, ‖J‖≈5–6.5, numerical radius w/ρ≈4×. The trained equilibrium transformer
  has a **huge field of values despite ρ<1**, so the singularity is swallowed and the sharp rate is defeated
  by non-normality even at ν≈0.3. **This killed a prediction (Route-B splits the faces) — a 4th clever
  prediction dying on measurement; the crude conditioning object survives** (pattern of §10/§11).
- **Corrects Note #9 / earlier framing:** "Faber is proper on the bidirectional face" is over-stated. What is
  proper there: *σ_min/conditioning governs reach* (Route A certifies). The Faber-FOV *rate* is not
  certifiable in this model class (0∈W(I−J)). Referee-proofing: we checked 0∉W(I−J); it fails; we fall back to
  the sound κ bound. Log: `checkpoints/c2_bidir_certified_log.txt`.

### 4c. The TIGHT a-priori certificate — block-transfer rate + adapted norm (2026-07-08, `c2_weighted_cert.py`)

The scalar bounds (proxy, Route A, Route B) are all vacuous/abstaining because they are worst-case over the
whole operator, blind to structure. The tight object: re-block (I−J) into w-token windows (block-tridiagonal),
form the **block-Jacobi iteration matrix** G = −D⁻¹(M−D) (D=blockdiag(M)); reach is governed by **ρ(G)**, the
cross-window transport rate. Measured (`block_transfer` + `c2_weighted_cert`):

| ckpt | ρ(G) bracket | ‖G‖ (transient) | exact resolvent per-hop | certified reach (tightest) @ const | measured filler ξ | Route A |
|---|---|---|---|---|---|---|
| bidir16 | (0, 0.35] | 5.6 (17×) | 0.36 | **0.95 hops @ 93** (mid 1.96 @ 16; loose 19.5 @ 7.1) | 0.41 | 93 (vacuous) |
| bidir24 | (0.40, 0.45] | 7.9 (19×) | — | **1.25 hops @ 81** | 0.51 | 79 |
| bidir40 | (0.80, 0.90] | 22 (26×) | 0.78 | **9.49 hops @ 61** (near-singular, honest) | <noise | 222 |
| curr24 (causal) | **0 (nilpotent)** | 5.6 | 0.55 | NILPOTENT → exact product ≤3 hops | — | — |
| curr40 (causal) | **0 (nilpotent)** | 6+ | — | NILPOTENT → exact product ≤4 hops | — | — |

The **ρ(G) bracket is read rigorously off the sweep** (largest divergent r < ρ(G) ≤ smallest valid r) and
validates the dense-eigenvalue values (bidir24 true 0.42∈(0.40,0.45]; bidir40 0.85∈(0.80,0.90]). Const =
√λmax(P) ≥ √κ(P) (since P⪰I), early-stopped (tighter than full convergence).

**The certificate:** ρ(G) is 10–100× tighter than the scalar bounds and tracks the exact resolvent, but
‖G‖≈5–22 (transient growth = non-normality) defeats a norm bound (‖G‖<1) — same disease as Route B. Fix =
**adapted (Stein/Lyapunov) norm**: for target rate r∈(ρ(G),1), P solves (G/r)\*P(G/r)−P=−I, giving
**‖Gᵏ‖₂ ≤ √κ(P)·rᵏ** → certified reach ξ=1/ln(1/r) hops, amplitude C=√κ(P)‖D⁻¹‖/(1−r). Rate r is tight
(→ρ(G)); non-normality is quarantined into the one-time √κ(P). bidir16 tradeoff (all certified): r=0.38→1.04
hops @ 57; 0.48→1.37 @ 26; 0.66→2.44 @ 13; 0.90→8.78 @ 7.7. **A-priori rigor recovered: certified ~1 hop
where well-conditioned, growing honestly near-singular, vs the vacuous 80–220 hop scalar Route A.**

**LINEAGE SHIFT + UNIFICATION (the important structural point):** this LEAVES approximation theory
(DMS/Benzi-Golub/Faber = "how a banded inverse decays") and JOINS dynamical-systems/Lyapunov (ρ of an
iteration operator + adapted-norm certificate for the transient; Stein eq; Kreiss) — **the causal face's OWN
lineage.** Causal M block-lower-triangular → G nilpotent (ρ(G)=0, confirmed curr24) → series terminates →
exact product-Lyapunov. Bidir M block-tridiagonal → ρ(G)∈(0,1) → geometric, certified. **The two faces are
two regimes of ONE object G**, not two disconnected proof families; Faber/DMS demoted to the loose scalar
floor. **"Conditioning not contraction" sharpened:** ρ(G)<1 = spatial-coupling contraction of the resolvent's
iteration, NOT temporal contraction of f (ρ(J) still to 8.4); two distinct axes govern, neither is ρ(J):
ρ(G) (spatial → a-priori reach) and σ_min (conditioning → a-posteriori error). Log `c2_weighted_final.txt`.
**Sweep COMPLETE (2026-07-08):** all checkpoints done after fixing the CPU-bound path — GPU-native
(vector-Gelfand ρ, early-stopped Gramian, power-iteration λmax; fp32 matmuls; ρ(G) bracketed off the sweep
so no reliance on a fragile estimate; block inverse kept fp64). Remaining theory: formalize √λmax(P) scaling;
scale story = Arnoldi ρ(G) + low-rank/block Stein solvers (deployment uses a-posteriori resid/σ_min anyway).

**BLOCK-DIAGONAL STEIN → a FREE sound directional charge + the causal↔bidir unification (2026-07-13,
`c2e_blockdiag_stein.py`, ~180 edits over the bidir ladder, one common certified rate/checkpoint).** Restricting
the Stein metric to block-diagonal-by-window `P=blkdiag(P_1..P_m)` is the *directional/per-source* refinement of
§4c. **The free half (Fable):** an edit δh supported in window j gives `u*Pu = u_j*P_jj u_j` EXACTLY
(off-diagonal blocks drop; u=D⁻¹δh), so the diagonal block of the ALREADY-CACHED dense Gramian yields a certified,
source+direction-anisotropic charge `√(u_j*P_jj u_j)·eff^d0/(1−eff)` at **zero new solve** — the certified cousin of
C2d's exact-but-heuristic `‖F_p δh‖`. MEASURED: (1) **sound universally** — `sound_far ≥ pred_far` with **0
violations on every checkpoint** (and global ≥ sound likewise); it upper-bounds the exact linear response, so the
tight directional tier becomes SOUND. (2) **anisotropy pays** — per-window √λmax(P_jj) spans 5.5–150×; using P_jj
vs whole-matrix √λmax(P) tightens ≈**5–22×** (median ~11×), sitting only ~6–20× above the exact response on
well-conditioned cells. (3) **recovers the DIRECTION half of the taxonomy, not the READER half** — cleanly ranks
*filler* (inert direction) 1–2 orders below the rest on all 5 checkpoints that have it, but does NOT reliably order
*irrelevant* vs *relevant* (monotone 3/8): that split is a reader/routing property P_jj is structurally blind to →
**empirically hands that job to DWR** (source-anisotropic a-priori & reader-blind here; DWR reader-anisotropic
a-posteriori & edit-blind — complements at opposite ends, now MEASURED not asserted). (4) **certifiable rate tracks
conditioning** — common r rises 0.55 (well-cond) → 0.90–0.95 (near-singular); stiffest cell (bidirnp40, σ_min=0.024)
admits NO common rate on the grid = loose exactly where conditioning is bad. **THE UNIFICATION (paper remark
regardless of the LMI):** block-diagonal Stein is the *family* containing both faces — causal G is strictly
block-bidiagonal so the inequality DECOUPLES into a backward recursion `P_i = I + T_i*P_{i+1}T_i/r²` (no SDP, always
feasible) which **IS** the product-Lyapunov cert of note #9; bidir is the genuine LMI (chordal SDP). So "one operator
G, two regimes" sharpens to **"one certificate family, two solution modes"** (recursion vs LMI); dense-P §4c is the
isotropic special case. Full-block LMI = one timeboxed chordal-SDP check (feasibility shrinks near ρ(G) —
diagonal-stability gap — but §3 says the *constant* dominates at d=1–3, so the free diagonal-block charge already
captures most of the prize). Folded into note #11 §1(h)–(i) + the certificate-map "sound directional charge" row.

---

## 5. The reader-set principle — the correct general statement of must-carry

**Supersedes "causal carries, bidirectional doesn't."** Three edit tiers, three *logical statuses*:
- **Queried-value:** transport to the cursor is **information-theoretically necessary in both faces** (if the
  answer changes it must arrive). Measured ridge far/near ~0.09–0.10 both faces. Nothing forgets these, ever.
- **Filler:** never carried, either face (the fair envelope witness).
- **Unqueried-value:** the tier where faces *can* diverge — **impossibility vs. permitted capability, not two
  guarantees.** Causally, carry is *forced* (the relay can't condition on future queries → must keep every
  binding; availability argument, architecture-level, theorem-flavored). Bidirectionally, selective forgetting
  is *permitted only if the readers are attendable*. **CORRECTION (2026-07-07, caught while designing C2t):**
  our trained bidir substrate has READONLY_Q — context tokens cannot attend to queries — so its context
  equilibrium is *independent of what the queries ask* and **cannot** be query-aware; and indeed its measured
  irrelevant-edit transport ≈ causal (far/near 0.061 vs 0.068 at the matched endpoint; the earlier "dissolves"
  contrast mistakenly compared against a stale v2-era causal ξ≈27 instead of final v5 ξ 0.75–1.82). This is the
  reader-set principle *working*: invisible readers force carry, in any architecture. Whether a QUERY-VISIBLE
  bidirectional substrate (readonly off + window curriculum — untested combination; round-4's readonly-off runs
  predate the curriculum) trains, and whether its irrelevant-edit transport then actually drops, is exactly what
  the bidirqv retrain + C2t measure.

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

**DWR / READER-SET BOUND — MEASURED, the keystone run before drafting (2026-07-13, `colored_dwr_insert.py`,
exp #3; colored-recall INSERTS, 16 seqs×4 classes×2 modes).** The escape row of the note-#11 certificate map was
cited but never run. colored-recall gives the EXACT stripe rule (an inserted (color,value) flips reader q IFF
q's color AND newly-selected: more-recent for latest / earlier for earliest; else q invariant), so the
goal-adjoint `w_reader·source` has an exact binary to check. Objects (reader q at re-solved fp, head H, residual
r of an aligned warm start): reader bound `‖H·R[q,:]‖·‖r‖` vs global `‖H‖·‖r‖/σ_min`. RESULTS: (1) **SOUND +
TIGHTER — the reviewer's "why didn't you run the adjoint bound" answered:** ladder actual < dwr_est(~1.5–7×) <
bound_reader(~10×) < bound_global(~100×), **100% sound** every insert; reader-restriction **6–12× tighter** than
global ‖r‖/σ_min (that gap = the far-field slack 1/σ_min charges blindly). (2) **reader-invariance to off-stripe
inserts certified;** goal-adjoint recovers the stripe cleanly on long-range earliest (in/off 8.6×) but only at
COLOR granularity on recency latest (2.5×: filler/other-color screened, shadowed same-color NOT separated from
selected). Mechanism: the linear adjoint is a color-stripe REACH detector; which same-color write is SELECTED is
a nonlinear pick the base-fp adjoint smears → invariance to a shadowed insert is carried by the small RESIDUAL,
not the adjoint. (3) **CHEAP/CACHED (the deployment object):** reuse the PRE-insert reader adjoint (base R's
reader row, aligned) + recompute only the 1-eval residual → the **sound bound SURVIVES** (100% sound, ≈oracle
tightness; cache the expensive adjoint, refresh only r, no R2 re-solve). But the tight point-estimate + flip
DISCRIMINATION do NOT: cached adjoint under-predicts a flip ~2× and in/off collapses to 0.9× on recency (3.6× on
write-once). So the sound BOUND is cheap; accurate flip magnitude/discrimination wants a fresh or
bordering-corrected adjoint. CAVEAT: cheap-bound soundness is empirical (adjoint-norm stability under a local
edit), not yet a theorem — a Schur/bordering correction would make it rigorous. This VALIDATES the
division-of-labor framing above: the reader-restricted bound is the sound guarantee; the stripe/selection is
measured mechanism. Folded into note #11 §4a + certificate-map row (measured) + skeleton tier-2.

**BORDERING/SCHUR CHEAP ADJOINT — the follow-up, a clean NEGATIVE that corrects an overclaim (2026-07-13,
`colored_dwr_bordering.py`).** Q: can a Schur/Woodbury correction make the cached cheap adjoint rigorous AND
length-independent? A: **NO.** The Woodbury algebra is exact (reader-row recon **1e-15** when the local set S spans
the support), BUT the insert's operator change is **NON-LOCAL** at the converged fixed point: ΔM=J(z₂*)−J(z*) keeps
**~45% of its mass beyond ±8 positions**, 17% beyond ±16, and vanishes only at S=whole sequence. Cause = STATE
DEPENDENCE: J depends on z, and the equilibrium shifts everywhere the insert *reaches* (rate ρ(G)), so ΔM is global,
not window-local → exact bordering needs S≈L → **no O(Wd) rigorous cheap correction exists** (refutes the "bordering
makes it rigorous" line I'd written — corrected in note #11 §4a + skeleton #4). What the **warm-local** correction
(truncate ΔM to a window, J(z_warm)) DOES buy: recovers flip **discrimination** on long-range earliest (naive 0× →
schur **13–20×** ≈ oracle 13–29×), 100% **sound**, but reconstructs the WARM operator not R₂ (recon 20–990% on big
flips) → a cheap **approximate answer-metering** signal, NOT a rigorous exact resolvent. Recency (latest) stays weak
(~1×: nonlinear selection). Efficiency: since exact support is global there's no clean O(1)-in-L win for a rigorous
bound; the approximate local LA clocked 3.6–7.3× faster than re-solve (growing with L) but at toy L~2W the support
fills 60–80% of the seq. **NET (honest): cheap sound bound (naive cached adjoint) + cheap approximate long-range
discrimination (warm-bordering); rigorous exact = re-solve. Structural-edit resolvent updates do NOT localise** —
a genuine result (the reach ρ(G)^dist bounds the RESPONSE decay, but the state-dependent JACOBIAN CHANGE spreads
globally). Better found pre-draft. Folded: note #11 §4a, skeleton #4, this entry.

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
  it really is global. **CORRECTION (this run):** the substrate is **not actually relative-PE yet** — `h0 =
  emb + posw[:L]` still adds a *learned absolute* positional embedding; the rel-bias was added *on top* for the
  binding fix, `posw` was never removed. Invisible for substitutions (no index shift), but it would smear an
  insert's response globally. See §11 (posw-ablation check + the aligned-frame reduction).
- **The Excel/NCA anecdote (ZJ):** linearly-interpolated Wolfram CA collapsing to gray 0.5 = consensus
  collapse; discrete CA stays interesting because thresholding is maximally non-averaging (= peaked softmax
  re-sharpening). One-line inhabitant of the NCA corner of the four-lands figure: continuous-CA gray-out =
  oversmoothing = rank collapse — one phenomenon, three fields' names.

---

## 10. The causal claw-back ladder + deferred-billing metering

The central honesty (§5 / blueprint) says causal edits aren't *local*. It does **not** say they aren't
*adaptively priced* — different claim, and it survives. Three tiers, decreasing guarantee strength:

- **Tier 1 — certified, a priori, DIRECTIONAL (small new experiment, = the product-form debt).** The scalar
  ξ-ball is vacuous because σ_min reports the *carry* direction and charges every edit as if its whole δh lay
  there. Refine scalar → projection: precompute the low-rank **carry subspace** (top singular directions of the
  accumulated per-hop transfer product `∏(I−D_i)⁻¹A_i`) once per context; δh is known *before* solving (the
  embedding delta at the edit sites), so project it — transverse component gets a certified short-ξ⊥ screened
  bound, carry component is "transported at gain≈1 but **rank-r**." Certified recompute set becomes *ξ⊥-ball +
  rank-r carry update* instead of "the whole suffix," and low-carry-projection edits get an a-priori
  containment verdict. Doesn't shrink for genuinely carry-exciting edits (they *do* reach the cursor — correct);
  it stops charging *every* edit for the carry. Validatable against the measured 3-tier far/near table.
- **Tier 2 — certified, a posteriori (DONE, C2 v4.1).** Even when the a-priori ball is vacuous, `‖z−z*‖ ≤
  resid/σ_min` holds at every point of a warm re-solve → any *partial* recompute carries a certified error bar;
  stop when `resid/σ_min < tol` and the result is provably close. The "third face" of σ_min, now stated as the
  causal face's a-posteriori guarantee.
- **Tier 3 — emergent metering (MEASURED; REVISED by C2m — see §10 result block): a BIDIRECTIONAL property.**
  Warm-start iterations track `‖z*_new − z*_old‖` cleanly on the near-normal bidirectional face (Spearman
  ~0.9, output-sensitive billing as a law) but only weakly on the causal face (mode-alignment dominates:
  carry-aligned movement = slow modes = disproportionate cost; absent at near-singular). The coarse 3-class
  ordering (filler ~4 → unqueried ~11–18 → queried ~22≈cold) still holds causally, but as a *step*, not a law.
  Feedforward/cold recompute is **input-sensitive** (flat toll, confirmed). Tier 3 sits inside the tier-2
  bound, so the composite stays sound — but on the causal face the reliable instruments are tiers 1–2 only.

**Deferred-billing / lazy-activation reading (ZJ, the resolution of "why edit if nothing reached the
cursor?").** Most edits are a **quiet build-up of relationships, stored locally, awaiting a future trigger
that activates them** — a renamed variable, a fixed config value, an updated fact: meaningful to *some*
eventual reader, inert for the next decode step. Metering **bills you when the reader arrives and excites the
carry**, not at write time. So (a) "meaningful" is reader-relative and time-distributed (the paradox
dissolves — the reader-set principle from the other side); (b) a cheap "nothing propagated" verdict is *itself
the product* (incremental-build value = confirming most outputs unchanged, cheaply, with a certified residual);
(c) the write→trigger gradient (single-digit → moderate → heavy iters, keyed to how much a reader activates) is
exactly the desired billing curve, and it's emergent — the solver discovers it without classifying anything.

**Deferred billing — C2t RESULT (MEASURED, NEGATIVE; report straight, do not spin).** The lazy-evaluation
prediction **failed**. Clean signal: **write-cost(edit unqueried value) is reader-INDEPENDENT on all three
substrates** — causal 7.6≈7.8, readonly 12.9≈13.0, query-visible 15.3≈15.7 (lazy≈eager iters). So the relay
carries unqueried bindings regardless of reader presence: **must-carry is empirically robust, and selective
forgetting did NOT emerge even where architecturally permitted (query-visible).** This is "emergent not
certified" coming back negative — nothing in the recall loss rewards selectivity (and the query-visible model
trained worse, recall 0.63@gap40). Reader visibility made query-retarget *more* expensive (readonly trigger ~4
iters vs query-visible ~11 — visible queries couple the reader into the context equilibrium), the *opposite* of
a laziness win: visibility costs iterations AND trainability. Path-independence held cleanly (totals conserved,
final states agree 1e-7 = a warm-start-exactness sub-result). **Measurement lesson:** the designed
dz@reader-position observable is confounded — retargeting a query changes the query token's own embedding (a
direct state change, not transport) — so reader-independence of *write-cost* is the clean signal.
**What survives:** tier-3 metering (cost ∝ realized ‖Δz‖ — contained edits cheaper than transporting ones)
holds qualitatively; the "two faces = eager vs lazy evaluation strategies" framing is **demoted** to "lazy is
permitted but not incentivized; these trained models are all eager."

**C2m RESULT (`c2m_metering.py`, curr+bidir 16/24/40; real edits + synthetic carry/transverse perturbations;
cold solves = the flat toll; ‖R·δh‖ = pre-solve forecast) — metering is REAL but FACE-DEPENDENT, inverting the
load-bearing assignment:**
- **Bidirectional face gets the clean law:** Spearman(n_warm, ‖Δz‖) = **0.90/0.92/0.89**, slopes 4.3–14.5
  evals/decade, partial corr(n, ‖δh‖ | ‖Δz‖) ≈ 0 (−0.10/+0.18 at bidir24/40) — cost meters output magnitude,
  input norm adds nothing. Output-sensitive billing, as a law.
- **Causal face: weak and mode-confounded** (0.67/0.65 with strongly *negative* partials −0.49/−0.66 —
  for matched movement, small-input/large-output = carry-aligned = slow modes = MORE iterations), and
  **absent at near-singular** (curr40: Spearman 0.09). Cost there is set by *which modes* moved, not how much.
- **Mechanism = ν, a third time:** near-normal J (bidir) has uniform per-mode convergence rates → magnitude
  metering; maximally non-normal causal J has wildly varying mode rates (per-hop ‖T_k‖ up to 25) → raw ‖Δz‖
  under-determines cost. The faces differ in proof family, trainability, and now **billing legibility**.
- **SOLVER-INDEPENDENCE CONFIRMED (2026-07-12, `c2m_metering_broyden.py`, n=126/face) — a predicted deflation that
  DIED on measurement.** Hypothesis (mine): the causal weakness is Anderson's spectral gap, so the affine-invariant
  Broyden solver should clean it up → "metering is bidirectional" would be a 2nd Anderson artifact (after "init
  doesn't help"). MEASURED opposite: **causal Spearman(n_warm,‖Δz‖)≈0.70 vs bidir≈0.90 under BOTH Anderson AND
  Broyden** (per-ckpt 0.68–0.77 causal, 0.88–0.92 bidir); causal partial-corr(n,‖dh‖|‖Δz‖) stays negative (−0.51
  Anderson → −0.35 Broyden), bidir ~0 both; cold flat both. So the **billing-legibility divergence is a genuine ν
  property, solver-independent** — it stands. MECHANISM REFINED (spectral-gap was wrong): Broyden is affine-invariant,
  so its iteration count tracks the operator's *intrinsic* difficulty, NOT ‖Δz‖ in the standard norm; ‖Δz‖ is a
  faithful cost proxy only when standard norm ≈ the operator's natural geometry (= near-normal = bidir). On the
  non-normal causal face ‖Δz‖ under-determines cost for ANY solver. (Broyden DID soften the causal partial and halve
  cold cost 34→19 evals, so a *small* mode-confounding component was spectral-gap; the Spearman gap is untouched.)
  Records `checkpoints/c2m_broyden_records.npz`, log `c2m_broyden_log.txt`. Same face-split root as ν → reinforces,
  not complicates, the two-faces story.
- **CLAW-BACK LADDER REVISION (honest):** tier-3 emergent metering is a **bidirectional** property, not the
  causal consolation prize I claimed. The causal face's reliable instruments are tier-1 (directional
  classification, C2d) and tier-2 (a-posteriori bound) only.
- **The universal winner:** the pre-solve forecast ‖R·δh‖ → ‖Δz‖ at Spearman **0.96–1.00 on both faces at all
  conditioning** — the single most robust relationship in the campaign (linear response, validated a third
  way). Flat toll confirmed everywhere (cold ≈ constant 15–58 evals, Spearman ≈ 0).
- Caveats: 18–36% of points at the few-eval floor (metering is coarse at the small end); slope-vs-σ_min
  prediction inconclusive (don't claim); curr40 rows inherit that checkpoint's multistability.
  Records: `checkpoints/c2m_records.npz` (the n_warm-vs-‖Δz‖ scatter with cold flat line = paper figure). **Bonus finding:** the query-visible
substrate is itself a **visibility↔trainability tension** (recall 0.94→0.63 at gap 40 vs readonly — making
readers attendable, to enable selectivity, degrades long-relay trainability).

**C2d RESULTS (2026-07-07, `c2d_directional.py` on curr16/24/40 — the directional certificate, MEASURED).**
Oracle = exact resolvent; far-reach map F_p = R[far rows, block col p]; pred_far = ‖F_p·δh‖ computed
**pre-solve**. Five validations:
- **V1 linearity:** log-log corr(pred, meas) per-position profiles: **0.984 / 0.981 / 0.927** mean (min 0.91 /
  0.83 / 0.33) — first-order reasoning survives finite token substitutions; degrades only at σ_min=0.028
  (curr40, where multistability lives).
- **V2 a-priori taxonomy:** monotone at every ckpt and *quantitatively* close — curr24: filler 4.3e-3 (meas
  3.1e-3), irrelevant 3.39 (3.42), relevant 8.5 (10.8). **Three orders of magnitude of class separation
  predicted from δh alone, before any solving.**
- **V3 soundness:** meas/pred median 0.86–1.03; first-order violations (meas>2×pred) 0/11, 2/26, 2/26 —
  the violators are *relevant* edits at near-singular conditioning (max ratio ~7×: nonlinear amplification of
  carry-exciting edits). **FALSE CONTAINMENTS: 0 everywhere** (all predicted-contained edits truly contained) —
  the safety-critical direction is clean.
- **V4 low-rank carry:** effective rank **7.1 / 7.7 / 8.3 of d=64**, stable across conditioning — the carry is
  ~rank-8 (≈2× the task's 4 bindings). The "rank-r update, not full-suffix recompute" claim is real.
- **V5 product form (discharges the old debt):** the coarse w-window T-product reconstructs the exact resolvent
  block at **relerr ~1.5e-15** at all three ckpts — the re-blocking theorem operationalized on real trained
  Jacobians. The **scalar** norm-product bound's slack vs the directional product: 7.5× at 2 hops, and at
  curr40 **25× → 395× → 764×** over hops 2–4, with per-hop ‖T_k‖ up to 25.4 — every per-hop *norm* exceeds 1
  (scalar bound predicts growth = vacuous) while the directional product decays. **Direction is not a
  refinement; at trained conditioning it is the entire content of the certificate.**
**Honest scope:** the directional object is a *predictor and classifier* (flawless as a classifier: taxonomy +
zero false containments), **not a certified upper bound** for carry-exciting edits at near-singular
conditioning (2–7× underprediction there). For a bound, either attach the measured nonlinearity margin or use
the singular-value split (σ₁‖P_carry δh‖ + σ_{r+1}‖P⊥δh‖); the *containment* verdict — the decision that saves
work — had zero failures.

**Certificate verdict + caveats.** Worth building at **one-subsection scale** (completes the causal ladder;
same computation as the product-form debt; one validation run). Harness hook worth **one paragraph**: δh known
pre-solve + carry basis precomputed at cache-build → O(r·d) test *before* paying (small carry projection → patch
the ξ⊥-ball, keep the downstream cache; large → schedule the full warm re-solve; tier-2 certifies whatever
partial work you do) — the role dependency summaries play in incremental compilers. **Caveats to attach:** it's
a **first-order** certificate (linearization at z*), and a token substitution is a *finite* perturbation — our
C2 envelopes already survived that gap empirically, but the directional refinement needs the same style of
validation, not just derivation; **rank choice** and **subspace stability across contexts** are the
reviewer-attack surfaces (measure subspace stability as part of validation).

---

## 11. Multi-token & insert/delete — directional theory + the posw finding

**Multi-token edits: superposition + one cheap experiment.** First-order theory is clean: stack the per-site
δh's, project jointly onto the carry subspace — transverse balls union, carry components combine (and can
**cancel** — a rename edits many sites coherently, its carry projection may interfere destructively). What
breaks superposition is the **nonlinear attention re-routing** (two edits jointly flip an attention decision
neither flips alone). → **Edit-interference experiment** (cheap, same C2 machinery, one new loop): response to
paired edits vs the sum of single-edit responses, as a function of separation — quantifies where the linear
regime ends = the validity boundary of the whole certificate story.

**Insert/delete: theory reduces cleanly.** In the **aligned frame** (match unchanged prefix + suffix), an
insert/delete under relative PE + banded attention **is** a multi-site substitution confined to the width-w
band straddling the cut (far regions shift uniformly → attention byte-identical). So directionally it's just
the multi-token case at the cut; no new theory. Insert/delete = "a width-w edit at the cut," two-sided screened
shadow prices the rest.

**BUT the substrate isn't relative-PE — posw is load-bearing (measured, `posw_ablation.py`).** `h0 = emb +
posw[:L]` still adds a *learned absolute* PE. Ablation (zero posw at eval): recall holds at gap 0 (1.000) but
**COLLAPSES at every gap > 0** (0.997→0.489, 0.987→0.487, 0.995→0.525, 0.938→0.399), and `‖posw‖` **grows with
gap** (2.84→5.48→7.91→9.68→11.15) — *the model recruits absolute position harder as the relay lengthens.* So
absolute PE is doing real cross-window relay work, not bookkeeping; an insert experiment on this substrate would
measure the absolute-PE artifact, not the screened shadow. There's also a genuine research question underneath:
*can a pure-relative-PE banded DEQ do the cross-window relay at all, or does the relay lean on an absolute
coordinate?* (The growing ‖posw‖ hints the latter — worth knowing, don't over-interpret from one probe.)

**RESOLVED — pure-relative RELAYS (`curriculum_bidir_noposw.py`, bidirnp00–40).** Recall
1.000/0.987/0.912/0.937/0.819 at gaps 0–40 (vs 1.000/0.997/0.987/0.995/0.938 with posw) — **no collapse
anywhere**, so the ablation collapse was about the trained-with-posw model's *learned reliance*, not a necessity
of the model class. Absolute PE is a **crutch, not a load-bearing wall**: it buys a few recall points and a
smoother gap-16 stage (the pure-relative run briefly pushed ρ to 1.35 with loose resid 3.7e-2 there before
recovering — treat the bidirnp16 spectrum row with suspicion), but the relay runs on relative position alone.
**Insert/delete is unblocked in principle**; remaining work for a v2 measurement: insert-type `apply_edit` with
alignment bookkeeping, measured in the aligned frame vs the "band at the cut" prediction, on the bidirnp
checkpoints. (Optionally re-run C2-bidir on bidirnp for full substrate consistency.)

**INSERT/DELETE MEASURED (`c2_insertdelete.py`, 2026-07-08).** Null filler insert/delete in the filler region,
aligned-frame diff (insert: old i≥c ↔ new i+1; delete: old i>c ↔ new i−1), normalized by the per-position
state norm. **POSITIONAL-SHADOW ratio = insert far-field / substitution far-field** (same-arch, PE the only
difference — the clean isolate) at gap40 (n=8, where the sequence is long enough to have a far field beyond
2W; **gap24 is too short — nothing sits past 2W of the cut, far_rel≡0, uninformative**):
- **bidir40 (bidir+ABS PE): ratio 70×** — a semantically-null insert casts a downstream far-field 70× a
  substitution's = the **dense positional-reindex shadow** (every shifted token's absolute h0 changed;
  downstream-dense 0.84).
- **bidirnp40 (bidir+REL PE): ratio 2×** — insert ≈ substitution-at-cut. **Relative PE cuts the insert shadow
  ~33× (70→2): the aligned-frame reduction, confirmed and isolated to the PE choice.**
- curr40 (causal+ABS): ratio 2× — small, but confounded (near-singular gap40 → substitutions already reach far,
  so the ratio understates; the clean causal test wants a well-conditioned **causal+relative** checkpoint,
  `currnp`, which we don't have — owed).
- **Far-field RANK 2–6 of d=64 across all** — structural-edit shadows compress to the carry 'highway' →
  certifiable via a rank-r update, same as substitutions.
- **Warm-start < cold** at gap40 (curr 25/53, bidir 18/34, bidirnp 38/64) — structural edits are maintainable
  via the aligned-frame warm start (splice a fresh slot at the cut, keep the rest).
- Causal (curr) **up_rel ≈ 0** — insert shadow is strictly downstream (causal structure preserved).

CAVEATS (state them): gap40 checkpoints are near-singular so the far field is thin (few positions past 2W) and
absolute magnitudes are large — the **ratio** is the clean signal, not the magnitudes; n=8; bidirnp24 warm>cold
(68/44) is the known near-singular outlier. Clean causal-relative test (`currnp` retrain) and a longer-context
(gap≥60) substrate with a thicker far field are the owed strengtheners. **Verdict: your "heavy causal shadow"
expectation is right for ABSOLUTE PE (70×); relative PE reduces insert/delete to a certifiable, low-rank,
maintainable width-w edit at the cut — the aligned-frame reduction is real.**

**INSERT/DELETE AS OPERATOR BORDERING — the machinery, the warm-start correction, the RAG metering hook
(2026-07-12, framework + planned experiments).** The empirical §11 shadow now has an exact algebraic home. A
VALUE edit is `δh` in a fixed space (the resolvent eats it); a STRUCTURAL edit changes the operator's SIZE, and
the native language is **matrix bordering + Schur complement**: an insert borders `M=(I−J)` with a new row/column
`(b, cᵀ, δ)` = the new token's couplings; the old-block correction is `−M⁻¹b·S⁻¹·(…)` with `S = δ − cᵀM⁻¹b` the
Schur complement (a rank-≤d update to the cached `R`). Delete = the dual downdating `(M_AA)⁻¹ = R_AA −
R_Ab(R_bb)⁻¹R_bA` (subtract every resolvent path routed through the deleted token; hub-ness = size of `R_{·b}`).
**All tiers inherit** because insert = (local band update `ΔM_band` at the cut, = the §11 renormalization/window-
churn, handled as a multi-site substitution) + (rank-d bordering sourced at the cut): tier-1 reach decays
`ρ(G)^{dist(cut)}` (the correction IS a resolvent column sourced at the cut = the measured rank 2–6 far field);
tier-2 residual bound holds on the new system, with the NEW conditioning risk carried SOLELY by `S`
(`det M'=det M·det S` → `σ_min(S)` is the new-token stiffness gate, from LOCAL blocks); directional/Woodbury
updates `R` by the rank-d Schur; **DWR reader-invariance** = `w_reader·(bordering source)` certifies whether the
edit changes any reader's output (a single inner product, no re-solve). Locality is genuine only in
**token-attached block** coordinates — index-attached blocking manufactures a fake all-blocks-changed (the same
relative-vs-absolute-PE lesson, one level up: structural edits are local in every structure attached to CONTENT,
global in every structure attached to COORDINATES). Lineage to cite: bordering/Schur (num LA); prolongation–
restriction + DWR (adaptive mesh refinement, DWR's home); Fock-space / domain-perturbation (general "change the
number/shape of inputs").
- **WARM-START CORRECTION (I overstated "no warm start"):** insert/delete DO have a warm start = the aligned-frame
  copy (old states at shifted positions + init the new slot); only the cut's ξ-ball re-settles. Woodbury-bordering
  is the "warmer-than-warm" PREDICTOR on top → the SAME warm-vs-warmer question as value edits, not warm-vs-cold.
- **DOES WARMER-THAN-WARM BEAT WARM ALL-IN?** Conditional: the predictor is a resolvent action `R·source`; LOCAL
  edit → only cached block-columns of `R` in the ξ-ball ≈ ONE local iteration's work → pays iff it saves >1 Broyden
  iter (toy: 15–30% = 2–4 iters). BUT only if `R` is AMORTIZED (cached across many edits to a fixed base);
  recompute-per-edit = a wash. **The amortized regime = insert/delete into a fixed base = RAG** — the mechanism
  that flips wash → net win (+ periodic `R` refresh for staleness). This is the honest answer to "can linear-
  prediction+Broyden beat plain warm start."
- **RAG EMERGENT METERING — a CAUSAL-FACE plus (corrected: NOT bidir-only).** Mainstream RAG = `[docs][query]` on
  a DECODER = CAUSAL with a KNOWN reader (query downstream, attends back). Cleaner on the causal face for 3 reasons:
  (i) **append free** (`b=0` bordering, prefix preserved — "no free append on bidir" from §6 is exactly this: free
  append is a CAUSAL property); (ii) **mid-insert one-sided** (downstream causal cone, prefix `<c` untouched, vs
  bidir two-sided); (iii) **query = known reader** → DWR applies causally DESPITE near-singularity (colored-recall
  83% precedent) → the causal DIAGNOSTIC register UPGRADES to a certificate. Metering: (a) compute = re-solve cost
  ∝ chunk ‖Δz‖; (b) DWR answer = `w_query·(chunk source)` = "does this doc change the answer" = attribution/pruning,
  sound+tight, no re-solve for inert. RAG = the convergence of the causal threads (cheap structural edits + known
  reader + amortized base). A DEQ context-ENCODER (bidir, Perceiver/cross-attn) is an ALTERNATIVE home, not primary.
  DEQ-RAG not deployed → mechanism only, speculative-but-concrete; the known-reader sub-regime the two-registers
  split (open-generation default) didn't cover → makes the causal face NOT only a diagnostic.
- **DEDICATED EXPERIMENTS (proposed — do a few):** (1) insert/delete **warm-vs-warmer ALL costs priced** (cold /
  aligned-warm / Woodbury-bordering under Broyden, COUNT the predictor's resolvent action, `R`-cached vs recomputed
  → the net-win question, answered); (2) **structural-edit emergent metering** (C2m for inserts: re-solve cost vs
  ‖Δz‖ Spearman by filler/relevant chunk); (3) **DWR reader-invariance for inserts on colored-recall** (insert a
  `(color,value)`; answer-flip ground truth KNOWN = same color & more recent; does `w_reader·source` predict it,
  sound+tight? — the cleanest test, known reader-set); (4) **bordering/Schur accuracy** (insert response =
  resolvent column sourced at cut, rank-d, `ρ(G)` decay — validate like C2d-V1/V5). Ties the reader-set/DWR
  keystone to a real use-case.

**Anchor-token contingency (recorded, NOT needed — keep in the drawer as an optional booster for the weak
gap-40 stage).** If pure-relative had failed (or to close the recall gap), a designated anchor token is the
minimal absolute scaffold, pleasing on four axes: (i) **BVP reading** — the bidirectional face is a
boundary-value problem and a translation-invariant band lacks boundary data; an anchor *is* the boundary value,
letting the equilibrium propagate a derived coordinate outward; (ii) it's how the **causal face gets position
free** (Haviv et al.: causal LMs learn position with no PE — the causal asymmetry is an implicit anchor at the
start); (iii) it's the **attention-sink/BOS** object real models invent spontaneously (= our C4 hub); (iv)
**Growing-NCA grows from a seed cell** for the same symmetry-breaking reason (four-lands echo). An anchor at
position 0 still supports the aligned-frame insert story for mid-context edits (inserts don't cross it), though
the suffix's distance-to-anchor shift arrives through the relay and is itself σ_min-screened — think through
before using.

**Scope caveats on the window-curriculum finding (state these ourselves before a reviewer does).**
(a) **What the curriculum actually encodes is thinner than it looks:** we did *not* encode "binding" — w=2 just
restricted connectivity and the binding hop emerged as the only loss-reducing path. The generic recipe is *grow
the context-context band, keep readout global*: small windows make the global-mixing shortcut **unavailable**
rather than merely unfavored, while local subcircuits stay learnable. Plausibly generic — for the same reason
the simplicity-bias literature says networks learn low-order structure first. (b) **Its honest boundary:** it
helps exactly when the task's enabling dynamics *decomposes locally* (small-window subcircuits exist and remain
useful as the band grows). A task whose minimal circuit is irreducibly global would be *starved* by a window
curriculum, and for a complex task you can't know in advance which case you're in — a genuine flexibility
constraint of the bidirectional method; name it plainly. (c) **The MLM-objective hypothesis (deflates the
blocker's generality — include it):** BERT-scale encoders train fine, and the reason may be the *objective*,
not scale. MLM supervises every masked position with a local cloze target — dense local gradients — whereas
MQAR supervises only sparse far-off queries at the tail. The blocker may be "sparse-readout objective ×
bidirectional mask," not the mask alone. Reader-set unification: **MLM works because it plants readers
everywhere** — every masked token is a local reader, so by the reader-set principle every position becomes a
supervised local-readout site; MLM *is* the curriculum, in effect. Scoped claim for the paper: blocker +
curriculum fix established for *minimal attention-only equilibrium cells with sparse readout*; the
MLM-objective hypothesis and the local-decomposability limit are the two open edges.

---

## 11b. Post-arc leads (registered end-of-arc, 2026-07-07; designed, NOT run)

- **Why is linear response this good? — mechanism hypothesis (probe #1, upgraded by ZJ's "transformers are
  secretly linear" pointer).** With attention patterns frozen, a transformer is LINEAR in the values (OV
  pathway; nonlinearity concentrates in QK/pattern formation — Elhage et al. framework; also a 2024
  "secretly linear" paper, ~0.99 inter-layer linearity — VERIFY before citing). Our block semantics says
  J = value transport (a_ij·Wv, linear) + re-routing (∂a/∂z, spiky). **Hypothesis: trained attention is
  saturated → ∂a/∂z ≈ 0 locally → edits that don't flip an attention decision propagate linearly; linear
  response fails exactly at attention TIES.** Testable: per-edit attention-pattern change should predict the
  linearity residual; plus an ε-sweep for the basin boundary. Explains the campaign's most robust result
  (forecast Spearman 0.96–1.00).
- **The collage connection (ZJ) — our a-posteriori bound is the conditioning upgrade of the collage theorem.**
  Collage/Banach: ‖z−z*‖ ≤ ‖f(z)−z‖/(1−c), global, needs contraction c<1. Ours: ‖z−z*‖ ≤ resid/σ_min(I−J),
  local (first-order, empirically validated), TIGHTER when contraction holds (σ_min ≥ 1−c) and VALID at ρ>1
  where collage is void — the paper's ρ→σ_min move in error-bound form. **Future work — "conditioned collage
  training":** ZJ's old collage-loss (supervise ‖f_θ(target)−target‖, never solve) fails silently near-singular
  (certifies nothing); train on the *certified* error resid/σ_min (or collage residual + σ_min-health
  regularizer) instead. Novelty flags: adjacent to Bai's Jacobian-Frobenius regularization (we'd regularize
  the certificate quantity, not a norm proxy) and to Jacobian-free/phantom training — run past Geng.
- **The semi-causal dial (C2ν, ZJ's confound catch — the missing control for "ν governs everything").** All
  ν-governs claims (proof family, trainability, billing legibility) are confounded: ν only varies through the
  mask, so "near-normal governs X" ≡ "bidirectional governs X" in our data. Control: an ASYMMETRIC band
  [i−w, i+βw], β ∈ {0, 0.25, 0.5, 1} — ν moves continuously with β; then metering legibility either tracks ν
  continuously (ν vindicated as the governing variable) or jumps at β>0 (strict triangularity is topologically
  special; ν was a proxy). Either outcome sharpens the claim. Practical mirror: streaming/ASR bounded-lookahead
  attention is literally semi-causal. Cost: one curriculum retrain + C2m pass per β → slated, not now.
- **Multistability: explicitly OUT OF SCOPE for this paper (ZJ decision).** Stays the natural v2/spin-off
  (branch-tracking, hysteresis/primed-branch probe, amplifier-coupling EoS thread from the dormant notes).
- **SPIN-OFF (GENG'S TERRITORY, NOT this paper; ZJ agrees 2026-07-08) — "incremental / streaming DEQ inference
  via certified low-rank warm-start prediction."** The inversion of our spine: instead of "the equilibrium
  *enables* certified maintenance," sell "the equilibrium *enables* cheap certified incremental inference." Same
  machinery (Woodbury low-rank resolvent predictor + §4 residual certificate), pitched on **inference cost**, not
  maintenance guarantees — it attacks DEQs' actual weakness (the iterative solve; a feedforward net has no
  iterate to warm). Squarely Geng's area (DEQ efficiency: Jacobian-free, phantom gradients; `torchdeq` is the
  vehicle). HONEST scale obstacles: predictor not free (need iterative M⁻¹U when J isn't materialized), cache
  staleness (J drifts across edits), overshoot near the well-posedness edge (the σ_min=0.028 miss), and large
  deployed DEQs don't yet exist. Separate systems/architecture project → **raise with Geng**, do NOT let it
  absorb the characterization paper. Prereq for credibility: the C5 efficiency harness (real convergence-to-tol
  iters) on the toy, then a non-toy DEQ.
- **Woodbury "warmer-than-warm" prior — SMOKE TEST RUN (2026-07-08, `c5_woodbury_prior.py`; a C5 ingredient,
  candidate appendix).** Claim tested (residual only, timing excluded): initialize a post-edit re-solve at
  `z*_old + (I−J)⁻¹δh` (the low-rank/Woodbury linear-response prediction of the NEW equilibrium) — does it land
  closer than plain warm-start (copy `z*_old`)? **RESULT: yes, 14/15 (ckpt×class) cells, 1.8–65× lower residual;
  rank-4 truncation ≈ rank-8 ≈ full EVERYWHERE (deployable cheap version = the C2d rank-8 carry).** The 1 miss
  is the honest one: curr40 (σ_min=0.028, most near-singular) × *irrelevant* edits **OVERSHOOTS** (0.78×, worse
  than warm) — exactly the Kantorovich regime (β=1/σ_min huge → h=βLη large → linear step overshoots), and
  precisely why the loop must be predict→**certify(§4)**→correct-or-fallback, not blind predict.
  **EFFICIENCY (iters fix, 2026-07-08): the flat 151 was a solver cap** (`counted_solve`: Anderson f_max_iter=150
  + UNREACHABLE f_tol=1e-6; Anderson floors ~1e-4 so always ran to the wall). Re-ran with an ACHIEVABLE tol (1e-3)
  so early-stop fires → iters now init-sensitive (range 5–266). **HONEST RESULT, more sobering than the residual:**
  (a) the dominant win is **warm-start over cold** (e.g. curr24 filler 46→6, already known); **Woodbury adds a
  modest further ~15–30%** over warm on well-conditioned cells. (b) **RESIDUAL ≠ ITERATIONS** — curr16 filler is
  the counterexample: Woodbury residual 8× lower (0.55 vs 4.30) yet MORE Anderson iters (19.8 vs 12.8). Anderson
  convergence tracks the error's slow-mode (carry) content + subspace conditioning, not initial residual
  magnitude (consistent w/ the σ_min slow-mode story). So the clean residual win does NOT cleanly translate to
  iteration savings under Anderson. (c) near-singular curr40 doesn't converge to tol (frac 0.17–0.50, quarantine).
  **INSIGHT: solver choice decides whether the prior pays — Newton (quadratic from a low-resid init) WOULD cash
  the residual advantage; Anderson doesn't.** → pair the Woodbury rung with a NEWTON corrector. Reinforces the
  appendix-corollary scoping: modest/inconsistent iteration win on this toy+Anderson, do NOT oversell.
  LIT BOUNDARY (searched 2026-07-08): Sherman–Morrison IS standard in DEQ —
  but **solver-internal** (Broyden approximating J⁻¹ *within one solve*, Bai 2019); DEQ warm-start across inputs
  exists but is plain state-copy (temporal DEQ, DEQ-MPC). **Cross-EDIT Woodbury-prior (measured low-rank
  footprint → predict new eq → residual-certify) NOT found** (provisional, 3 searches — verify w/ Geng before a
  novelty claim). Scoping: in-scope as a CONSTRUCTIVE COROLLARY of the low-rank corridor (C2d-V4), appendix/short
  main-text mention, NOT a headline — the paper stays characterization-forward.
- **PE DECISION RESOLVED — the certificate is PE-AGNOSTIC (2026-07-09, `curriculum_currnp.py` + C2/C2d re-run).**
  Trained `currnp` (causal + PURE relative PE: NO_POSW+REL_BIAS via the window curriculum) → the causal relay
  works with relative-only position: σ_min ladder 0.197→0.024 (vs curr 0.184→0.028), recall 1.0/1.0/0.974/0.893/
  0.810. Re-ran the two load-bearing experiments on it (`load_checkpoint` now restores substrate flags; curr*.pt
  unaffected; the `curr*` glob also matches `currnp*` → one run = abs-vs-rel compare). **EVERY property holds on
  relative PE:** C2 **ENVELOPE OK** on all currnp filler cells (ξ_meas 0.77/1.49/1.18 ≪ ξ_faber 8.8/13.5/18.2),
  filler far/near≈0, must-carry ridge preserved; C2d **V1** 0.97/0.96/0.99, **V3 zero false containments
  everywhere**, **V4** carry rank ~6–8/64, **V5** product-form exact ~2e-15. HONEST NUANCE: currnp is MORE
  ill-conditioned at matched gaps (κ 2–8× curr's — gap24 733 vs 91) → looser (still sound) ξ_faber; a touch more
  near-singular multistability wobble. **ADOPTED FRAMING:** *the certificate holds under BOTH PE encodings; relative
  is a bit harder-conditioned AND makes structural edits local (aligned-frame)* — a stronger, more general claim
  than picking one. **Relative = PRIMARY substrate; absolute = pedagogical intro + PE-agnosticism ablation.**
  Build C6 (hub/spoke) on relative from birth.
- **QK-NORM (cosine attention) — NULL: conditioning ↑ but recall ↓, net loss (2026-07-09, `curriculum_currnp_qk.py`).**
  Simple modern drop-in to fix currnp's ill-conditioning: L2-normalize q,k over the head dim + a learned
  per-head temperature τ (replacing 1/√dh), decoupling attention SHARPNESS from logit MAGNITUDE.
  RESULT — σ_min ~**DOUBLED** (currnp 0.030/0.040/0.036/0.024 → currnpqk 0.067/0.076/0.080/0.042) but recall
  **COLLAPSED 15–24 pts** (currnp 1.0/0.974/0.893/0.810 → currnpqk 0.853/0.681/0.589/0.568), and training loss
  stayed high (gap16/24 0.73/1.02 vs currnp 0.21/0.47 — HARDER to fit). τ settled ~5–8/head (head 0 stayed
  highest, ~8). Mechanism: capping logit magnitude starves the peaky (near one-hot) attention that long-gap
  retrieval needs — **the peaking↔contraction tension is real and TIGHT.** VERDICT: net loss → keep plain
  relative currnp primary; QK_NORM stays an opt-in flag (default off, curr*/currnp* unaffected).
  **SILVER LINING: σ_min↑ / recall↓ moved in LOCKSTEP = a clean empirical demonstration of the conditioning↔recall
  tension** (the paper's "conditioning, not contraction" core) — the failed fix is *evidence for* the mechanism.
  Also rules QK-norm OUT and points at **RoPE** (orthogonal rotation, leaves logit magnitude/sharpness alone)
  as the one un-ruled-out simple conditioning option.
- **RESIDUAL / STATE-SKIP smoke test — a σ_min-LOWERING block-diagonal knob (NOT a third failed fix); the linear
  control for the MLP prediction (2026-07-12, `curriculum_currnp_residual.py`, `sw.RESIDUAL`).** The curr/currnp
  cell is residual around the INJECTION (out = h0 + s·A(z)) so J = s·∂A/∂z has NO identity component. Added a
  learnable STATE skip `out += r·z` (r ∈ [0,R_MAX=0.8)) → J gains +r·I → I−J = (1−r)I − s·∂A/∂z. Trained on the
  currnp recipe (causal + rel-PE, window curriculum), TRIMMED ladder {0,16,40}, + a no-residual same-ladder CONTROL
  (`DEQ_RES=0`, currnptrim*). **FOUR clean reads** (post-hoc σ_min & ρ(G) on matched MQAR seqs, `res_rhog`):
  (1) **Recall collapse at gaps 16/40 is the TRIMMED LADDER, not the residual** — the no-residual control craters
  identically (recall 0.618/0.340 vs full-ladder currnp 0.974/0.810; residual 0.571/0.347 = a wash vs control). So
  the residual is recall-NEUTRAL here; a full-ladder run is owed for a clean long-gap recall verdict.
  (2) **The residual robustly LOWERS σ_min ~0.63×** at every gap incl. the matched-recall gap 0 (0.120 vs control
  0.197; 0.052 vs 0.081; 0.038 vs 0.059) — a σ_min-**worsening** knob (the (1−r)I diagonal shift toward
  singularity), the OPPOSITE of QK-norm/RoPE → **NOT a third failed fix** in that family.
  (3) **The model RECRUITS it** (r settles ~0.23, not driven to 0) even though it worsens conditioning & doesn't
  help recall — a state-skip is wanted.
  (4) **ρ(G) = nilpotent(0) for BOTH** residual and control (causal → strictly-lower G). So the residual moves
  σ_min while leaving ρ(G) — but on the causal face ρ(G)=0 is STRUCTURAL (nilpotency forced by the mask), so this
  is NOT a real (σ_min,ρ(G)) decorrelation, it just confirms the skip is a **block-DIAGONAL knob** (touches D, not
  the off-diagonal reach structure). MEANING: this is the cleanest witness of the "block-diagonal → σ_min-only"
  mechanism (a pure identity path, no nonlinearity) → it is the **LINEAR CONTROL that de-risks the MLP's
  'moves σ_min not ρ(G)' prediction** before the MLP is ever run on a semantic task. CAVEATS: causal only (ρ(G)=0
  pinned), trimmed ladder (recall inconclusive). OWED: (a) **BIDIR residual** = the real decorrelation test (does a
  NONZERO ρ(G) move? the additive skip changes D→D⁻¹→G so likely moves BOTH on bidir; the clean σ_min-only lever is
  the RELAXATION residual z+α(g−z) which scales I−J by α with ρ(G) exactly invariant — but that's a solver reparam,
  same fixed points); (b) full-ladder residual. Files: `curriculum_currnp_residual.py` (DEQ_RES toggle),
  `sw.RESIDUAL`/`R_MAX`, currnpres/currnptrim{00,16,40}.pt, logs currnpres_log.txt / currnptrim_log.txt.
- **Per-WINDOW residual early-stop — TEST 1 (retrospective): SAFE but MODEST (2026-07-09, `c5_window_earlystop.py`).**
  Freeze a window once its certified error bound `Σ_j‖R[i,j]‖‖res_j‖ < 1%·‖z*_i‖` during a warm post-edit
  re-solve; the active set shrinks toward the edit. **v1 was unsafe** (rampant false-freezes) from a TARGET BUG:
  near-singular ckpts are near-multistable, so the WARM re-solve lands on a different branch than a COLD solve —
  scored against the wrong fixed point. **Fixes:** target = Newton-polished WARM-branch fixed point (+ R there);
  freeze gated on the linear regime (bound only holds near z*, Kantorovich) + PATIENCE=3 consecutive (Anderson
  residual is non-monotone). **RESULT: SAFE — 0 false-freezes everywhere, final_err ~3–5e-3 ≤ tol.** But COMPUTE
  savings are MODEST: global-gate 59–99% of full window-iters; a per-WINDOW local gate (freeze once window i is
  locally converged, not waiting for the ball) gave only MARGINAL gains (58–95%; one 8-pt win curr24 filler
  79→71%). Two limiters: (1) sequences too short (3–5 windows, ~no far field past the ball); (2) the NORM bound
  is LOOSE — `‖R[i,ball]‖‖res_ball‖` over-counts the true directional influence `R[i,ball]·res_ball` (the C2d
  norm-vs-directional slack, 7.5–764×), so far windows stay active until the ball settles. It WOULD scale with
  sequence length (far field ≫ ball) → the owed longer-context (gap≥60) substrate; a directional bound is the
  tightness ceiling (but that's the oracle predictor). **META (both efficiency ingredients):** Woodbury prior
  AND per-window early-stop are both SAFE-BUT-MODEST on this toy → the certificate's value is the **guarantee**
  (provably-correct partial recompute), not a toy-scale compute win. Test 2 (real masked solve) DEFERRED — would
  confirm the same modest number. Verdict: appendix-level "certified, safe, scales with length," NOT a headline.
- **WOODBURY + BROYDEN — POSITIVE (2026-07-09, `c5_woodbury_broyden.py`). Corrects a prior mis-read.** Smoke
  test (2 seqs × 2 edits/mode over ALL 15 curr*/currnp*/currnpqk* ckpts; off-the-shelf torchdeq Broyden,
  Woodbury only as init; NO custom solver / NO Jacobian seeding — an earlier over-engineered B0-seeding design
  was scrapped as out-of-scope). **First read at tol=1e-6 looked like a NULL ("Broyden saves nothing") — that
  was a CAPPING ARTIFACT** (neither solver reaches 1e-6, all pinned at MAXIT). Re-run at an ACHIEVABLE tol=1e-4
  (Anderson's floor) flips it. TWO wins: **(A) the SOLVER SWAP is the big one** — Broyden converges 100%
  EVERYWHERE incl. near-singular curr40/currnp40/currnp08 where **Anderson STALLS** (curr40 filler: Anderson-warm
  201 evals @0% conv = failed → Broyden-Woodbury 20.8 @100%; currnp08 filler 115@75% → 15.5@100%; ~5–10× fewer
  f-evals AND actual convergence). **(B) the Woodbury init pays WITHIN Broyden** (fair same-solver): Broyden-Woodbury
  ≤ Broyden-warm almost everywhere, ~15–30% typical (curr24 filler 6.2→5.2), up to 2× (curr40 relevant 70→29);
  one mild inversion (curr08 relevant 15→16.5, an overshoot — c5_woodbury nonlinear-overshoot note). **The prior
  now CONVERTS to iters** (under Anderson it didn't). **THEORY CORRECTION:** the earlier "residual≠iterations, init
  doesn't help" (c5_woodbury) is ANDERSON-SPECIFIC, not a law — fixed-point iteration is spectral-gap-limited so a
  stiff (small-σ_min) operator stalls it; **quasi-Newton is affine-invariant** (local convergence ~conditioning-
  independent) so Broyden takes the stiff mode in stride. My "no solver escapes the conditioning" was too strong.
  CAVEATS: cross-solver f-eval counts not strictly comparable (Broyden line-search evals ARE counted → its
  advantage is if anything understated); n=4 edits/cell so magnitudes noisy though direction consistent across all
  15 ckpts. **SCOPE:** this is the EFFICIENCY SPIN-OFF's result (Geng's area), not this paper's headline — but it
  UPGRADES the Woodbury rung from "safe-but-modest appendix" to "cross-edit prior + quasi-Newton solver = a real
  incremental-inference win on stiff DEQ operators." Records file `c5_woodbury_broyden_records.npz`. Next if
  pursued: scale to more seqs/edits for firm magnitudes; L2 (low-rank Woodbury update of R_old, the deployable
  cheap prior) now clearly worth building since L1 pays.
- **NONCONTRACTIVE-BUT-LOCAL WITNESSES — the "conditioning, not contraction" evidence, on record (2026-07-09).**
  The claim that edit-locality is governed by σ_min(I−J), not by contraction ρ(J)<1, is DIRECTLY witnessed, and
  we've been underplaying it: **curr40 has ρ(J)=8.37 (strongly NONcontractive) yet the filler envelope HOLDS**
  (ξ_meas≈0.97 hop, ENVELOPE OK, σ_min=0.026, recall 0.83); **currnp is noncontractive at gaps 0/16/24/40**
  (ρ(J) 1.69/1.26/2.11/4.44) with C2 + C2d holding on it. So the noncontractive regime is real AND edit-local —
  the old angle is empirically supported, not folklore. IMPORTANT
  ASYMMETRY: the ρ>1 witnesses are all on the **CAUSAL face** — the trained *bidir* face landed contractive
  (ρ<1 throughout, 0.43→0.87), and the pointer-chase / QK-norm runs also landed contractive. So the
  noncontractive evidence lives on the causal face. FRAMING UPGRADE: "conditioning, not contraction" is demoted
  from thesis to HOOK; the crown went from a single invariant (σ_min over ρ(J)) to **TWO invariants — {ρ(G)
  reach, σ_min amplitude} over ρ(J)** — with ρ(M)=ρ(I−J) a non-participant (a red herring, not the contrast).
- **WITNESS SOLVER-INDEPENDENCE CHECK + POSTER-CHILD SWAP (2026-07-09, `c2_witness_solver_check.py`).** Solved
  each stiff cell TWO ways (Anderson+Newton vs Broyden+Newton, same zero init) to answer "is ρ>1-yet-local a
  solver artifact?" **4/5: CONFIRMED solver-independent** — currnp16/24/40 (ρ=1.26/2.11/**4.44**) + currnpqk40
  agree on z*, σ_min, ρ(J), far-field R to 6+ digits (‖zA−zB‖~1e-7). So the noncontractive witness is a property
  of the OPERATOR, not the solver — and we now have a clean TREND (ρ 1.26→4.44, all edit-local, solver-checked),
  stronger than any single point. **SURPRISE: curr40 (the old poster child) is MULTISTABLE** — the two solvers
  land on DIFFERENT fixed points from the same init (‖zA−zB‖=0.14, both polished to ~1e-7 = both legitimate),
  with σ_min differing by branch (0.026 vs 0.040) and R blocks disagreeing 34×; ρ(J)=8.37 at both. Quantifies the
  near-multistability we'd flagged in the Test-1 target bug. **DOES NOT refute the thesis:** claim holds at BOTH
  branches (both noncontractive + edit-local); σ_min DIFFERS by branch = exactly "reach governed by σ_min"; and
  our certificate is LOCAL (a-posteriori, around a given z*) so each branch is independently well-posed —
  multistability only bites a COLD solve expecting a canonical answer, NOT the maintenance regime (warm-start from
  the cached branch = branch well-defined). It even reinforces the maintenance framing / the Test-1 warm-branch
  fix. **ACTION: poster child → currnp40 (ρ=4.44, single fixed point, solver-checked); currnp16/24/40 = the
  trend.** curr40 REFRAMED: kept as "strong non-contraction can bring multistability, and the local/maintenance
  certificate is robust to it" — a point in our favor, not a clean-witness example. (Reporting nit: the script's
  `(B matches)` on the far-field line is hardcoded text, not a computed check — the `rel_R`/`dz` numbers are the
  real signal; label to be fixed.)
- **POST-EDIT TIER-2 CERTIFICATE — SOBERING, deflates "tight deployable" (2026-07-09, `c2_postedit_certify.py`).**
  Measured the DEPLOYABLE residual (real post-edit warm residual `r=f_new(z*_old)−z*_old`, 6 edits/class × 4 cells
  curr08/16, currnp16/40; z*_new = warm-branch target) vs the true error. **(1) scalar bound LOOSE for real edits:
  ratio_lin=(‖r‖/σ_min)/‖Δz*‖ = 7–34×** (curr08 filler 17.5; currnp40 filler 34). **(2) WHY — refutes my mechanism
  hypothesis:** cos_stiff (alignment of r with top-8 stiff left-sing dirs) = **0.01–0.14** → real edit residuals are
  NOT stiff-dominated; they're broadband/fast-mode-dominated (the INITIAL edit residual, before iterating — residuals
  only go stiff-dominated AFTER fast modes decay). So the scalar σ_min bound (worst-case-over-stiff) over-charges by
  ~1/cos. **(3) RIGOR — NK NEVER fires: 0% in-TR every edit** (h=β²L‖r‖>½; ratio_NK=nan). β=18–41 ⇒ needs ‖r‖≲1e-3–
  1e-4 = near-convergence; real ‖r‖=O(1–10), 3–4 orders too far. So NK = a "you've nearly converged" cert = a
  certified STOPPING rule (iterate the re-solve until h≤½, THEN certify), NOT one-shot "certify-the-reuse-without-
  solving." Robust to L (β² exact; local L if anything under-estimates). **(4) Woodbury still an accuracy win**
  (‖r_wood‖≪‖r‖, 18–39×: curr08 filler 5.47→0.31; currnp40 filler 0.29→0.010) but does NOT reach the NK TR in one
  shot (0%). **CONSTRUCTIVE REFRAME:** the TIGHT deployable object is the DIRECTIONAL `‖R·r‖` (= first-order actual
  error z−z*≈M⁻¹r, the C2d/Woodbury resolvent-action machinery), NOT scalar ‖r‖/σ_min — the scalar is loose
  precisely because it assumes stiff alignment the real residual lacks. So: **scalar σ_min = cheap LOOSE screen;
  directional R·r (resolvent-needing, Woodbury-updatable) = the tight estimate; NK = rigorous stopping rule
  (conservative on ill-cond).** **PAPER IMPACT:** soften tier-2 "tight and deployable" → "SOUND; scalar = conservative
  screen, directional/resolvent = tight, NK = certified stopping rule." CORE (characterization, invariants,
  witnesses) untouched; consistent w/ the honest "don't oversell deployment" positioning. OWED: measure ‖R·r‖/actual
  directly to confirm the directional-tight claim (have R,r in the script — quick add).
  **DIRECTIONAL MEASURED (2026-07-09, same script +ratio_dir=‖R·r‖/actual): the IMPOSSIBLE TRIANGLE, empirically.**
  `‖R·r‖` is TIGHT — **0.98–1.01 (exact) on small (filler) edits across ALL cells incl. near-singular; 10–34×
  tighter than scalar.** BUT on LARGE edits (irrelevant/relevant) it **UNDER-predicts** (ratio_dir 0.66–0.81 on
  currnp16/40; 1.12–1.17 slight-over on well-cond curr08/16) → `‖R·r‖<actual` ⇒ NOT a rigorous upper bound there
  (nonlinearity + cached-resolvent drift R_old≠R_new; degrades with edit size AND κ). So the three certs each give
  up ONE corner: **scalar ‖r‖/σ_min = sound+cheap, NOT tight (7–34×); directional ‖R·r‖ = tight+cheap-ish, NOT
  rigorous (optimistic, under-predicts 0.66× on big/near-singular edits); NK R₋ = rigorous+tight, NOT cheap/wide
  (near-convergence only, 0% in-TR).** PICK TWO — measured, not asserted. Deployment upshot: the TIGHT deployable
  object is effectively a HEURISTIC (directional estimate, excellent predictor ~1.3× either way, no guarantee) —
  matches ZJ's intuition that past the loose-sound bound it's good heuristics. The one open PRINCIPLED escape is a
  DIFFERENT axis: reader-set / goal-oriented (adjoint / dual-weighted-residual) bound — bound only the OUTPUT at
  reader positions, drop the far-field slack → sound-AND-tighter than scalar (still an over-estimate). Orthogonal
  to the linear-vs-nonlinear axis these 3 live on = the genuine "do better and stay rigorous" lever.

---

## 11c. C6 pointer-chase — substrate capacity ceiling; reader-set stays a principle (2026-07-09)

Built a **pointer-chase-to-root** task (`pointer_chase.py`) with a known ground-truth dependency graph (each
query's start→root path = its dependency lane; subtree size = hub-ness) to test the reader-set machinery beyond
planted-uniform MQAR. Chose chase-to-**root** because fixed-k chase isn't equilibrium-computable (a DEQ can't
count hops); root-label propagation `root(i)=root(ptr(i))` is the equilibrium-natural fixed point.
- **Directionality REFUTED as the ceiling:** causal and bidirectional relative-PE DEQs both cap at ~0.68 (final
  by depth d1/d2/d3/d5: causal 0.69/0.69/0.70/0.66, bidir 0.69/0.68/0.65/0.65 — essentially identical). So the
  earlier hypothesis "pointer-chase is bidir-natured" is wrong; the mask doesn't matter here. (Aliasing also
  ruled out: the disjoint value-vocab fix moved depth-1 0.711→0.710.)
- **It's a CAPACITY ceiling, N-dependent:** recall vs node count — N=8/2roots ~0.68, N=6/2roots ~0.77,
  N=4/1root 1.0 (degenerate: single root = "find the root", no chase). No sweet spot that is both learnable
  (~0.9) AND a non-trivial multi-hop chase at d=64/single-tied-layer. Content-based *iterated* associative
  lookup (iterated-MQAR + terminal detection) compounds per-hop error; MQAR (1 hop) trains to ~1.0, the chase
  doesn't. Conditioning stayed HEALTHY throughout (σ_min 0.05–0.18, ρ<0.84) — the substrate is well-behaved;
  the model just can't learn the task to high accuracy.
- **DECISION (pre-committed):** since a ~0.9 non-trivial substrate didn't materialize, **demote the reader-set
  to a discussion-level PRINCIPLE** (already evidenced by must-carry + C2t on both faces) and mark
  **pointer-chase / C6 = FUTURE WORK** (needs a larger or multi-layer/multi-head substrate). The paper stands
  on its core: two-tier certificate, two faces, PE-agnosticism, noncontractive witnesses. Don't rabbit-hole the
  substrate. Code kept: `pointer_chase.py`, `pointer_chase_train.py`(+`_bidir`), `pointer_chase_min.py`.

---

## 12. Scale path — everything survives losing the dense Jacobian (ZJ's practicality question)

The dense J / exact resolvent is the toy-scale **oracle we validate estimators against**, not the method.
Every quantity has a matrix-free analog built from three standard primitives — **JVP/VJP** (autodiff gives
`J·v` and `vᵀ·J` at one forward/backward each), **Krylov solvers** (GMRES/CG on those products), and
**randomized low-rank sketching** (Halko-style):
- **Edit responses, warm-start, multistability probes:** never needed J — forward solves only. Scale-free.
- **σ_min, κ, ρ:** inverse/power iteration with JVP/VJP + a Krylov solve; tens of J-products.
- **Resolvent columns / F_p:** `F_p·v` = one matrix-free solve of `(I−J)x = e_p⊗v` (GMRES). The carry basis
  via randomized range-finding: ~r+5 solves per region, **amortized at cache-build** — and V4's r≈8 is exactly
  what makes sketching cheap.
- **Causal product form:** T_k actions are *window-local* solves (never materialize T_k); norms by power
  iteration; carry propagation = pushing an r-column sketch through the T-product (k window-solves on r vectors).
- **Newton polish → Jacobian-free Newton–Krylov** (standard).
- **ν (normality):** Hutchinson trace estimation on JVP∘VJP compositions — estimable, variance-limited (the
  one genuinely noisy item).
Honest cost: every Krylov solve inherits the κ of (I−J) — near-singular conditioning means many iterations
(the same critical slowing that loosens the certificate; preconditioning = open). So the scale story is
"replace the oracle with standard matrix-free estimators, at iteration counts set by the very conditioning
we measure" — a limitation *quantified by our own invariant*, which is the right kind of limitation.

---

## 13. Remaining (non-spine) debts before drafting

1. ~~Pure-relative-PE retrain~~ **DONE — pure-relative RELAYS** (bidirnp00–40: 1.000/0.987/0.912/0.937/0.819;
   §11). Application narrative viable; posw was a crutch not a wall; insert/delete unblocked in principle.
2. Full reads of the 4 must-read refs: Vogt 2006.14123, Cirone 2402.19047, Benzi–Golub decay, 2411.04400.
3. ~~Directional certificate (C2d)~~ **DONE — validated (§10):** a-priori taxonomy quantitative, zero false
   containments, carry rank ~8/64, product-form identity at machine precision, scalar slack up to 764×.
4. ~~Deferred-billing / trigger (C2t)~~ **DONE — NEGATIVE (§10):** must-carry robust even when permitted;
   write-cost reader-independent on all substrates; lazy evaluation didn't emerge; visibility↔trainability
   tension found. Tier-3 metering survives; eager/lazy framing demoted.
5. **Edit-interference experiment (§11, claim C2i)** — paired-edit response vs sum-of-singles vs separation;
   maps the linear-regime validity boundary. Cheap, same C2 machinery.
6. C4 multiscale — optional/stretch.
7. **Insert/delete (§11) → v2 spine, now UNBLOCKED:** insert-type `apply_edit` + alignment bookkeeping on the
   bidirnp checkpoints, vs the "band at the cut" prediction. Optional: re-run C2-bidir on bidirnp for substrate
   consistency; anchor token in the drawer if gap-40 recall (0.819) needs a boost.

---

## 14. Anchor / global register: canonical (from-scratch) vs graft, matched pairs (2026-07-14)

The gap-60 near-singular relay (pure-banded craters: currnp 0.70/smin~0, bidirnp 0.46/smin~0) motivated a
GLOBAL REGISTER token (sw.ANCHOR, slot 0, hub-and-spoke, rank-d border). Two build modes, each matched against a
same-recipe ANCHORLESS control so the anchor is the ONLY difference:
  - GRAFT (curriculum_anchor60): warm-start the trained *40 body (strict=False, fresh anchor), short 40/60 ramp.
  - CANONICAL from scratch (curriculum_anchor): ANCHOR on from step 0 through the full window->gap curriculum,
    PHASE_B extended to 60. Anchorless control = curriculum_anchorless (identical recipe, ANCHOR off).

Gap-60 results (recall / rho(J) / sigma_min(I-J)):

  CAUSAL (currnp)
    anchorless control (scratch) : 0.640 / 10.33 / 0.0092
    anchor  from scratch         : 0.617 /  3.69 / 0.0294
    anchorless control (GRAFT)   : 0.627 /  1.36 / 0.0004   <- matched twin of the graft anchor
    anchor  GRAFT                : 0.801 /   -   / 0.0159

  BIDIR (bidirnp)
    anchorless control (scratch) : 0.478 /  1.14 / 0.0029
    anchor  from scratch         : 0.855 /  2.91 / 0.0156
    anchor  GRAFT                : 0.751 /   -   / 0.0001

READ (substrate-dependent; corrects an earlier "just use the graft" instinct):
  * CONDITIONING rescue in every case: from-scratch anchor lifts sigma_min ~3x (causal 0.009->0.029) to ~5x
    (bidir 0.003->0.016) over the matched control, and tames rho (causal 10.3->3.7).
  * BIDIR: from-scratch canonical anchor is the outright winner -- recall 0.855 beats BOTH the control (0.478)
    and the graft (0.751). The register pays off most exactly where the banded relay craters. Use canonical.
  * CAUSAL: graft wins on recall (0.80), and its MATCHED graft-control makes it clean -- same warm-start recipe,
    anchor the only difference: anchor 0.801/0.0159 vs graft-control 0.627/0.0004 = +0.17 recall, 40x sigma_min.
    The control also craters (currnp40 0.81 -> 0.627 when fine-tuned out to gap 60 without the hub), so the anchor
    is demonstrably what holds recall + conditioning. From scratch instead destabilizes (0.617, resid ~1e-2)
    because the ever-present hub shortcuts the already-clean causal relay -- graft preserves that relay. Use graft.
  * NEGATIVE: co-training the register from scratch on the CAUSAL substrate settles into a high-gain,
    poorly-converged equilibrium (rho 3-4, DEQ residual ~1e-2 vs the ~1e-4 the solver should reach) -- the loss
    fits but the fixed point is not clean. The graft avoids this by keeping the already-well-conditioned body.

Artifacts: checkpoints {currnp,bidirnp}anchor{00..60}.pt (canonical), {currnp,bidirnp}ctl{00..60}.pt (control),
*anchor60_graft.pt (preserved graft), currnpgraftctl60.pt (causal graft-control). Scripts:
experiments/curriculum_anchor.py, curriculum_anchorless.py, curriculum_graftctl60.py.
Compute note: eval spectrum() needs ~5.7GB (dense ~4.5k^2 Jacobian + SVD), near-fills the 6GB card and
serializes concurrent evals -> the 3-way concurrent run took ~170-224 min/substrate; see experiments/PERF_NOTES.md.
