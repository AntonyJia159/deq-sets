## Paper outline (working draft, TMLR)

**Title/thesis:** *Conditioning, not contraction: certified, maintainable equilibria in DEQ transformers.* Edit-locality is governed by σ_min(I−J) (amplitude) and ρ(G) (reach) — not ρ(J) — and the certificate system is **closed under context edits**: the geometric constants (C, ρ, the R-blocks) are cached, and each edit costs one *local scalar gate* (value edits via tier-1/2; insert/delete via σ_min(S) or ‖R_bb⁻¹‖), with no new Stein solve. "Conditioning, not contraction" stays as the falsifiable hook; "maintainable" is the enrichment — it upgrades the claim from *static* certification to a closure property. (Final title call held until C7 lands: if C7 disappoints, "maintainable" retreats from the title to a section.)

### §1 Intro

Problem: when is a mid-context edit to a DEQ's fixed point provably local? Contributions list = the two invariants, the certificate taxonomy, the converse theorem, structural-edit extension, and the maintenance pipeline. One paragraph on looped/depth-recurrent transformers as the bridge to scale (residual certificate = certified early exit).

### §2 Setup + the two invariants

- Fixed point z*=f(z*,h), M=I−J, resolvent R=M⁻¹. Edit = δh at token i; effect = R·(∂f/∂h)δh.
- **Prop 1 (amplitude):** ‖δz‖ ≤ ‖δforcing‖/σ_min(M). **Prop 2 (reach):** block-Jacobi G; off-diagonal decay ‖[R]_{ij}‖ ≤ C·ρ(G)^{d(i,j)}.
- Explicitly a *non-participant* lemma: ρ(J) (contraction) controls neither. State assumption tags on every proposition (linearization, branch-locality, partition width).

### §3 Two faces of G

- **Causal = nilpotent** (ρ(G)=0, exact finite product form): diagnostic register — near-singularity is *the mechanism of recall* (Ganguli–Sompolinsky ancestry: optimal memory = non-normal delay line).
- **Bidirectional = geometric** (ρ(G)∈(0,1)): clean certificate register — screened propagator, measured ξ.
- One theorem statement covering both via the Stein/Lyapunov adapted norm: ‖G^k‖ ≤ √κ(P)·r^k. Mean-mode deflation as the unifying pattern: **low-rank exception channel + screened bulk** (name it once, reuse for carry subspace, Perron mode, bordering, **and the anchor/global-register border** — a rank-d hub bolted on the screened band, certified by the *same* banded-body + low-rank-border decomposition; §8 measures it).

### §4 Certificate taxonomy (the core)

Organize as a 2×2-ish table: {a priori / a posteriori} × {scalar / directional}:

| | scalar | directional |
|---|---|---|
| **Tier-1 (a priori)** | C·ρ^d envelope (Stein) | **sound directional charge** √(vᵀP_jj v)·eff^d — directional *and* sound at ~10× slack, free from the cached Gramian (block-diag P family) |
| **Tier-2 (a posteriori)** | ‖r‖/σ_min | **reader-set/DWR (✓ measured)**: ladder actual < dwr_est (1.5–7×) < reader-bound (~10×) < global (~100×), 100% soundness, reader-restriction worth 6–12×; prior R·δr (heuristic, flagged) |
| **Tier-3** | NK ball R₋ | — |

The organizing sentence: the P_jj charge anisotropizes on the *source/direction* side (a-priori, reader-blind); DWR anisotropizes on the *reader* side (a-posteriori, edit-blind) — complements at opposite ends of the pipeline, and the split is *measured*, not asserted (the charge ranks filler 1–2 orders down on every checkpoint but cannot order relevant-vs-irrelevant — a routing property the source block is structurally blind to, monotone 3/8; that failure boundary is exactly the hand-off to DWR).

Then the **impossible triangle** (cheap/tight/sound — pick two) with your measured looseness numbers as the evidence, stated for *reader-agnostic* bounds — DWR is the measured partial crack (sound + tight + cheap *when the reader is known*), the priced exception. NK caveat to state: rigorous NK needs an *upper bound* on the curvature L over the whole ball; cheap L-probes (a JVP finite-difference along the step direction) give a point sample, so cheap-L quietly demotes tier-3 back toward tier-2 — certified-L is the expensive part, not the formula.

### §5 Converse: no free locality

Task correctness at distance Δ ⟹ ‖[R]_{reader,write}‖ ≥ c ⟹ σ_min ≤ 1/c and ξ ≥ Δ/ln(C/c). Corollaries: QK-norm/RoPE nulls, the recall-vs-conditioning tension is *forced*, not incidental. This is the theorem that elevates the paper above "we measured a thing."

### §6 Structural edits (insert/delete)

**Setup.** An insert borders M_old = I−J_old with the new token's couplings b (old→new), cᵀ (new→old), δ (self-block). Windowed attention ⟹ b, c are nonzero only for old blocks inside the new token's window — spatially supported *at the cut*. That locality is the whole engine.

**Two distinct theorems — keep them separate (they answer different questions):**

- **(a) Bordering two-term envelope ("no new Stein solve") — per-edit, spatial.** The old-old block of the new inverse is the Schur identity ΔR = (Rb)S⁻¹(cᵀR), with S = δ − cᵀRb the new token's effective stiffness (d×d) after eliminating the old system. Applying the *cached* tier-1 bound to each leg (w = R acting on a cut-local source ⟹ ‖[w]_i‖ ≤ C′ρ^{d(i,cut)}; likewise ‖[u]_j‖ ≤ C″ρ^{d(cut,j)}) gives the new envelope
  ‖[R_new]_{ij}‖ ≤ C·ρ^{d(i,j)} + (C′C″/σ_min(S))·ρ^{d(i,cut)+d(cut,j)}.
  Physically: the new token is a **scattering center** at the cut — any i→j response gains one extra path (i→cut, scatter with gain S⁻¹=1/σ_min(S), cut→j), each leg decaying at the old rate ρ. Triangle inequality d(i,cut)+d(cut,j) ≥ d(i,j) ⟹ the correction is never slower-decaying than the direct term: a subdominant bump localized at the cut. **Rate inherited, constant picks up one factor.** Cost: nothing geometric recomputed (C, ρ, R-blocks cached); the only new number is the scalar gate σ_min(S), and S needs only R-blocks near the cut (b, c cut-local). Delete = exact dual (downdating): remaining-block inverse R_AA − R_Ab(R_bb)⁻¹R_bA, gated by ‖R_bb⁻¹‖ = the deleted token's diagonal resolvent block (its *hub-ness*). This is what "the certificates close under structural edits" means concretely.

- **(b) Stein budget theorem — cumulative, temporal.** Margin η=1 (RHS normalized to −I) spent by λ_max(Δ) per edit; the *cached Gramian P itself* stays valid only while η − Σλ_max(Δ) > 0. This asks a different question than (a): not "does this edit's envelope hold" but "how many edits before the geometry certificate goes stale and needs a recert." Δ = G̃*PẼ+Ẽ*PG̃+Ẽ*PẼ, Hermitian, rank-O(d), localized at the cut.

**Scope honestly:** bordering does *not* yield a cheap rigorous *adjoint* (ΔM is global because J is state-dependent and the equilibrium shift spreads at rate ρ(G)) — that measured negative lives in §8. The two-term envelope (a) is about the *response reach* update, which is legitimately cheap+cut-local; don't let a referee conflate the two. Evidence for this whole section is C7 (pending, see gate).

### §7 Pipeline: predict → certify → correct

Open with a named subsection — **"what prediction buys, measured"** — the 11b warmer-than-warm results: warm-starting the corrector from the tangent/Woodbury prediction vs. from the stale cache alone, in solver iterations-to-tol (k_tol), plus the honest scaling law **win ∝ L/ξ** (fraction of the sequence already converged at warm init: pays on long contexts / short reach; cannot pay at toy L~2W where the ball fills the sequence). This grounds §7 as *proposal + measured primitive*, and sets up C7's columns (same k_tol accounting extended to structural edits).

Then: Euler–Newton continuation framing (Keller/Deuflhard; explains Broyden>Anderson observation). The 7-step loop; Woodbury + carry-basis predictor; σ_min ladder gating (tangent / corrected / bail-to-solve near fold). Position as *proposal + toy validation (C7)*, not a benchmarked system. Fold-point = multistability onset (one paragraph, v2 teaser; if extended: the interesting trajectory questions are frontier-saturation / lead-lag / criticality — a bare "σ_min drops over training" plot is near-tautological given §5).

**Taxonomy of reuse (organizing table for this section).** Three families, by *what object* is reused: **S**tate (z*), **O**perator (M⁻¹/J/solver memory), **C**ertificate (C, ρ, P, adjoint w — the family unique to us). Each family fails independently (state can go stale while the operator is fine; both fine while P's Stein budget is spent), which is *why* the pipeline needs separate gates rather than one global freshness check. The "order" column is deliberately not a single scale — predictors, correctors, and certificates control error in genuinely different senses (Taylor order in edit size / convergence-to-tolerance / a-priori bound), so it's labeled **error control**, not "order":

| Technique | Family | Reused object | Error control | Gate (when it dies) | Cost | Status |
|---|---|---|---|---|---|---|
| Stale-cache warm start | S | old z* | O(ε) — ignores the edit | trust region / edit size | free | measured (baseline warm) |
| Certified skip outside ξ-ball | S | old z* + envelope | exact outside the ball, no claim inside | tier-1 envelope | free | measured (invalidation claim) |
| Euler tangent δz = R·δh | S | z* + R | O(ε²) — 1st-order Taylor, misses curvature | σ_min (fold proximity) | one R-action | measured (warmer-than-warm) |
| Carry-basis Woodbury predictor | S+O | z*, R, Θ_k basis | O(ε³) *within the carry subspace only* | carry-basis staleness | 8 coeffs + tangent solve | proposed (C7 hook: Θ_k stability) |
| Chord-Newton corrector | O | frozen M factorization | to-tolerance (iterative; reuse sets rate, not final error) | ‖δJ‖ vs contraction margin | reuse factorization | partially measured |
| Broyden secant memory | O | secant pairs | to-tolerance (iterative) | regime (stiff/tight tol) | free carryover | measured (Broyden>Anderson) |
| Bordering ΔR (structural) | O | R-blocks at cut | exact by theorem, at the linear level (J itself has since moved) | σ_min(S) / ‖R_bb⁻¹‖ | local blocks + scalar gate | theory ✓, C7 pending |
| Cached adjoint w (DWR) | C | reader adjoint | a-priori bound (not an ε-order) | selection nonlinearity | free per edit | **measured, survives caching** |
| Cached tier-1 constants (C, ρ, P) | C | Gramian geometry | a-priori bound | Stein budget η−Σλ_max(Δ) | free until recert | theory ✓, C7 pending |
| Approx. local bordering adjoint | C | cut-local adjoint | **none** — no error bound exists | — (the §8 negative) | 3.6–7.3× faster than re-solve | measured, *not sound* |

Two things the table makes visible: (1) every row with a genuine gate is sound; the one gateless row is exactly the measured negative — close to the thesis of this section in one line ("reuse is free; knowing when to stop reusing is the certificate"); (2) the S/O/C failure-independence above justifies the pipeline's per-family gates.

### §8 Experiments

- **Substrate table first** (task, PE, face, σ_min regime, ξ) — referees need this map before any result.
- **Claims–evidence table**: each proposition ↔ which experiment validates it ↔ measured tightness.
- Order: C2d carry (rank 8/64, stable) → claw-back ladder → tightness of the triangle (7–34×, 0.66×, NK 0%) → **DWR (✓ done, headline result)**: the four-rung ladder, plus the invariance-vs-selection split — the linear adjoint robustly certifies *who a reader does not read* (8.6× off-stripe separation long-range; 2.5× at colour granularity), while nonlinear *selection* among same-colour writes needs a fresh adjoint (the honest scope line for the whole linearized framework; place adjacent to C2i) → **certifiable rate tracks conditioning** (r≈0.55 well-conditioned → 0.90–0.95 near-singular; bidirnp40 at σ_min=0.024 admits no common rate — the cleanest empirical coupling of the two invariants; place next to the σ_min-vs-gap fit) → blocking invariance (ρ(G) at w/2w/4w, reach in **tokens**) → **insert/delete + budget (C7)** — read as *two separate verdict columns*: (a) the per-edit bordering two-term envelope (measured post-insert response vs C·ρ^d + (C′C″/σ_min(S))·ρ^{d(i,cut)+d(cut,j)}, gated by σ_min(S)) and (b) the cumulative Stein budget (running η − Σλ_max(Δ), recert when exhausted). One run covers both; if they disagree (envelope (a) holds while budget (b) says recert), that gap itself measures how conservative λ_max(Δ) is → σ_min-vs-gap fit → negative results subsection (QK-norm, RoPE, dead predictions, **and** the bordering-adjoint negative: no length-independent cheap+rigorous bound for structural-edit resolvent updates — state it as scoping §6, not as a wound; one subsection, no memoir tone).
- **Anchor / global register — the wall made concrete + a banded+low-rank intervention (preview→canonical matched pairs).** Pushing the windowed relay to gap 60 drives σ_min→0 (*every* no-anchor variant craters: causal 0.0000, bidir 0.001–0.003) — a direct instance of §5's forced recall-vs-conditioning tension. A rank-d **global register** (hub-and-spoke, O(1) reach) is a **σ_min rescue in every case** (~3–5× over a *matched anchorless control*, the anchor the only difference) and a **recall rescue for the bidir face specifically**. Substrate-split pick: **CAUSAL → graft onto the trained body** (0.80 / σ_min 0.016 vs graft-control 0.63 / 0.0004 = +0.17 recall, 40×; from-scratch destabilizes — the hub shortcuts the *already-clean* causal relay, ρ 3.7 / resid 7e-2); **BIDIR → co-train from scratch** (0.855 / 0.016 vs control 0.478 / 0.003, beating the graft 0.751 — the banded relay craters alone so the hub is genuinely load-bearing). This is the §3 **low-rank-exception channel** physically instantiated (banded body + rank-d border, certified by the same Schur/Woodbury machinery as §6 inserts); far-field probe shows **emergent content-filtering** (hub carries key/value 15× over filler), not global diffusion — near-field banded locality survives, far field becomes a *content-selective floor* (edit-locality goes content-dependent, not broken). Scale hook for §9.3: one anchor = rank-d bottleneck that *defers* the wall; O(L/W) landmark registers → banded + rank-(k·d). Scope honestly: preview-grade architecture result (single-GPU, toy), matched-control-clean but not yet a benchmarked system.

### §9 Discussion: application tilts (ordered by how much we can honestly claim)

1. **Lead: scientific certification / UQ.** The strongest honest register and the TMLR fit. Our certificates are a *goal-oriented error-estimation layer* for equilibrium models — DWR is literally the FEM/adjoint tradition imported (Becker–Rannacher lineage; verify cite), and abstention (Route B, no-common-rate cells) is a feature in this register, not a failure: the certificate *tells you* when it can't certify. Pitch: DEQs used in scientific/physics settings (implicit graph nets, equilibrium solvers) need exactly this layer, and nothing else provides it. This claim is fully backed by measurements today.
2. **Second: KV-cache / RAG / edit-heavy inference efficiency.** Regime-scoped, stated with the scaling law: win ∝ L/ξ, so it pays for **many small edits against one long cached base** (RAG chunk swaps, streaming updates, interactive editing) and *cannot* pay at toy L~2W. The differentiator vs CacheBlend/PIE-style heuristic reuse is **certified partial invalidation** — provably valid outside the ξ-ball, warm-start re-solve inside. Two honesty flags stated up front: on the *causal* face a long-memory model has ξ ≈ whole suffix (the useful regime is anisotropic reach, not small ξ); on the *bidirectional* face there is no free append (a tail token perturbs its own ξ-ball backward).
3. **Third: depth recurrence / looped transformers.** The bridge-to-scale paragraph: weight-tied iterated models (Universal Transformer, latent-recurrent-depth reasoning — verify cites) are DEQ-adjacent and *rising with test-time compute*; the tier-2 residual certificate transfers as-is (= native certified early exit / adaptive halting with an error bar), while tier-1 needs a windowed-Jacobian structure they don't automatically have. One sentence of Euler–Newton framing: "certified incremental inference for iterated models" — *suggests*, not claims; no scale benchmark exists.

4. **Fourth (suggestive): molecular / biological editing.** A natural cross of tilts 1+2 where the geometry is literal. Equilibrium/implicit GNNs on molecular graphs are edit-screening workloads by construction — lead optimization (R-group/scaffold swaps), **mutational scanning** (point mutations on a fixed backbone) = "many small edits to one cached base," the amortized `win ∝ L/ξ` regime. Both invariants pick up chemical meaning: **ρ(G) reach = structure-activity locality** (how far a functional-group/residue edit propagates through the graph), **σ_min near-singularity = conjugation/resonance/allostery** (delocalized coupling = the hard-to-localize regime). The three payoffs are all *surrogate-side* (not physical guarantees): (i) **sound cheap reuse** — reuse the cached prediction exactly for edits provably inside the reach ball; (ii) **escalation triage** — delocalized-response edits are where the surrogate is least trustworthy, so the certificate targets the expensive ground truth (DFT / assay); (iii) **consistency audit** — a "small" edit that propagates globally flags a learned allosteric effect *or* a model artifact. Honest caveats up front: it certifies the *model's* edit-response, not the chemistry, and needs an equilibrium/implicit architecture (looped-transformer bridge widens that). Sharpest one-line hook: mutational scanning. **Suggestive only — no domain experiments; a discussion paragraph, not a claim.**

Rule for the section: each tilt gets its claim strength labeled (measured / regime-scoped / suggestive) — the paper's credibility rests on never mixing these registers.

### Visualizations

1. **Pseudospectral portrait** (the unifying figure): Λ_ε(J) with +1 marked; σ_min = distance, transient growth, abstention region — one panel doing five jobs.
2. Heatmap of |[R]_{ij}| vs the tier-1 envelope overlaid (causal vs bidir side by side — the "two faces" figure).
3. **DWR four-rung ladder** (actual / dwr_est / reader-bound / global per insert, log scale) — shows soundness *and* the 100× → 10× → 7× tightening in one panel. (The generic bound-vs-measured scatter, one marker per tier, is the fallback version.)
4. Budget-depletion trace: η remaining vs edit count, with the recert trigger.
5. Pipeline block diagram (predict/certify/correct with gate conditions).

### On-the-ground details to internalize (easy to lose through LLM mediation)

These are the things a referee will ask and you must answer without notes:

- **Units of ξ**: hops vs tokens; ρ(G) is *partition-dependent* — the blocking-invariance experiment exists precisely because a referee will catch this. Always report tokens. (Blocking-invariance is analysis, not a run: re-partition the *cached* Jacobians at w/2w/4w, rebuild G, check token-denominated reach agrees.)
- **How ξ is actually measured** (fit_xi): slope of log‖δz‖ vs distance, ξ=−1/slope by least squares — a fitted screening length, *not* a threshold-crossing distance or expectation. Distances are hop-binned (⌈d/W⌉) *before* fitting (the envelope decays per-hop; per-position fits fake ξ=∞). The only cutoff is the noise floor: drop points below 3× measured solver noise (fitting through the converged tail flattens the slope — the v4 "inf" bug). The fit typically survives on n≈3–5 hops: it's a point estimate, don't over-claim precision.
- **What an "edit" physically is**: a change to the input injection h at token i, *not* a weight edit — distinguish from Sinitsin/ROME-style model editing in related work or you'll be misshelved.
- **Everything is linearized at z***: every certificate has a trust region; C2i measures where it ends. Never say "guarantee" without the linearization caveat attached.
- **σ_min is of I−J, not J**, and it's the distance of +1 from the pseudospectrum of J (Λ_ε is *defined* by σ_min(zI−J)≤ε; evaluate at z=1) — you should be able to derive Prop 1 on a whiteboard in 3 lines (δz=Rg; ‖R‖=1/σ_min; done). +1 is special because I−J is what you invert (implicit-function condition), *not* because of f(z*)=z*. Normality nuance: for normal J, near-singular ⟺ a true eigenvalue near +1 (thin contours); the *fat*-pseudospectrum case (+1 pseudo-close while eigenvalues are far) is what non-normality buys — the interesting case, and the causal face.
- **The actual numbers**: d=64, carry rank ~8, 10% mean-mode floor, ξ≈3.1 tokens (segment-average), looseness 7–34×, directional under-predicts at 0.66×, NK fires 0% in trust region. Know which checkpoint each comes from.
- **Tier-2 costs a residual evaluation, not a solve** — this is *why* it's cheap; if asked "why not just re-solve," the answer is the amortized-many-edits regime.
- **What's wrong with tier-2 without NK** (referee question): ‖r‖/σ_min evaluates σ_min *at the current point* and assumes it holds along the whole path to z* — an estimate, not an enclosure (hence the 0.66× under-prediction); and it never certifies z* *exists* (small residual near a fold is compatible with no solution, or two). NK buys existence+uniqueness+enclosure by paying for L — and fires 0% at edit distances. That gap *is* the triangle.
- **DWR numbers**: ladder 1.5–7× / ~10× / ~100×, 100% sound; reader-restriction worth 6–12×; cached adjoint survives (the deployment object). Separations 8.6× (write-once) vs 2.5× (recency — weaker because *selection* among same-colour writes is nonlinear; the linear adjoint is a reach detector, not a selector).
- **η=1 by normalization** (RHS of Stein = −I), so budget numbers are dimensionless margins, not physical tolerances — the conversion to physical tolerance shrinks with λ_max(P).
- **Solver claims are regime-scoped**: Broyden wins on stiff/near-singular at tight tol only; characterization itself is solver-independent.

### Appendices

A. Proofs (full Stein/bordering derivations). B. Routes A/B + metaphor inventory (pruned from main text). C. Substrate training details + the bidir trainability blocker (pointer to workshop paper). D. Negative results in full (dead predictions, QK/RoPE fixes). E. Solver comparison (Anderson/Broyden). F. DWR implementation details. G. Reproducibility: seeds, tolerances, compute (be candid: single-GPU, toy scale — TMLR rewards this).

**Pre-draft gate (updated):** DWR ✓ done (landed, headline result). Two blockers remain, in order: **blocking-invariance** first (cheap analysis over cached Jacobians; referee-proofing for claims already written), then **C7** (it *creates* the evidence for §6/§7). Fallback if C7 slips: §7 ships honestly as "proposal + 11b warm-start measurements" with the budget theorem theory-only — weaker but publishable; §6 with *no* C7 numbers is the bigger wound.
