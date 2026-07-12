# Causal vs Bidirectional — the latent divide (living index)

Purpose: track which contributions live on which face, where the threads MERGE, where we HIGHLIGHT the split,
and the balance/focus decision. Update as results land. (Started 2026-07-09.)

The one-line frame: **it is one operator `G` (the block-Jacobi of `I−J`) in two regimes** — causal = the
*nilpotent* corner, bidirectional = the *geometric* corner. The certificate itself is PE-/face-agnostic
(validated: C2 + C2d hold on both `curr` and `currnp`). So this is NOT two papers; it is one spine with two
faces. Balance decision at the bottom.

**UPDATE 2026-07-12.** Two stale items corrected below: (1) **"forced-carry" is over-stated** — must-carry
is not an architectural impossibility on the causal face; it conflates *model-storage* (broad on both faces,
**objective**-driven — C2t) with *certificate-selectivity* (available on both faces when readers are **known** —
colored-recall adjoint recovered the true dep 83% *causally*). Genuine carry-forcing survives only under
**open-ended generation** (unknowable future readers) — a knowability condition, not the mask. (2) The
**"bidir = use-case / market" framing is superseded by TWO REGISTERS** (causal = *diagnostic* for the big
generative ecosystem, bidir = *clean certificate* for the small equilibrium/science niche; deployment claim
revised down, scientific claim kept — see paper_skeleton 2026-07-12 recalibration). Loose-ends/gaps section
appended at the bottom.

---

## Shared spine (PE-/face-agnostic — the unification)

- **`σ_min(I−J)` screening length** + the **a-posteriori residual bound** `‖z−z*‖ ≤ resid/σ_min` (+ the
  Newton–Kantorovich upgrade). Deployable certificate, both faces.
- **Two invariants:** `ρ(G)` = reach/shape, `σ_min` = amplitude/scale. "Conditioning, not contraction."
- **The block-transfer certificate:** one `G`; **causal = nilpotent (ρ(G)=0)**, **bidir = geometric (ρ(G)∈(0,1))**.
  ← THE merge point. (Notes #10/#11.)
- **The reader-set principle (sharpened 2026-07-12):** the axis that governs selectivity is reader-**KNOWABILITY**,
  not directionality. Certificate-selectivity (restrict edit-response to a reader-set) is available whenever the
  readers are known/present — on **either** face (measured: colored-recall adjoint recovered the true dep 83% on
  the *causal* face). It is denied only for unknown/future readers, in **any** architecture. The causal mask is
  merely the special case whose *default deployment mode* (open generation) has unknowable readers — it is not
  itself the forcing condition. ← unifies must-carry across faces via knowability, not the mask.
- **rank-r carry / Woodbury prior** (C2d-V4, rank ~8 both faces); **conditioning↔recall tension** (QK-norm null);
  **Lyapunov / dynamical-systems lineage** (bidir joined causal's).
- **Noncontractive-but-local witnesses** (the "conditioning, not contraction" evidence): curr40 ρ(J)=8.37 yet
  edit-local (envelope OK); currnp ρ(J)>1 at gaps 0/16/24/40. The claim is shared-spine/PE-agnostic, but the
  ρ>1 **witnesses are CAUSAL-face** (the trained bidir face landed contractive ρ<1). Crowned invariants =
  {ρ(G) reach, σ_min amplitude} over ρ(J); ρ(M)=ρ(I−J) is a non-participant. **Poster child = currnp40**
  (ρ=4.44, single fixed point, solver-checked Anderson≡Broyden); currnp16/24/40 = the ρ 1.26→4.44 trend.
  curr40 REFRAMED (was the poster child): it is MULTISTABLE (two branches 0.14 apart, σ_min 0.026/0.040, both
  ρ=8.37) → now the "strong non-contraction ⇒ multistability, local certificate robust to it" datapoint, not
  the clean witness. See digest §11 witness-solver-check.

## Causal-face contributions (theory depth + field connections; the DIAGNOSTIC register)

- **C2d directional certificate** (far-reach map `F_p`, pred_far, the 3-tier claw-back ladder). The scalar
  `σ_min` ball is *vacuous* causally (σ_min = the carry direction) → **direction is the entire content**.
- **Product-Lyapunov / nilpotent structure:** causal `J` block-lower-triangular → `G` strictly-lower →
  nilpotent → Neumann series terminates → exact product form (V5 reconstructs the resolvent to 1e-15).
- **Scalar/linear-recurrence limit** of this face (a plain linear recurrence: our σ_min certificate generalizes
  its edit-decay rate). NOTE: describe generically as "linear recurrences / linear sequence models" — do NOT
  name Mamba/SSMs (discretization/scan lineage baggage that isn't ours; see `feedback_naming`).
- **Must-carry under open generation** (SOFTENED from "impossibility, theorem-flavored" — 2026-07-12): a causal
  relay in *open-ended generation* can't condition on unknowable future queries → forced to carry every binding.
  This is genuine — but it is a **knowability** claim (future readers unknown), NOT an architectural property of
  the mask: on a *fixed* causal sequence with a *known* downstream query the reader-set escape IS available
  (colored-recall, 83%). Do not state it as a causal-face impossibility theorem.
- **C2m metering weak / mode-confounded** here (Spearman ~0.70 vs bidir ~0.90; negative partials) — **CONFIRMED
  SOLVER-INDEPENDENT (2026-07-12, `c2m_metering_broyden`):** holds under Broyden too, so it's a real ν/non-normality
  property, not an Anderson artifact. On the non-normal face ‖Δz‖ (standard norm) under-determines cost for any solver.
- **RNN-Lyapunov lineage anchor** (Lyapunov spectrum; the general *non-autonomous* / time-varying case).
- `currnp` checkpoint (causal + relative PE); PE-agnosticism validated on it.
- ~~pointer-chase-to-root as a causal limit~~ **RETRACTED (2026-07-09):** it is NOT a directionality effect —
  causal ≈ bidir (both ~0.68), it's a model-capacity ceiling. C6 demoted to future work; see digest §11c.

## Bidirectional-face contributions (the CLEAN-CERTIFICATE register; equilibrium/science home)

- **C2-bidir = the σ_min/conditioning face** and the **use case** (edit-heavy, local readout).
- **Geometric `ρ(G)∈(0,1)`** regime + the **Stein/adapted-norm certificate** (needed because `G` is non-normal
  with ρ∈(0,1); the nilpotent causal corner doesn't need it) → the certified reach.
- **C2m clean output-metering law** (billing legible; Spearman 0.90) — **PROVISIONAL 2026-07-12:** whether this
  is a *bidir* property or an *affine-invariant-solver* property is under test (Broyden re-run); it may hold on
  both faces once the solver is fixed.
- **Selective forgetting** — CAVEAT (2026-07-12): C2t showed the trained query-visible *bidir* model did NOT
  actually forget selectively (objective doesn't reward it). So "bidir can forget" = *availability* (readers in
  context), not a realized behavior of these models; and the same availability holds causally when readers are
  known. Must-carry returns for edit-now/query-later (unknowable future readers) — on either face.
- **`ν(J)` less non-normal**; **insert/delete aligned-frame** (bidir + relative PE; 70×→2× positional-shadow).
- **KV-cache serving framing** (editable context; the sound CacheBlend).
- ~~pointer-chase-to-root is a bidirectional-face task~~ **RETRACTED (2026-07-09):** refuted — bidir ≈ causal
  (~0.68), the cap is model capacity, not directionality. C6/reader-set = discussion principle + future work.

---

## Where the threads MERGE (lead with these)

1. "**Two faces of one operator `G`**" (nilpotent vs geometric) — the central unification.
2. The **reader-set principle** — one statement subsuming must-carry on both faces.
3. The **σ_min certificate + residual bound** — PE-/face-agnostic (proved on both substrates).
4. The **Lyapunov lineage** shared by both.

## Where they DIVERGE (highlight the contrast)

1. **Proof family:** nilpotent / product / *directional* (causal — scalar vacuous) vs geometric-`ρ(G)` / Stein /
   *σ_min-envelope works* (bidir).
2. **Billing legibility:** causal mode-confounded vs bidir clean metering law. **CONFIRMED SOLVER-INDEPENDENT
   (2026-07-12, `c2m_metering_broyden.py`, n=126/face):** causal Spearman(n_warm,‖Δz‖)≈**0.70** vs bidir≈**0.90**
   under BOTH Anderson AND Broyden; causal partial-corr stays negative (−0.51 Anderson, −0.35 Broyden), bidir ~0
   both. My "it's an Anderson artifact" prediction died on measurement. Mechanism REFINED (not spectral-gap):
   ‖Δz‖ in the *standard* norm is a faithful cost proxy only for a **near-normal** operator (bidir); on the
   non-normal causal face the natural geometry ≠ standard norm, so ‖Δz‖ under-determines cost for *any* solver.
   So this divergence is a genuine **ν** consequence — the SAME root as the nilpotent/geometric split — not a
   solver quirk. (Broyden did soften the causal partial −0.51→−0.35 and halve cold cost 34→19 evals, so a *small*
   part was spectral-gap; the Spearman gap is untouched.)
3. ~~**Selectivity:** causal *forced-carry* vs bidir *can-forget*~~ **RETRACTED/REFRAMED (2026-07-12) — this was
   miscategorized on both halves.** Not a mask divergence: model-storage is broad on *both* faces (objective-
   driven, C2t — the query-visible bidir model didn't forget either), and certificate-selectivity is available
   on *both* faces when readers are known (colored-recall 83% causal). The real axis is **reader-knowability**
   (readers-known → selectivity available; open-generation/edit-now-query-later → forced carry), which cross-cuts
   the mask. See the 2×2 in the gaps section.
4. **Task fit:** MQAR is causal-friendly (one backward lookup). ~~pointer-chase-to-root is bidir-natured~~
   RETRACTED — it's directionality-INDEPENDENT (causal≈bidir), capacity-bound; not a face-divergence example.
   (content-random multi-hop) — a concrete task that *motivates* the bidirectional face.
5. **Lens duality:** causal = the general *non-autonomous* case (time-varying Riccati, owed); bidir = the
   *autonomous* special case where the clean scalar object exists.

---

## Balance / focus decision

**Do not pick a side — the unification is the contribution.** Framed as *one operator, two regimes*, now as
**two registers** (2026-07-12 recalibration — supersedes the old "bidir = market/use-case" split, which oversold
bidir as a deployment target):

- **Spine** = the PE-/face-agnostic certificate (σ_min + block-transfer `G` + residual bound).
- **Bidirectional = the CLEAN-CERTIFICATE register** — the face where the certificate is *non-vacuous and tight*
  (the Green's tent). Its home is the small-but-real **equilibrium/science** niche (fixed-point reasoning,
  perturb-a-boundary-condition edit-sensitivity), NOT a big serving market. This is where a certificate is both
  tight *and* wanted. (KV-cache stays a running illustration, not an adoption claim.)
- **Causal = the DIAGNOSTIC register + theory-depth face** — carries *why it's rigorous and connected* (the
  linear-recurrence limit, the RNN-Lyapunov lineage, the directional certificate, the nilpotent product form),
  AND is where the clean certificate *fails* → it ships as a **locality diagnostic/warning** for the big
  generative ecosystem (the near-singular carry subspace = the ripple channel; degrades to the residual bound +
  needs a known reader). A useful negative for MEMIT-style editors, not a maintenance pitch. (Linear-recurrence
  limit = a light generic hook, not a Mamba/SSM comparison — see `feedback_naming`.)

So the registers match where each ecosystem actually is: bidir = a tight certificate for the (small) place
fixed-point models live; causal = a diagnostic for the (large) place generative models live. The **block-transfer
`G` unification is the keystone**. Writing order: develop the unified certificate → a two-faces/two-registers
section (nilpotent + directional + diagnostic | geometric + Stein + tight certificate) → close with the
reader-set (knowability) unification. Own, don't hide, that the clean result sits on the face today's ecosystem
uses least.

Maintenance: append new elements to the right face as they land (pointer-chase/C6 results, a RoPE conditioning
fix, the time-varying causal Riccati, etc.); promote anything that turns out PE-agnostic into the shared spine.

---

## Loose ends & generative gaps (2026-07-12, parallel to the digest pass)

**Loose ends (referee-facing):**
1. **"Must-carry impossibility" was over-stated** (fixed above): it conflated model-storage (broad on both faces,
   objective-driven — C2t) with certificate-selectivity (available on both faces when readers known — colored-
   recall 83% causal). Genuine forcing = open generation (unknowable readers), a knowability condition, not the
   mask. Divergence #3 retracted.
2. **Billing-legibility divergence (#2) — TESTED, SURVIVED (2026-07-12).** `c2m_metering_broyden.py` (n=126/face):
   causal Sp≈0.70 vs bidir≈0.90 under BOTH solvers → solver-independent, a genuine ν property. The "second Anderson
   artifact" hypothesis was WRONG (my prediction, killed by the run — good hygiene). Mechanism refined to
   norm-geometry mismatch (‖Δz‖ tracks cost only when near-normal). Divergence #2 stands, on firmer ν footing.
3. **"Two invariants {ρ(G), σ_min}"** — say *distinct/complementary*, not *independent*. All trained points so far
   lie on the correlated diagonal (near-singular ⇔ long-reach). The decorrelation lever already exists: an **MLP
   is block-diagonal → moves σ_min but not ρ(G)** → an off-diagonal point in the (σ_min, ρ(G)) plane. Same run as
   the entity-tracking MLP ablation.
4. **Lens duality "causal = non-autonomous / time-varying Riccati (owed)"** is a promissory note, never built — a
   field-connection gesture, not load-bearing. Build the minimal version or demote from a "divergence" to a remark.

**Generative gaps:**
- **A. The axis that governs selectivity is reader-KNOWABILITY, not directionality — a cleaner organizing frame
  than the mask.** The corrected must-carry implies a 2×2, {causal, bidir} × {readers known, readers unknown}:
  selectivity is available in *both* readers-known cells (colored-recall = causal-known; C2-bidir = bidir-known)
  and denied in *both* readers-unknown cells (open generation = causal-unknown; edit-now/query-later = bidir-
  unknown). The mask only sets each face's *default* knowability regime. This subsumes must-carry more cleanly
  than "causal carries / bidir forgets" and is the sharper statement of the reader-set principle.
- **B. The face divergences are collapsing to ONE structural one** (the digest kill-pattern, here too). Of the
  five listed: selectivity (#3) is knowability/objective, not mask; billing (#2) may be solver, not mask (under
  test); task-fit (#4) already half-retracted (pointer-chase); lens-duality (#5) is unbuilt. What survives
  solver-/objective-/training-INVARIANT is **proof family (#1): G nilpotent (causal) vs geometric (bidir)** — the
  structural spectral property. Honest consequence: the defensible two-faces claim is *one structural divergence
  (nilpotent vs geometric G) + several contingent behavioral ones that keep reducing to non-mask causes.* This
  STRENGTHENS "two regimes of one operator G" (the surviving divergence IS the G-property) while deflating "the
  faces behave differently in many ways." Make it explicit — smaller, more defensible claim.
- **C. The reader-set principle (merge #2) has no certificate — it needs the DWR / adjoint bound** (the digest's
  identified keystone). The colored-recall 83% recovery *was* a partial realization (goal-oriented adjoint reach).
  Turning the principle into the dual-weighted-residual (goal-oriented) certificate is what makes it operational
  AND is the escape from the impossible triangle (sound AND tight) AND the tool that reopens the causal register
  (linear ξ=∞ is vacuous; the nonlinear reader-set is sparse). Same missing object as the digest keystone —
  highest-value, independently actionable (derivation + validation on existing ckpts), no new task/use-case needed.
