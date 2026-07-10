# Paper skeleton — working title, abstract, spine (draft, 2026-07-09)

STATUS: working draft. The RoPE conditioning fix is **SCHEDULED / PARKED, not blocking** — spec in
`experiment_rope_spec.md`; it is a tightening + modernity experiment, NOT load-bearing (certificate already
holds PE-agnostically on the ill-conditioned `currnp`; C2 + C2d passed). Title/abstract can firm up now; RoPE
result only *tightens* tier-1 or pairs with the QK-norm null. C6 (pointer-chase reader-set test) is FUTURE
WORK — capacity-ceilinged at toy scale (digest §11c); the reader-set ships as a discussion-level *principle*
(evidenced by must-carry + C2t on both faces), not a headline experiment.

---

## Title (working)

**Editable Equilibria: Certifying Local Edits in Deep Equilibrium Transformers**
*(subtitle / hook: Conditioning, Not Contraction)*

Alternatives: "Certifying Local Edits in Equilibrium Transformers: Conditioning, Not Contraction";
"A Conditioning Certificate for Local Edits in Deep Equilibrium Transformers".

Note: the headline is the **two-tier certificate**; the hook is **certificates-vs-heuristics**;
"conditioning, not contraction" is demoted all the way to a **corollary** (the noncontractive witnesses).

## Abstract (working draft — problem-first, two-tier, maintenance-framed)

> When an in-context edit changes one token, which downstream computation must be redone? Today this is answered
> *heuristically*, with no guarantee the reused part is still correct. In a **Deep Equilibrium (DEQ) transformer**
> — a sequence as the fixed point of a single weight-tied layer — we make it a **two-tier certificate**.
> *A priori*, an edit's reach obeys a **geometric envelope** whose decay rate is a block-transfer transport
> radius `ρ(G)` (certified via a Stein/adapted-norm bound) — it bounds *how far* an edit propagates and screens
> the positions that could change. *At runtime*, a **residual bound** `‖z−z*‖ ≤ ‖f(z)−z‖/σ_min` certifies from
> the candidate alone that a partial recompute is within tolerance — the deployable tier. Because reach is set
> by `ρ(G)` and error by `σ_min` — **not** by the contraction rate `ρ(J)` — an equilibrium stays edit-local
> *even when strongly noncontractive* (a corollary, not the thesis). The same operator splits into **two
> regimes** — a nilpotent *causal* corner (a directional certificate) and a geometric *bidirectional* corner
> (the edit-heavy / KV-cache setting) — unified by a **reader-set principle** for which edits must propagate.
> Operating in the warm-start neighborhood of a cached equilibrium — exactly where in-context editing lives —
> this turns the invalidation region of a KV cache into a **theorem, not a heuristic**.

Rationale for the opener: DEQs are a minority architecture, so a flat "theoretical study of DEQ edit-locality"
loses non-partisan readers by sentence two. Hook via **certificates-vs-heuristics** (KV-cache as the concrete
instance) — honest (no system claimed) and it gives the reader a stake. Do NOT lead with DEQ-benefits-in-general
(architecture advocacy) or an applications pitch (over-promises at toy scale); weave the "equilibrium is what
*makes* the certificate exist" point in later, and keep applications to the one payoff sentence.

Two-tier naming discipline (fixes an earlier conflation — the abstract must not backslide): tier-1 = the
**geometric / Stein reach envelope** (rate `ρ(G)`) = *how far*, sound-but-loose (screening/characterization);
tier-2 = the **residual bound** = *how much / did I recompute enough*, **sound and deployable**.
"Conditioning" is a tier-2 (σ_min) word — do NOT use it to describe the a-priori reach. "Conditioning, not
contraction" is a **corollary** (noncontractive witnesses), not the throughline.

TIER-2 TIGHTNESS DISCIPLINE (fixes an over-claim; `c2_postedit_certify`, 2026-07-09): do NOT call the SCALAR
`resid/σ_min` "tight" — for REAL edits it is LOOSE 7–34× (real edit residuals are fast-mode/broadband, cos with
the stiff subspace 0.01–0.14, so the worst-case-over-stiff scalar over-charges). The IMPOSSIBLE TRIANGLE (measured, `ratio_dir`): the three
certs each give up ONE corner. SCALAR `resid/σ_min` = SOUND + cheap, NOT tight (7–34× on real edits). DIRECTIONAL
`‖R·r‖` (resolvent action, C2d/Woodbury-updatable) = TIGHT + cheap (0.98–1.01 exact on small edits, 10–34×
tighter than scalar) but NOT rigorous — it UNDER-predicts on large/near-singular edits (ratio_dir 0.66), so it's
a tight *estimate/heuristic*, not a guaranteed bound. NK `R₋` = RIGOROUS + tight but NOT cheap/wide (a certified
*stopping rule*, near-convergence only — 0% of one-shot post-edit residuals are in its trust region on ill-cond
cells). PICK TWO. Deployment upshot: the tight deployable object is effectively a heuristic; the one PRINCIPLED
route to sound-AND-tighter is a different axis — the **reader-set / goal-oriented (adjoint / DWR) bound** (bound
only the OUTPUT at reader positions, drop far-field slack). Deployment claim scoped accordingly (matches "don't
oversell"); reader-set bound = flagged open direction.

## Maintenance-regime framing (the trust-region view-shift, 2026-07-09)

The bounds live **near `z*`** (Test 1's linear-regime gate; the Kantorovich `h=βLη` caveat). This is NOT a
limitation — it is the **use case**: in-context editing = hold a cached equilibrium `z*`, make a local edit,
**warm-start from `z*`** → you are near the new fixed point *by construction*. So the **trust region = the
warm-start neighborhood = the maintenance regime**; "only near `z*`" is the operating point, not a caveat.
View the result as a **maintenance certificate for local edits**, not a global "conditioning governs
edit-locality" claim. Honest division of labor: tier-1 envelope *plans* the recompute ball (sound over-set,
loose), tier-2 residual *verifies* it (tight). Value = the **guarantee** in the maintenance regime, not big
speedups (consistent with the modest-efficiency finding: Woodbury prior + per-window early-stop both
safe-but-modest).

## Logical DAG (do NOT collapse to "downstream of σ_min")

- **ROOT:** the resolvent `M⁻¹ = (I−J)⁻¹` — the linear edit-response; exists *because* we're at a fixed point.
- **TWO SIBLING invariants of `M⁻¹`** (independent spectral facts, NOT parent-child):
  - `σ_min(I−J)`: `‖M⁻¹‖ = 1/σ_min` = **amplitude**.
  - block-Jacobi `ρ(G)`: spatial decay rate of `M⁻¹`'s blocks = **reach**.
- **TWO certificates:** a-posteriori `‖z−z*‖ ≤ resid/σ_min` (downstream of **σ_min**) | a-priori reach envelope
  `‖[M⁻¹]_{ij}‖ ≤ C·ρ(G)^d` (downstream of **ρ(G)**, *not* σ_min).
- **σ_min plays TWO roles:** (i) a **positivity gate** — `σ_min>0` ⟺ `I−J` invertible ⟺ well-posed ⟺ the
  resolvent exists — *upstream of everything*; (ii) a **magnitude** — `1/σ_min` = amplitude — parent of the
  residual bound only, sibling of `ρ(G)`.
- So the spine is **resolvent → two invariants → two certificates**, NOT "σ_min → everything." Writing it as
  "downstream of σ_min" reverts to the old single-invariant framing and undersells the reach machinery
  (block-transfer + Stein), which is the more technical contribution.

## Spine (maps 1:1 to `causal_bidir_index.md`)

1. **Hook** — certificates vs heuristics: which downstream computation an edit invalidates, made a theorem
   (KV-cache as the concrete instance). Scoped as a **maintenance** certificate (warm-start neighborhood).
2. **Certificate (spine), two tiers** — tier-1 **geometric/Stein reach envelope** (rate `ρ(G)`) = *how far*,
   sound-but-loose (screens the recompute ball); tier-2 **residual bound** = *did I recompute enough*, **sound
   and deployable** (scalar `resid/σ_min` = loose screen; directional `‖R·r‖` = tight; Newton–Kantorovich = a
   certified *stopping rule*, near-convergence). PE-agnostic (proved on both `curr` and `currnp`).
3. **Two faces** — nilpotent *causal* (directional certificate) | geometric *bidirectional* (Stein/adapted-norm,
   clean metering; the KV-cache use case). One operator `G`, two regimes.
4. **Reader-set principle** — which edits must propagate; unifies must-carry across faces.
5. **Corollary** — reach set by `ρ(G)`, error by `σ_min`, not by contraction `ρ(J)` → edit-local even when
   noncontractive (poster child **currnp40, ρ(J)=4.44**, single fixed point, solver-checked; currnp16/24/40 =
   the ρ 1.26→4.44 trend). *A corollary, not the throughline.* (curr40 ρ=8.37 is MULTISTABLE — reframed as the
   "strong non-contraction ⇒ multistability, local certificate robust" datapoint, not the clean witness.)
6. **Motivation** — editable context / certified KV-cache invalidation (bidir). (Reader-set as a principle;
   hub/spoke task = future work, capacity-ceilinged at toy scale.)

## Where the certificate lands — outlook pointers (one hook + three unequal gestures)

Framing levels, kept separate (resolves the KV-cache-vs-honesty tension): **identity** = core-ML (the
characterization + certificate primitive; portfolio wants this, TMLR fits it) — this is what the paper *is*.
**Hook** = KV-cache, as Geng advises, but scoped as a *running illustration* of the primitive, NOT a deployment/
speed claim (the only failure mode is claiming a serving benchmark — don't). **Pointers** = a *breadth* of honest
gestures — breadth is itself the core-ML "general primitive" signal (lands in serving + verification + science =
generality, the opposite of an applied paper). AI4Science is the truest *deployment* home but stays a POINTER,
not the frame (ZJ already has two applied works; this one's identity is core-ML by choice).

A short **"Where the certificate lands"** discussion paragraph, unequal weight, each 1–3 sentences:
1. **Serving — certified KV-cache invalidation** (the hook, carried through). The certificate = which cached
   computation an edit invalidates, made a theorem. Illustration of the primitive; not an adoption claim
   (no one switches to DEQ-LMs for this soon — say so implicitly by scoping, don't oversell).
2. **Robustness certification** (α,β-CROWN / IBP / monDEQ lineage). Input-robustness = our edit-response, same
   resolvent `(I−J)⁻¹∂f/∂x`. We add what black-box-Lipschitz lacks: `σ_min` = a *local* Lipschitz at the operating
   point, and *position-resolved* decay `C·ρ(G)^{d}` (tight for ℓ0 / patch / token attacks). The reframe: our
   worst-case scalar bound — "loose" for typical edits — is the certifier's *tight* bound (adversary picks the
   stiff direction). Honest limit: we're local/first-order; a sound over-the-ball enclosure needs the interval-
   Newton / Krawczyk extension of NK (conservative). Niche = structured local cert, not "first DEQ robustness cert."
3. **Scientific / continuation UQ** (the truest deployment home; a pointer). "Editing" → parametric re-solves
   (continuation, sweeps, adjoint sensitivity) — a small perturbation to a cached equilibrium = our maintenance
   regime. The residual bound is the *verification* half of V&V (numerical error to the model's fixed point, not
   model-vs-reality); compose with classical/Bayesian calibration (Kennedy–O'Hagan) → a genuine total error bar.
   "Certified numerical error you can drop into a UQ stack" is native currency there.

## Framing decisions (locked)

- Hook via **certificates-vs-heuristics** (not "conditioning, not contraction" — that's now a corollary).
- **Two-tier naming discipline:** tier-1 = geometric/Stein reach envelope (ρ(G), *how far*, loose); tier-2 =
  residual bound (*how much*, **sound & deployable** — scalar loose screen / directional tight / NK stopping
  rule; NOT "tight" for the scalar). "Conditioning" is a σ_min/tier-2 word — never use it for the a-priori reach.
- **Maintenance-regime framing:** the bounds live near z*, which = the warm-start neighborhood = the use case,
  NOT a limitation. Frame as a maintenance certificate for local edits; value = the guarantee, not big speedups.
- Crown = **two** invariants `{ρ(G), σ_min}` over `ρ(J)`; `ρ(M)=ρ(I−J)` is a non-participant (don't feature it).
- Feature the **noncontractive witnesses** — poster child **currnp40 (ρ=4.44)**, trend currnp16/24/40
  (ρ 1.26→4.44), all solver-checked (Anderson≡Broyden); causal-face. curr40 (ρ=8.37) is MULTISTABLE → the
  "strong non-contraction ⇒ multistability, local certificate robust to it" datapoint, not the clean witness.
- **Do NOT name Mamba / SSMs casually.** Too overengineered; drags in discretization / scan lineages that are
  not ours; and no one does editability on them anyway. If a linear limit is worth naming at all, say "linear
  recurrences" / "linear sequence models" generically — or omit. See `feedback_naming` memory.
- Balance: don't pick a face; bidir = motivation/use-case, causal = theory-depth/connections, block-transfer
  `G` unification = keystone.
