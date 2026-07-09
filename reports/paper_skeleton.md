# Paper skeleton — working title, abstract, spine (draft, 2026-07-09)

STATUS: working draft. Pending before finalizing — a simple conditioning fix for the relative substrate (RoPE
is the un-ruled-out option after the QK-norm null). C6 (pointer-chase reader-set test) is now FUTURE WORK —
capacity-ceilinged at toy scale (digest §11c); the reader-set ships as a discussion-level *principle* (already
evidenced by must-carry + C2t on both faces), not a headline experiment. Title/abstract firm up after RoPE.

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
tier-2 = the **residual bound** `resid/σ_min` = *how much / did I recompute enough*, tight and deployable.
"Conditioning" is a tier-2 (σ_min) word — do NOT use it to describe the a-priori reach. "Conditioning, not
contraction" is a **corollary** (noncontractive witnesses), not the throughline.

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
   sound-but-loose (screens the recompute ball); tier-2 **residual bound** `resid/σ_min` (+ Newton–Kantorovich)
   = *did I recompute enough*, tight and deployable. PE-agnostic (proved on both `curr` and `currnp`).
3. **Two faces** — nilpotent *causal* (directional certificate) | geometric *bidirectional* (Stein/adapted-norm,
   clean metering; the KV-cache use case). One operator `G`, two regimes.
4. **Reader-set principle** — which edits must propagate; unifies must-carry across faces.
5. **Corollary** — reach set by `ρ(G)`, error by `σ_min`, not by contraction `ρ(J)` → edit-local even when
   noncontractive (curr40, ρ(J)=8.37 yet ξ≈1 hop). *A corollary, not the throughline.*
6. **Motivation** — editable context / certified KV-cache invalidation (bidir). (Reader-set as a principle;
   hub/spoke task = future work, capacity-ceilinged at toy scale.)

## Framing decisions (locked)

- Hook via **certificates-vs-heuristics** (not "conditioning, not contraction" — that's now a corollary).
- **Two-tier naming discipline:** tier-1 = geometric/Stein reach envelope (ρ(G), *how far*, loose); tier-2 =
  residual bound resid/σ_min (*how much*, tight, deployable). "Conditioning" is a σ_min/tier-2 word — never
  use it for the a-priori reach.
- **Maintenance-regime framing:** the bounds live near z*, which = the warm-start neighborhood = the use case,
  NOT a limitation. Frame as a maintenance certificate for local edits; value = the guarantee, not big speedups.
- Crown = **two** invariants `{ρ(G), σ_min}` over `ρ(J)`; `ρ(M)=ρ(I−J)` is a non-participant (don't feature it).
- Feature the **noncontractive witnesses** (curr40; ρ>1 currnp) as explicit evidence — they are causal-face.
- **Do NOT name Mamba / SSMs casually.** Too overengineered; drags in discretization / scan lineages that are
  not ours; and no one does editability on them anyway. If a linear limit is worth naming at all, say "linear
  recurrences" / "linear sequence models" generically — or omit. See `feedback_naming` memory.
- Balance: don't pick a face; bidir = motivation/use-case, causal = theory-depth/connections, block-transfer
  `G` unification = keystone.
