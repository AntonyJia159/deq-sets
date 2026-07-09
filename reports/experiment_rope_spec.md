# Scheduled experiment — RoPE conditioning fix (PARKED, not run) — 2026-07-09

STATUS: **scheduled / deferred.** Not highest priority. The paper does NOT depend on it (certificate
already holds PE-agnostically on the ill-conditioned relative substrate `currnp`; C2 + C2d passed). This is a
*tightening + modernity* experiment, not a load-bearing fix. Pick up after the draft spine is written.

## Why (one paragraph)

`currnp` (causal + relative PE) is **2–8× more ill-conditioned** than `curr` (absolute PE) at matched gaps
(larger κ, lower σ_min mid-gaps). QK-norm fixed conditioning but **tanked recall** (peaking↔contraction:
capping logit magnitude starves the peaky attention recall needs) — a clean NULL, recorded. RoPE is the one
un-ruled-out simple option: a **norm-preserving orthogonal rotation** of Q/K that leaves logit
magnitude/sharpness alone, and replaces the *learned additive relative bias* `relb` (which can grow large and
sharpen attention toward near-singularity — the same "‖posw‖ grows with gap" pathology) with a fixed rotation.
**Hypothesis:** improves σ_min / closes the κ gap **without** QK-norm's recall cost. (Mechanism reasoning,
untested — same epistemic status QK-norm had before we ran it, and QK-norm's recall surprise is the caution.)

## What to build

- New trainer `experiments/curriculum_currnp_rope.py`, cloned from `curriculum_currnp.py`, with:
  - `ROPE = True`, and **`REL_BIAS = False`** (RoPE *replaces* the learned additive bias — do not stack them;
    stacking reintroduces the `relb` blow-up we're trying to remove).
  - Keep `BIDIR=False, READONLY_Q=True, NO_POSW=True`, same window curriculum
    (PHASE_A=[(2,400),(4,400),(10,600)], PHASE_B gaps). Save `currnprope00-40.pt`.
- RoPE application in `sliding_window_reach.py` `f()`: rotate `q,k` by position-dependent angles *before* the
  `q@kᵀ` score (standard RoPE: split head_dim into pairs, apply per-position rotation). Gate behind a
  `ROPE = False` flag next to `QK_NORM` so it's off by default and PE-agnostic loaders stay unaffected.
- `c2_edit_locality.py` already restores substrate flags from the checkpoint — add `sw.ROPE = ck.get("rope",
  False)` next to the existing `qk_norm` restore so `currnprope` runs through C2/C2d unchanged.

## What to measure (the decision gate)

Run C2 + C2d on `currnprope`, compare **against `currnp`** (the honest baseline — NOT against `curr`):
1. **Conditioning:** κ(I−J) and σ_min across gaps {0,4,16,24,40}. Win = closes the 2–8× κ gap vs `currnp`.
2. **Recall:** MQAR recall. Constraint = must **not** drop vs `currnp` (this is where QK-norm failed: −15..−24 pts).
3. **Certificate tightness:** ξ_faber envelope — should get *less loose* if conditioning improves (bounds are
   sound either way; this just tightens tier-1).

**Verdict rule:** conditioning↑ AND recall flat/↑ → adopt RoPE as primary relative PE (modern + tighter).
Conditioning↑ but recall↓ → second instance of the peaking↔contraction tension (report as a *pair* with
QK-norm — "even the norm-preserving fix pays recall" would be a stronger finding than the QK-norm null alone).
No conditioning change → RoPE is cosmetic/modernity-only; keep `relb`, note it.

## Cost estimate

~3 currnp-scale training runs (the min pointer-chase runs were ~200–380s each; currnp curriculum is heavier —
budget a few hours on the 4050) + C2/C2d eval. Cheap enough to run in one sitting when prioritized.

## On pick-up

Record result in `bidirectional_run_findings.md` §11 and `project_deq_theory` memory; if adopted, promote to the
shared spine in `causal_bidir_index.md` and update the skeleton PE line. Either way it resolves the last
"pending" flag in `paper_skeleton.md`.
