# Causal vs Bidirectional — the latent divide (living index)

Purpose: track which contributions live on which face, where the threads MERGE, where we HIGHLIGHT the split,
and the balance/focus decision. Update as results land. (Started 2026-07-09.)

The one-line frame: **it is one operator `G` (the block-Jacobi of `I−J`) in two regimes** — causal = the
*nilpotent* corner, bidirectional = the *geometric* corner. The certificate itself is PE-/face-agnostic
(validated: C2 + C2d hold on both `curr` and `currnp`). So this is NOT two papers; it is one spine with two
faces. Balance decision at the bottom.

---

## Shared spine (PE-/face-agnostic — the unification)

- **`σ_min(I−J)` screening length** + the **a-posteriori residual bound** `‖z−z*‖ ≤ resid/σ_min` (+ the
  Newton–Kantorovich upgrade). Deployable certificate, both faces.
- **Two invariants:** `ρ(G)` = reach/shape, `σ_min` = amplitude/scale. "Conditioning, not contraction."
- **The block-transfer certificate:** one `G`; **causal = nilpotent (ρ(G)=0)**, **bidir = geometric (ρ(G)∈(0,1))**.
  ← THE merge point. (Notes #10/#11.)
- **The reader-set principle:** selectivity is possible w.r.t. readers *present at solve time*; unknown/future
  readers force carry in *any* architecture; **causal is the special case where all readers are future.** ←
  unifies must-carry across faces.
- **rank-r carry / Woodbury prior** (C2d-V4, rank ~8 both faces); **conditioning↔recall tension** (QK-norm null);
  **Lyapunov / dynamical-systems lineage** (bidir joined causal's).
- **Noncontractive-but-local witnesses** (the "conditioning, not contraction" evidence): curr40 ρ(J)=8.37 yet
  edit-local (envelope OK); currnp ρ(J)>1 at gaps 0/16/24/40. The claim is shared-spine/PE-agnostic, but the
  ρ>1 **witnesses are CAUSAL-face** (the trained bidir face landed contractive ρ<1). Crowned invariants =
  {ρ(G) reach, σ_min amplitude} over ρ(J); ρ(M)=ρ(I−J) is a non-participant. Poster child = curr40.

## Causal-face contributions (theory depth + field connections)

- **C2d directional certificate** (far-reach map `F_p`, pred_far, the 3-tier claw-back ladder). The scalar
  `σ_min` ball is *vacuous* causally (σ_min = the carry direction) → **direction is the entire content**.
- **Product-Lyapunov / nilpotent structure:** causal `J` block-lower-triangular → `G` strictly-lower →
  nilpotent → Neumann series terminates → exact product form (V5 reconstructs the resolvent to 1e-15).
- **Mamba = the linear/scalar corner** of this face.
- **Must-carry impossibility** (availability argument, architecture-level, theorem-flavored): a causal relay
  can't condition on future queries → forced to carry every binding.
- **C2m metering weak / mode-confounded** here (negative partials at near-singular).
- **RNN-Lyapunov lineage anchor** (Lyapunov spectrum; the general *non-autonomous* / time-varying case).
- `currnp` checkpoint (causal + relative PE); PE-agnosticism validated on it.
- **NEW (2026-07-09):** pointer-chase-to-root causal relay caps at ~1 hop for a content-random layout — a causal
  *limit* that motivates the bidirectional face (see divergence #4).

## Bidirectional-face contributions (use case + application motivation)

- **C2-bidir = the σ_min/conditioning face** and the **use case** (edit-heavy, local readout).
- **Geometric `ρ(G)∈(0,1)`** regime + the **Stein/adapted-norm certificate** (needed because `G` is non-normal
  with ρ∈(0,1); the nilpotent causal corner doesn't need it) → the certified reach.
- **C2m clean output-metering law** (billing legible; Spearman 0.90) — the face that *bills legibly*.
- **Selective forgetting** (readers in context) — but must-carry returns for edit-now/query-later.
- **`ν(J)` less non-normal**; **insert/delete aligned-frame** (bidir + relative PE; 70×→2× positional-shadow).
- **KV-cache serving framing** (editable context; the sound CacheBlend).
- **NEW (2026-07-09):** pointer-chase-to-root is a **bidirectional-face task** (content-random multi-hop chase
  needs two-sided label propagation) → C6 substrate is bidirectional.

---

## Where the threads MERGE (lead with these)

1. "**Two faces of one operator `G`**" (nilpotent vs geometric) — the central unification.
2. The **reader-set principle** — one statement subsuming must-carry on both faces.
3. The **σ_min certificate + residual bound** — PE-/face-agnostic (proved on both substrates).
4. The **Lyapunov lineage** shared by both.

## Where they DIVERGE (highlight the contrast)

1. **Proof family:** nilpotent / product / *directional* (causal — scalar vacuous) vs geometric-`ρ(G)` / Stein /
   *σ_min-envelope works* (bidir).
2. **Billing legibility:** causal mode-confounded vs bidir clean metering law.
3. **Selectivity:** causal *forced-carry* vs bidir *can-forget*.
4. **Task fit:** MQAR is causal-friendly (one backward lookup); **pointer-chase-to-root is bidir-natured**
   (content-random multi-hop) — a concrete task that *motivates* the bidirectional face.
5. **Lens duality:** causal = the general *non-autonomous* case (time-varying Riccati, owed); bidir = the
   *autonomous* special case where the clean scalar object exists.

---

## Balance / focus decision

**Do not pick a side — the unification is the contribution.** Roughly balanced, framed as *one operator, two
regimes*:

- **Spine** = the PE-/face-agnostic certificate (σ_min + block-transfer `G` + residual bound).
- **Bidirectional = the motivation / "home turf" face** — carries *why you'd want this*: KV-cache, edit-heavy
  local readout, selective forgetting, clean metering, and now the pointer-chase task. This is where the
  application story lives.
- **Causal = the theory-depth / field-connections face** — carries *why it's rigorous and connected*: the
  Mamba corner, the RNN-Lyapunov lineage, the must-carry impossibility theorem, the directional certificate.
  After the theory rework this is **substantial, not a sideshow** — it is the theoretical anchor.

So bidir carries motivation, causal carries theoretical weight, and the **block-transfer `G` unification is the
keystone** that makes them one story. Suggested writing order: motivate via the bidir use-case → develop the
unified certificate → a two-faces section (nilpotent + Mamba + directional | geometric + Stein + metering) →
close with the reader-set unification. Reviewers who want rigor get the causal depth; reviewers who want
relevance get the bidir use-case; the keystone stops it reading as two half-papers.

Maintenance: append new elements to the right face as they land (pointer-chase/C6 results, a RoPE conditioning
fix, the time-varying causal Riccati, etc.); promote anything that turns out PE-agnostic into the shared spine.
