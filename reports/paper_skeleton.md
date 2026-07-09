# Paper skeleton — working title, abstract, spine (draft, 2026-07-09)

STATUS: working draft. Pending before finalizing — (a) a simple conditioning fix for the relative substrate
(RoPE is the un-ruled-out option after the QK-norm null); (b) C6 (bidirectional pointer-chase reader-set test).
Title/abstract will firm up once those land.

---

## Title (working)

**Editable Equilibria: Certifying Local Edits in Deep Equilibrium Transformers**
*(subtitle / hook: Conditioning, Not Contraction)*

Alternatives: "Certifying Local Edits in Equilibrium Transformers: Conditioning, Not Contraction";
"A Conditioning Certificate for Local Edits in Deep Equilibrium Transformers".

Note: "conditioning, not contraction" is demoted from *thesis* to *hook* — the headline is the certificate.

## Abstract (few-sentence draft)

> A Deep Equilibrium (DEQ) transformer represents a sequence as the fixed point of a single weight-tied layer.
> We show that such a fixed point's response to an in-context edit is governed by the **conditioning** of `I−J`
> — its smallest singular value `σ_min` — rather than by contraction: an equilibrium can be recall-capable and
> provably **edit-local even when strongly noncontractive** (`ρ(J) ≫ 1`). We turn this into an *a-priori*
> certificate that splits edit-reach into **two invariants** — a block-transfer transport rate `ρ(G)` (how far
> an edit propagates) and `σ_min` (how much it amplifies) — and a deployable *a-posteriori* bound
> `‖z−z*‖ ≤ ‖f(z)−z‖/σ_min` that certifies any partial recompute from the residual alone. The same operator
> yields **two regimes** — a nilpotent *causal* corner (met by a directional certificate) and a geometric
> *bidirectional* corner (the edit-heavy / KV-cache setting) — unified by a **reader-set principle** for which
> edits must propagate. The upshot: an equilibrium state is a KV cache whose invalidation region is a
> **theorem, not a heuristic**.

## Spine (maps 1:1 to `causal_bidir_index.md`)

1. **Hook** — conditioning, not contraction (noncontractive-but-local; poster child curr40, ρ(J)=8.37 yet ξ≈1 hop).
2. **Certificate (spine)** — two invariants `{ρ(G) reach, σ_min amplitude}`; the a-posteriori residual bound
   `resid/σ_min` (+ Newton–Kantorovich); PE-agnostic (proved on both `curr` and `currnp`).
3. **Two faces** — nilpotent *causal* (directional certificate) | geometric *bidirectional* (Stein/adapted-norm,
   clean metering; the KV-cache use case). One operator `G`, two regimes.
4. **Reader-set principle** — which edits must propagate; unifies must-carry across faces.
5. **Motivation** — editable context / certified KV-cache invalidation (bidir); pointer-chase task (C6).

## Framing decisions (locked)

- Demote "conditioning, not contraction" to hook; headline the certificate.
- Crown = **two** invariants `{ρ(G), σ_min}` over `ρ(J)`; `ρ(M)=ρ(I−J)` is a non-participant (don't feature it).
- Feature the **noncontractive witnesses** (curr40; ρ>1 currnp) as explicit evidence — they are causal-face.
- **Do NOT name Mamba / SSMs casually.** Too overengineered; drags in discretization / scan lineages that are
  not ours; and no one does editability on them anyway. If a linear limit is worth naming at all, say "linear
  recurrences" / "linear sequence models" generically — or omit. See `feedback_naming` memory.
- Balance: don't pick a face; bidir = motivation/use-case, causal = theory-depth/connections, block-transfer
  `G` unification = keystone.
