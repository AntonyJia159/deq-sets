# DEQ-Transformer edit-locality — experiment blueprint & honest scoping

Design doc for the sequence direction (Geng meeting 2026-07-02). Substrate is validated: on a first
probe (`seq_recall_probe.py`) a softmax-attention DEQ solves MQAR associative recall (acc 1.00) that a
linear-attention / SSM cell cannot (0.25), and its edits are warm-start-exact — but **edit-locality is
over the attention graph, not sequence distance**. So sparse / sliding-window attention is the
load-bearing choice: it makes the attention graph metric-local, so the edit-response resolvent decays
with distance (Demko–Moss–Smith / Faber), giving a *certified* reach.

`Mamba : us  ::  InstantGNN : us` — the linear incumbent gets cheap incrementality from linearity; we
detach it (get it from σ_min conditioning) and buy a more expressive nonlinear member of the same
maintainable class.

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

## Honest scope — what this is NOT (do not overclaim)

- **NOT a decode/generation speedup.** The causal KV cache is already optimal for append; a DEQ
  re-solves per token (overhead), and at inference it stores O(n·d) equilibrium state ≈ a cache — the
  famous "O(1) memory" is a **training** property, not an inference one. **Drop any decode-speed claim.**
- **NOT cheap for long-range-relevant edits in a pure sliding-window model.** An edit that must reach a
  far generation point costs O(distance) — fundamental (information must travel), not fixable by any
  method. Partially resolved by the multi-scale arm (C4).
- **The genuine regime** = edit-heavy / local-readout: code editing (re-predict near the cursor — PIE's
  setting), agent scratchpad revision, RAG chunk swaps re-read locally. The *characterization* (recompute
  exactly the certified ξ-ball) also transfers to standard transformers; the DEQ is the clean setting
  where it is provable.

---

## Claims to test

- **C1 (reach).** A sliding-window softmax DEQ solves **cross-window** recall (gap `G ≫ w`) via
  equilibrium propagation, *beyond* a matched `K`-step unroll of the same cell (reach capped at `K·w`).
- **C2 (maintainability).** A value edit is warm-start-exact (warm==cold); `|Δz|` **decays with sequence
  distance** (metric-local, unlike dense attention), screening length `ξ` set by conditioning;
  `ξ ≤ Faber bound` (sound), tight when well-conditioned.
- **C3 (tradeoff).** Window `w` dials solve-iterations (mixing) vs edit-reach `ξ` (locality) — a Pareto
  curve. (Small `w`: slow solve, local edits. Large `w`: fast solve, global edits.)
- **C4 (multi-scale resolution).** Adding `O(log n)` coarse / global nodes lets a *long-range-relevant*
  edit reach the generation point in `O(log n)` via the coarse channel (local ball + `O(log n)` coarse
  updates) instead of full-suffix recompute — at the cost of the coarse nodes being bounded (`O(log n)`)
  locality-breaking hubs. This is the concrete answer to "how does the signal reach the end without a
  whole recomputation."

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
