# Literature review — the two-faces edit-locality certificate (sequence direction)

Targeted novelty scan (2026-07-03) for the theory in Report #9: is the causal product–Lyapunov certificate,
the bidirectional σ_min/Faber certificate, and their use for *edit-locality of an equilibrium attention model*
already published? Verdict up front: **both proof families are mature, off-the-shelf theory we cite (good —
rigor rests on solid ground); the unoccupied sliver is applying them as a *spatial edit-reach certificate* on
a *nonlinear equilibrium* attention model, plus the two-regime split and the maintenance/warm-start reading.**
Caveat: originally based on targeted web search (snippets). **2026-07-07: all 4 starred refs now read in full — corrections folded in below (see "Full-read corrections").**

---

## Front 1 — RNN Lyapunov / Jacobian-product theory  → the causal face's home (ESTABLISHED)

The causal certificate (influence = product of per-step Jacobian blocks; decay rate = Lyapunov exponent /
singular values of the long-term Jacobian) is **classical RNN dynamical-systems theory**. We do NOT own it.

- **★ Vogt, Puelma Touzel, Shea-Brown, Lajoie, "On Lyapunov Exponents for RNNs" (arXiv 2006.14123, Frontiers
  Appl. Math. Stat. 2022).** LEs characterize information propagation; singular values of the long-term
  Jacobian (= product of per-step Jacobians) regulate how signals propagate across time steps. *This is our
  product-Lyapunov statement, in the RNN setting.*
- Engelken et al, "Lyapunov spectra of chaotic RNNs" (2006.02427, Phys. Rev. Research 2023).
- "Gradient Flossing: … Dynamic Control of Jacobians" (2312.17306) — controls the Jacobian-product spectrum
  during training; establishes the vanish/explode ⇔ product-norm growth/decay correspondence operationally.

**Implication for us:** cite Vogt et al as the parent theory of the causal face. Our contribution is *not* the
product-decay math; it is (i) applying it to **edit-locality / maintenance** rather than gradient propagation,
(ii) on an **equilibrium attention** cell (not a trained-forward RNN), and (iii) tying it to σ_min. Do **not**
present the product form as novel mathematics. It is prior art we generalize the *use* of.

## Front 2 — Selective SSM memory/decay theory  → Mamba's home (ESTABLISHED; partial overlap with our duality)

- **★ Cirone, Orvieto, Walker, Salvi, Lyons, "Theoretical Foundations of Deep Selective SSMs" (2402.19047,
  NeurIPS 2024).** Signature/path view; input-controlled transitions; high-order dependencies.
- "Mathematical Formalism for Memory Compression in Selective SSMs" (2410.03158).
- "Controllability Analysis of State-Space-based LM" (2511.17970) — control-theory lens on SSM LMs.
- SD-SSM (dense transition dictionary + softmax router, universality for regular languages) — the closest
  "dense transition matrices" precedent, but still SSM, not an attention equilibrium.

**Partial-overlap flag (forces an honesty edit).** This literature *already notes* that the state-transition
spectrum controls **both memory horizon and the response amplitude to input perturbations** ("the learned
memory kernel controls not just the memory horizon but also … response amplitude to input perturbations").
So the **forward** half of our "one number, two readings" duality is at least gestured at in SSM-land. What is
**not** there: the **maintenance/edit reading** — a *mid-context edit* and the *re-solve operation* it triggers
— because an SSM's fused state has no such operation. **Adjust the claim:** the forward sensitivity is
partially known; the *maintenance* reading and its warm-start operation are the unpriced part.

## Front 3 — Non-normal / field-of-values matrix decay  → the bidirectional face's home (ESTABLISHED math)

The FOV/Faber decay theory for inverses (and functions) of non-Hermitian banded matrices is mature
numerical analysis; we cite, we don't invent.

- **★ Benzi & Golub / Benzi & Razouk, "Bounds for the entries of matrix functions … " (BIT; Benzi decay-bounds
  line).** Chebyshev-on-interval → Faber-on-field-of-values; the exact machinery Report #9 invokes.
- "Improved Approximation Bounds for Moore–Penrose Inverses of Banded Matrices … LQ Control" (2411.04400) —
  **applies banded-inverse decay to control (an initial-value/causal structure)**: precedent that these bounds
  live on causal systems too, useful bridge cite.
- "Quasiseparable LU decay bounds for inverses of banded matrices" (2506.16339); non-Hermitian Toeplitz
  pseudospectra / eigenvector-decay transitions (2512.03757) — the pseudospectra story behind "FOV can contain
  0 while invertible" (our vacuity caveat, now confirmed as a known phenomenon).

**Implication:** the bidirectional face is standard applied math; our job is the correct per-regime application
and the σ_min-tightness measurements, not new decay theorems.

## Front 4 — Edit-locality / incremental maintenance of transformer or DEQ hidden states  → THE GAP (mostly OPEN)

- DEQ sensitivity/robustness: Subhomogeneous DEQ (2403.00720), monotone/contractive well-posedness — study
  perturbation sensitivity for **adversarial robustness / well-posedness**, via IFT δz=(I−J)⁻¹δ, but **do not**
  turn (I−J) conditioning into a **spatial screening length / edit-locality** statement. (Search explicitly:
  "(I−J) inverse conditioning bounds … were not detailed.")
- Incremental transformers / TAPIR (2305.10845), "Reconsidering the Past: Optimizing Hidden States"
  (2112.08653): revise/recompute hidden states, but by **autoregressive suffix recompute or learned revision**,
  with **no locality certificate** (which tokens *can* be affected).
- Knowledge editing (ROME/MEMIT; "Superimposed Noise in Sequential Knowledge Editing" 2505.07899): edits
  **weights**, a different problem from maintaining a fixed point under **input** edits.

**Verdict:** the specific object — a **certified spatial edit-reach** (screening length) for the **fixed point
of a sparse-attention equilibrium**, via (I−J) conditioning (bidirectional) / product-Lyapunov (causal), enabling
a **warm-start local re-solve** with a σ_min uniqueness → exact-vs-branch-tracking guarantee — appears
**unoccupied**. Adjacent, not overlapping.

---

## Net novelty statement (what to claim, what to cite)

**Cite as parent theory (do not claim):** product-of-Jacobians / Lyapunov decay (Vogt et al) for the causal
face; Benzi FOV/Faber decay for the bidirectional face; IFT δz=(I−J)⁻¹δ for DEQ sensitivity; SSM memory-kernel
↔ perturbation-response (Cirone et al) for the *forward* half of the duality.

**Claim (the sliver):** (1) reading the resolvent as a **spatial edit-reach certificate** on a **nonlinear
equilibrium attention** model (not gradient propagation, not adversarial robustness, not weight editing);
(2) the **two-regime IVP/BVP split** with per-regime certificates matched to the causal/bidirectional attention
directions; (3) the **maintenance operation** — warm-start local re-solve — with σ_min uniqueness as the
exact-vs-branch-tracking boundary; (4) the **maintenance reading** of the memory↔reach duality (forward reading
is partly prior art). Framing contribution: a **translation** placing the numerical-analysis / RNN-dynamics
decay lenses onto the equilibrium-transformer maintenance object — legitimate scholarship, stated as such.

**Honesty edits forced by this review:**
- Report #9 / blueprint: the product–Lyapunov certificate must be presented as *RNN-Lyapunov theory applied to
  edit-locality*, citing Vogt et al — not as new math. (Already hedged in #9's "linear special case" framing;
  make the citation explicit.)
- The "memory horizon never read as edit-reach" line must soften to "the *forward* response-amplitude reading
  exists in SSM theory; the *maintenance/edit* reading and its re-solve operation are the unpriced part."

**Must-read-in-full before any paper text:** ★ Vogt 2006.14123, ★ Cirone 2402.19047, ★ Benzi–Golub decay
bounds, ★ 2411.04400 (banded-inverse decay for control = the causal/IVP precedent). 2606.02680 (locality≠
reachability) remains the closest *reachability* precedent (Report #8-era note).

---

## Full-read corrections (2026-07-07)

All four read in full. Two snippet-era framings were wrong or over-claimed; two decay papers reweighted.

- **Vogt 2006.14123 — narrow further; it is NOT a bound source.** In its own words it "positions LEs as an
  interpretive lens," an empirical training-stability readout — not a theorem paper. The Lyapunov exponent is a
  **trajectory average** (asymptotic log-growth of ∏Jₜ over a long orbit); our σ_min(I−J) is a **static, finite,
  single-map** object. Take exactly two things: (i) the forward-info ⟺ backward-gradient **duality** (anchors the
  BPTT bridge), (ii) the precedent that the **product-of-Jacobians spectrum** is the right object for the
  non-normal/causal direction. Do NOT imply Vogt supplies any bound we use. One-sentence anchor.
- **Cirone 2402.19047 — even more limited than flagged; fixes a mis-attribution above.** Cirone is a
  **signatures / rough-path expressivity** result (hidden state = low-dim projection of the input signature).
  It is **not** about conditioning, memory horizon, or response-amplitude. The "state-transition spectrum
  controls both memory horizon AND perturbation-response" overlap in Front 2 (lines ~39–45) does **not** come
  from Cirone — it belongs to the memory-compression/decay line (2410.03158, Mamba-decay analyses). What we take
  from Cirone: only *"rigorous modern SSM theory exists and centers input-controlled linear recurrences"* — an
  anchor for the SSM leg of the unification, nothing about the duality.
- **Benzi–Golub — confirmed as the SYMMETRIC ROOT only.** Exact: banded Hermitian A, |[f(A)]ᵢⱼ| decays exp in
  |i−j|; for the inverse, Demko–Moss–Smith base **q=(√κ−1)/(√κ+1)**, bandwidth m divides the exponent
  (q^{|i−j|/m}). But it requires **Hermitian (normal)** A. Our J is only *near*-normal (ν>0), so the
  load-bearing cite for our actual regime is the **FOV/Faber generalization** (Benzi–Razouk / Benzi–Boito:
  Chebyshev-on-interval → Faber-on-field-of-values). Benzi–Golub = clean baseline; Faber/FOV = what we invoke.
  The Hermitian→near-normal citation boundary IS the two-faces split (ν the discriminator).
- **2411.04400 (Shin–Tan–Anitescu) — UPGRADE from bridge-cite to σ_min-native backbone.** Stated in **singular
  values** directly (not a translated proxy); covers the **Moore–Penrose pseudoinverse** and **rank-deficient**
  case; prefactors depend on **conditioning, not dimension**. This is the rigorous handle Benzi–Golub can't
  give: at the **carry-subspace / near-singular edge** (σ_min→0), the pseudoinverse **restricted to the
  transverse complement still decays exponentially** — exactly the object C2m's ‖R·δh‖ forecast and the
  carry/transverse SVD split rely on. So it is the backbone of the claw-back ladder + carry decomposition, not a
  footnote. Provenance gift: it is an **LQ-control** perturbation bound for a banded saddle/KKT system → reframes
  our edit-locality as "sensitivity of a banded fixed-point/optimal-control solution to a local perturbation,"
  an object mature control theory already bounds. Good outer-framing sentence.
