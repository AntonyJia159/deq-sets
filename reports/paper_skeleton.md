# Paper skeleton — working title, abstract, spine (draft, 2026-07-09)

STATUS: working draft. The RoPE conditioning fix is **DONE — NEGATIVE** (2026-07-10): RoPE both *replacing*
and *on top of* the learned relative bias COLLAPSES recall (the cross-window relay plateaus ~0.80 vs currnp's
clean 1.0), so it joins QK-norm as a *second* failed attempt to lift `σ_min` → the **conditioning↔recall
tension has no cheap architectural escape** (see the "two failed fixes" subsection below). The certificate is
untouched — PE-agnostic, C2 + C2d hold on the ill-conditioned `currnp`. C6 (pointer-chase reader-set test) is
FUTURE WORK — capacity-ceilinged at toy scale (digest §11c); the reader-set ships as a discussion-level
*principle* (evidenced by must-carry + C2t on both faces), not a headline experiment.

UPDATE (2026-07-12): folded in the new material since the RoPE note — (1) a **deployment recalibration**
(revise the deployment claim *down*, keep the scientific claim; the two faces become **two registers** —
causal = a *diagnostic* for the big ecosystem, bidir = a *clean certificate* for the small equilibrium/science
one); (2) the **semantic validation** (colored-recall near-singular causal, segment-average well-conditioned
bidir Green's tent) + the **entity-tracking program** that unifies them on one substrate with an **MLP knob**;
(3) **positioning** vs the entity-tracking/binding literature and the bidir-in-mech-interp gap; (4) the
**emergent-reach / iterated-map (NCA)** framing that answers "you just built in locality with the window."
See the four dated sections below.

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

## Conditioning↔recall tension — the near-singularity is intrinsic (two failed fixes, 2026-07)

The ill-conditioning our certificate operates around is **not an artifact to engineer away** — it is *dual to
the model doing its job*. Two independent, simple architectural attempts to **lift `σ_min`** on the near-singular
`currnp` substrate both **failed by collapsing recall**:
- **QK-norm (cosine attention, 2026-07-09):** `σ_min` ~doubled (0.03–0.04 → 0.07–0.08) but recall dropped
  **15–24 pts** — capping logit *magnitude* starves the peaky attention long-range recall needs.
- **RoPE (2026-07-10):** the orthogonal, norm-preserving alternative (can't cap peaking, so a-priori expected
  harmless) instead **disrupts cross-window relay formation outright** — present *or* absent the learned relative
  bias, recall plateaus ~0.80 vs currnp's clean 1.0 (stable, not under-training; extra steps don't move it).

So `σ_min` and recall move in **lockstep**: you cannot buy conditioning without paying recall, by *two* orthogonal
mechanisms (magnitude-capping vs relay-disruption). This is the empirical face of **edit-locality is dual to
forgetting** — a model with long memory *must* sit near-singular, which is exactly why the certificate is needed
(the near-singularity is **load-bearing**, not fixable). Use as a **robustness / motivation** point: the two nulls
are *evidence for* the tension, not failures to hide. Scope caveat: small single-tied-layer substrate; a tuned/
multi-layer RoPE isn't formally excluded, but the *simple drop-in fix* is ruled out (don't rabbit-hole). The
certificate itself is untouched — PE-agnostic (Jacobian-level), holds on `curr` and `currnp`. Full saga (from the
approximation-theory origin of `σ_min` to its response-height role) recapped in report §0 of Note #11.

## Deployment recalibration — two registers, honest sizing (2026-07-12)

Revise the **deployment** claim pessimistically; do **not** revise the scientific one; relocate where each face
earns its keep rather than dropping bidir.

- **Own the scope limitation up front.** Decoder-only is the frontier; knowledge editing (ROME/MEMIT) runs on
  *causal* decoders; the bidir encoders that survive (RAG, ModernBERT) edit the **corpus**, not the transformer
  state. "Bidir-edit transformers are a big/rising market" would not survive review — do NOT let the intro lean
  on market size. The clean certificate sits on the face today's ecosystem uses *least*; state that as scope.
- **The contribution is the characterization, not the use-case.** "*When* is a local edit certifiable, and why"
  *requires both poles*: the causal face is exactly where the clean certificate **fails** (near-singular, ξ=∞,
  reader-set-must-be-known); the bidir face is where it is **non-vacuous** (the Green's tent). You cannot state
  the taxonomy with one face. TMLR rewards this scoping result independent of market size (venue-fit, not pitch).
- **Two registers — the reframe that makes both faces pull weight, each matched to where its ecosystem is:**
  - **Causal = a *diagnostic* for the big ecosystem.** Where knowledge-editing actually happens we do not hand a
    clean certificate — we hand a **warning**: the near-singular carry subspace *is* the ripple channel, so the
    certificate degrades to the residual bound and needs a *known* reader. A useful negative for MEMIT-style
    editors (tells them *when an edit is not certifiably local*).
  - **Bidir = a *clean certificate* for the small-but-real equilibrium/science ecosystem.** Where fixed-point
    models genuinely live, the certificate is tight and deployable (the Green's-function result).
- **Honest sizing (tiered):** (a) bidir DEQ transformer as a deployed edit-target — *small, flat*; (b)
  fixed-point / equilibrium reasoning in perception/science/control (optical flow, MPC, physics-informed,
  graph/structure equilibria) — *real but modest*, undirected/bidir, and IS edit-sensitivity (perturb a boundary
  condition → certify local response = literally our Green's tent); **the one niche where the clean certificate
  is both tight and wanted**; (c) iterative-refinement / diffusion denoisers (full-attention, bidir) — the
  genuinely *large* iterative ecosystem, **if** the conditioning/reach analysis ports to a denoiser's
  edit-response = **big-if-true, currently unearned → speculative bridge, labeled as such**.
- **One-line pitch (replaces any market pitch):** *"a taxonomy of when local edits are certifiable in
  equilibrium transformers — a **tight certificate** on the well-conditioned (equilibrium/science) face, a
  locality **diagnostic** on the ill-conditioned (generative) face — and the clean result sits on the face
  today's ecosystem uses least."* Smaller claim, actually true, limitation owned not hidden.

## Semantic validation & the entity-tracking program (2026-07-12)

Beyond MQAR, two semantic substrates now exercise both faces × both conditioning regimes; a *unified* successor
task is spec'd (**not yet built**).

- **colored-recall (causal, selection) → near-singular.** Latest/earliest same-color value; σ_min small, ξ=∞
  (resolvent blocks don't decay). Learns + length-generalizes on *latest* (recency); *earliest* (write-once,
  long-range) does NOT length-generalize — governed by a **nonlinear retrieval horizon, not the linear reach**
  (honest scope: linear certificate reach ξ ≠ nonlinear task memory). Reader-set from *known* query positions
  recovered the true dependency ~83% → **corrects** "reader-set escape impossible on the causal face": it is
  denied only for *unknown future* readers (open generation), not for the causal mask per se (reader-KNOWABILITY
  is the condition).
- **segment-average (bidir, aggregation) → well-conditioned.** Exp-distance-weighted mean within content-marked
  segments, factored `[mode|value]` + regression head; ρ~0.8, σ_min~0.06–0.12; length-generalizes (rel_err~0.27
  flat to L=64 untrained). Edit validation = **the clean payoff colored-recall could not give**: the resolvent
  certifies a compact two-sided **Green's tent** (corr 0.955, ξ=3.1 *finite*, 91% energy inside segment,
  linear≈nonlinear) — edit-locality demonstrated on the well-conditioned BVP face.
- **Why the pair is patchy (the confound to fix):** the two co-vary **four** axes at once — face (causal/bidir),
  operation (select/aggregate), output-head (classification/regression), input-encoding (discrete/real). So
  "conditioning drives it" is not cleanly isolated; a reviewer will want a *motivated* pair, not two ad-hoc tasks.
- **Successor — the entity-tracking family (SPEC).** One domain (records of entities whose states update, then
  are queried), the contrasts as **controlled knobs on ONE real-valued/regression substrate** (kills the head +
  encoding confound: select copies one real value-part, aggregate means the value-parts — the **only** difference
  is peaked vs flat attention, so any σ_min flip is attributable to conditioning alone; the tightest form of the
  thesis). Controlled **2×2**: conditioning axis = select (peaked → near-singular) vs aggregate (smooth →
  well-conditioned); face axis = causal state-tracking vs bidir cloze. **Ground-truth reader-sets = the entity
  bindings** (sparse for select, dense for aggregate, cluster for cloze) = the object the certificate's reach is
  checked against. Edit-locality = the **knowledge-editing ripple** (edit an entity's state → which readers
  change) → *the task IS the application*. Subsumes colored-recall (causal,select) and segment-average's regime
  (aggregate) inside one family — the fix for "non-comparable and patchy."
  - **Honest asymmetry (surfaced while spec'ing — state it, don't paper over it):** genuine two-sided
    *constraint* is intrinsically a smoothing/aggregation phenomenon; selection is intrinsically *disjunctive*
    (pick-one). So bidir+aggregate is a real both-sides BVP, but bidir+select can only be coref/redundancy fill
    ("any mention, possibly future" = bidirectional *access*, not both-sided *constraint*); and a *mutable-state*
    stream admits **no** cloze at all (a later overwrite doesn't constrain an earlier value — future evidence
    exists only if the attribute is a *constant fact* redundantly asserted, or a *smoothing* field). The face and
    conditioning axes are therefore **not perfectly orthogonal** — bidir-select's value is testing whether R's
    reader-set correctly reaches a *future* position when the mask permits (a known masked reader → knowable →
    certificate applies), not two-sided constraint.
  - **The MLP knob (per ZJ: MLPs are not inert; pure attention is too unrealistic).** An MLP is per-position →
    **block-diagonal Jacobian** → adds to the diagonal `D` of `J` → **predicted to move σ_min (conditioning) but
    leave ρ(G) (reach) and the nilpotent/geometric two-face split invariant.** Sharp falsifiable test: run each
    near-singular (select) cell with MLP off/on; σ_min *should* lift (a Lipschitz-nice MLP regularizes the
    peaking, softening the tension) while reach-structure holds — if σ_min doesn't move or ρ(G) does, the
    block-diagonal picture is wrong (informative either way). And **multi-attribute** `(entity,attribute)→value`
    is a *conjunctive* bind (match both keys) — the classic thing attention does poorly and an MLP rescues — so
    the realistic enrichment is exactly where the MLP *earns its place* (answers "pure attention is unrealistic").
    Residual connections = a separate, sharper knob (they change `I−J` **structurally**, not just `D`).

## Positioning — entity tracking, binding, and the bidir gap (2026-07-12)

- **The recognized task is only our (causal, select) cell.** Entity/state tracking in the literature = *discrete
  retrieval of one entity's final state, queried at the end, read by classification*: bAbI (Weston 2015), the
  "boxes" task (Kim & Schuster 2023), world-state probing on Alchemy/TextWorld (Li, Nye & Andreas 2021), and the
  mech-interp binding line — Prakash et al. 2024 (binding *heads*) and Feng & Steinhardt 2023 (abstract
  *binding-ID* vectors, binding by position/order). **Aggregate and bidir cells are OUR extensions** manufacturing
  the conditioning/face axes — do NOT dress them as canonical. (Verify exact venues/years before citing.)
- **The mech-interp binding finding corroborates our σ_min prediction:** binding = positional/order selection = a
  peaked pick onto one position = a near-unit-gain soft mode of `I−J` = **near-singular**. We give a
  *conditioning-theoretic* account of the same phenomenon they characterize circuit-wise — and none of them
  measure *edit-response / knowledge-editing ripple* on the tracking task (they probe / read accuracy / trace
  circuits). That's genuinely our angle. **Adopt the recognizable dress:** anchor on (causal,select) in
  boxes-style *move*-operation framing; cite binding results as external support for the near-singularity.
- **Why bidir is under-studied — pre-empt the reviewer.** Mech-interp grew up *after* the pivot to decoder-only
  (~2020+), so its whole toolkit (induction heads, MQAR, patching) is *causal*, and the canonical synthetic tasks
  are intrinsically sequential (unnatural to pose bidirectionally). Bidir's own tradition is **probing, not
  circuits** (BERTology — Rogers et al. 2020; structural probes — Hewitt & Manning 2019; head analysis — Clark et
  al. 2019). Our resolvent reader-set is the *Jacobian-of-the-equilibrium* version of input-attribution — which
  is why it is a real contribution on the bidir face AND why nobody checked (the community that would have wasn't
  looking at bidir).

## Emergent reach, not imposed locality — the iterated-map framing (2026-07-12)

Sharpens *why the certificate is non-vacuous* and heads off the "you just built in locality with the window"
objection.

- **The window imposes the per-hop *granularity* (J banded), not the *reach*.** The resolvent `(I−J)⁻¹` of a
  banded `J` is generically **full** — solving to equilibrium propagates influence across arbitrarily many hops;
  the band does not survive the inverse. What is *emergent* (learned, conditioning-set) is the **decay rate** of
  the resolvent's blocks = the actual edit-reach ξ, governed by ρ(G), σ_min — NOT by the window. (Causal mask
  imposes the exact *upstream* zeros; the window imposes granularity; conditioning imposes decay.)
- **Our own two tasks prove reach is emergent:** same windowed architecture, same `W` — colored-recall came out
  ξ=∞ (blocks don't decay, non-local), segment-average ξ=3.1 (decaying, local). A factor of ∞ apart under one
  architecture ⇒ the window sets the *grid*, the *conditioning* sets the *reach*.
- **The real contrast is finite-depth vs iterated-to-steady-state (NOT conv-vs-attention).** Finite-depth local
  nets have reach = depth×kernel (capped, structural, *tautological to certify*) — the only genuinely trivial
  bucket. Iterated/equilibrium maps (DEQ, **NCA**, diffusion denoisers) have no depth cap: a still-local kernel
  reaches globally through the resolvent, and only conditioning re-caps it. **NCA is not a trivial foil** — it is
  iterated indefinitely, so it must be read through *powers* of its update map, and `(I−J)⁻¹ = Σ Jⁿ` is exactly
  the Neumann/geometric sum of those powers (NCA studies the *transient* `Jⁿ`; we study the *accumulated* `Σ Jⁿ`
  — same operator spectrum, transient vs. return-to-attractor). NCA is a **crude, under-formalized cousin** whose
  headline result (local regeneration) is *uncertified* emergent edit-locality; our σ_min/reach machinery is the
  rigorous instrument it lacks — a motivating precedent, not a foil to dismiss.
- **Scope boundary this draws:** stay on attention transformers + (opt-in) MLP — the regime where reach is
  emergent and the certificate has content. Convolutional *finite-depth* vision is the trivial slice (imposed,
  capped). The one non-trivial slice of vision is *attention-based generative* (DiT / diffusion-transformer
  inpainting = "generative fill / iterative local regeneration") — the **same** emergent-reach object on 2D
  tokens, and the honest bridge to the large use-case (ties to the speculative diffusion pointer above). Note it
  once; do not become a vision paper.

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
3. **Two faces = two registers** — nilpotent *causal* (directional certificate; where the clean certificate
   *fails* → ships as a **diagnostic/warning** for the big generative ecosystem) | geometric *bidirectional*
   (Stein/adapted-norm, clean metering; where the certificate is **non-vacuous** → the clean certificate for the
   small equilibrium/science ecosystem). One operator `G`, two regimes; the register split matches where each
   ecosystem actually is (see 2026-07-12 recalibration).
4. **Reader-set principle** — which edits must propagate; unifies must-carry across faces.
5. **Corollary** — reach set by `ρ(G)`, error by `σ_min`, not by contraction `ρ(J)` → edit-local even when
   noncontractive (poster child **currnp40, ρ(J)=4.44**, single fixed point, solver-checked; currnp16/24/40 =
   the ρ 1.26→4.44 trend). *A corollary, not the throughline.* (curr40 ρ=8.37 is MULTISTABLE — reframed as the
   "strong non-contraction ⇒ multistability, local certificate robust" datapoint, not the clean witness.)
6. **Motivation** — editable context / certified KV-cache invalidation (bidir). (Reader-set as a principle;
   hub/spoke task = future work, capacity-ceilinged at toy scale.)

## Where the certificate lands — outlook pointers (one hook + three unequal gestures)

READ WITH the **2026-07-12 deployment recalibration** above: these pointers stand, but the *framing* is now
"two registers" (causal = diagnostic, bidir = clean certificate) and the sizing is explicitly tiered/owned — do
not let this section drift back into a market pitch. A fourth, **speculative** gesture (diffusion denoisers) is
added below and labeled unearned.

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
4. **Iterative-refinement / diffusion denoisers** (SPECULATIVE, unearned — flag as such). Full-attention
   denoisers are the *large* iterative ecosystem, and "regenerate this region → does it ripple?" (DiT/diffusion
   inpainting, generative fill) is the *same emergent-reach object* on 2D tokens (§emergent-reach). IF the
   conditioning/reach analysis ports to a denoiser's edit-response, this is the big use-case — big-if-true only;
   do NOT claim it, gesture once. This is the honest bridge from "small equilibrium/science niche" to scale.

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
- Balance: don't pick a face; **two registers** — bidir = the *clean certificate* (equilibrium/science niche) +
  theory-clean pole, causal = the *diagnostic/warning* for the big generative ecosystem + theory-depth/connections;
  block-transfer `G` unification = keystone. (Supersedes the earlier "bidir=use-case, causal=theory" split, which
  oversold bidir as a market — see 2026-07-12 recalibration; deployment claim revised *down*, scientific claim kept.)
- **Emergent reach is the non-vacuity argument** (pre-empt "you built in locality with the window"): the window
  imposes per-hop granularity, the *equilibrium* uncaps reach, *conditioning* re-caps it (colored-recall ξ=∞ vs
  segment-average ξ=3.1, same architecture). Stay on attention + opt-in MLP; conv/finite-depth vision is the
  trivial (imposed-capped) slice; NCA is a crude *cousin* (iterated, powers ≈ our Neumann-sum resolvent), not a foil.
- **Semantic tasks:** the colored-recall/segment-average pair is a *confounded* demo (co-varies 4 axes); the
  entity-tracking family (one real-valued substrate, controlled 2×2, reader-sets = entity bindings, MLP knob) is
  the motivated successor — SPEC only, not built. Anchor on the recognized (causal,select) cell; aggregate + bidir
  are our extensions, not canonical entity-tracking.
