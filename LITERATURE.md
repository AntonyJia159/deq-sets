# Literature review — running list

Papers to cite / position against in the eventual write-up. Each entry notes its
**role** relative to our project (set-DEQs with path-independence / exact unlearning).

Status legend: ✅ read & verified · 🟡 skimmed / from search, verify before citing ·
⬜ identified, not yet read.

Role legend: **[backbone]** build on it · **[contrast]** related but different goal ·
**[theory]** anchors a claim · **[baseline]** compare against · **[prior]** cite as background.

---

## A. DEQ core & training machinery

- 🟡 **[prior]** Bai, Kolter, Koltun — *Deep Equilibrium Models*, NeurIPS 2019.
  arXiv:1909.01377. The foundational DEQ: solve `z=f(z,x)`, implicit-function-theorem
  backward pass. Note: already reports LayerNorm "stabilizes" — relevant to our
  normalization framing (do NOT claim we discovered this).
- 🟡 **[prior]** Bai, Koltun, Kolter — *Multiscale Deep Equilibrium Models (MDEQ)*,
  NeurIPS 2020. arXiv:2006.08656. Multi-resolution states; the template for DEQs
  whose output shape matches input — relevant to set/measure I/O.
- 🟡 **[prior/theory]** Bai et al. — *Stabilizing Equilibrium Models by Jacobian
  Regularization*, ICML 2021. arXiv:2106.14342. Explicitly notes DEQs "diverge without
  LayerNorm" but frames it as brittleness to patch via Jacobian reg. Key citation for
  our "norm is load-bearing but under-theorized" point.
- 🟡 **[backbone]** Geng et al. — *Phantom Gradients* (On training implicit models),
  NeurIPS 2021. Cheap approximate DEQ gradient. Advisor's work; we use it via torchdeq.
- 🟡 **[backbone]** Geng & Kolter — *TorchDEQ: A Library for Deep Equilibrium Models*,
  arXiv:2310.18605, 2023. The implementation substrate we build on.
- 🟡 **[theory]** Winston & Kolter — *Monotone Operator Equilibrium Networks (monDEQ)*,
  NeurIPS 2020. Provable unique fixed point via monotone-operator parametrization. The
  principled-uniqueness alternative we contrast normalization against.
- 🟡 **[theory]** *Subhomogeneous Deep Equilibrium Models*, 2024. arXiv:2403.00720.
  Existence/uniqueness via subhomogeneity conditions on weights/activations. Another
  point on the "constraints for uniqueness" axis.
- 🟡 **[theory]** Liu, Ding, Osher, Yin — *Expressive Power of Implicit Models: Rich
  Equilibria and Test-Time Scaling*, 2025. arXiv:2510.03638. Proves expressiveness
  scales with test-time iterations. Backs our test-time-scaling claim.

## B. Implicit / equilibrium models on graphs

- 🟡 **[prior]** Gu et al. — *Implicit Graph Neural Networks (IGNN)*, NeurIPS 2020.
  arXiv:2009.06211. Fixed-point GNN; well-posedness via Perron-Frobenius bound on weight
  norms. The closest conceptual ancestor (equilibrium over a relational structure); our
  setting is sets rather than graphs, downstream loss rather than node labels.

## C. Set / permutation-invariant architectures

- 🟡 **[prior/backbone]** Zaheer et al. — *Deep Sets*, NeurIPS 2017. The sum/mean-pool
  decomposition of permutation-invariant functions. Our DeepSets update derives from this.
- 🟡 **[prior/backbone]** Lee et al. — *Set Transformer*, ICML 2019. Attention-based
  permutation-invariant set encoder. Our attention update is in this family.
- 🟡 **[baseline]** Qi et al. — *PointNet / PointNet++*, CVPR 2017. Canonical point-set
  networks; DDEQ uses them as baselines, we likely will too.

## D. DEQ on sets / measures — the nearest neighbors

- ✅ **[backbone]** Geuter, Bonet, Korba, Alvarez-Melis — *DDEQs: Distributional Deep
  Equilibrium Models through Wasserstein Gradient Flows*, AISTATS 2025. arXiv:2503.01140.
  **The closest prior work.** Extends DEQs to point clouds viewed as discrete measures;
  forward pass = Wasserstein gradient flow on MMD; defines the EI property (equivariant in
  latent Z, invariant in input X); attention-encoder architecture; torchdeq-based.
  → **Our relationship (per ZJ review):** their architecture (EI property, attention
  encoders, measure-to-measure) is a strong **backbone** for what we want. **Our
  departure:** DDEQ targets *geometric* point clouds where the output is itself a shape to
  match a ground-truth distribution (hence Wasserstein/MMD). We operate in *abstract
  semantic* feature space with a *downstream task loss* — so we do NOT need the
  Wasserstein/MMD machinery and can use standard solvers. We also focus on properties they
  do not study: path independence, exact unlearning, streaming/dynamic sets, train-small/
  infer-large. Their stated limitations (slow MMD-flow forward pass; no scaling/unlearning
  experiments) are our opening.

- ✅ **[contrast]** Özcan, Shi, Ioannidis — *Learning Set Functions with Implicit
  Differentiation*, AAAI 2025 (ext. arXiv:2412.11239). Uses implicit differentiation to
  train *set functions* for the optimal-subset-selection problem (energy-based model; the
  fixed point is of their inference optimization, used to get cheap gradients).
  → **Our relationship (per ZJ review):** **a genuinely different perspective.** Their
  "set function" maps a set to a utility for *choosing a subset*; the fixed point / implicit
  diff is a *training tool* for that objective. We are not selecting subsets and our fixed
  point IS the representation-with-properties (path independence, unlearning), not a
  gradient trick. Cite as a related use of implicit differentiation on sets, clearly
  distinguished in goal and object of study.

## E. Machine unlearning (the property's home subfield)

- ⬜ **[prior]** Cao & Yang — *Towards Making Systems Forget*, IEEE S&P 2015. Origin of
  machine unlearning. Verify details.
- ⬜ **[prior/baseline]** Bourtoule et al. — *Machine Unlearning (SISA)*, IEEE S&P 2021.
  Retrain-from-shards baseline; the standard "exact unlearning is expensive" reference our
  cheap-warm-start story contrasts with. Verify.
- ⬜ **[theory/contrast]** Guo et al. — *Certified Data Removal*, ICML 2020. Approximate
  unlearning with guarantees on convex models. Verify.
- ⬜ **[prior]** Sekhari et al. — *Remember What You Want to Forget*, NeurIPS 2021. Verify.
- ⬜ NOTE: search for any "unlearning + equilibrium / implicit models" work to ensure our
  angle is unclaimed.

## F. Privacy evaluation (how we measure leakage)

- ⬜ **[baseline/method]** Shokri et al. — *Membership Inference Attacks Against ML
  Models*, IEEE S&P 2017. The MIA we will use as the unlearning-quality metric. Verify.
- ⬜ **[method]** Dwork & Roth — *The Algorithmic Foundations of Differential Privacy*,
  2014. DP background for the noise-masking fallback. Verify.
- ⬜ **[method]** Abadi et al. — *Deep Learning with Differential Privacy (DP-SGD)*,
  CCS 2016. Verify if we go the DP route.

## G. Cellular automata / learned local dynamics (the CA thread)

- ✅ **[prior/contrast]** Grattarola, Livi, Alippi — *Learning Graph Cellular Automata
  (GNCA)*, NeurIPS 2021. arXiv:2110.14237. GNN as a learned CA transition rule, iterated
  over graphs; experiments on Voronoi rules, Boids flocking, and point-cloud morphogenesis.
  Stability of target states achieved via BPTT + cache/replay (Growing-NCA trick).
  → **Our relationship:** the closest realization of the CA→learned-dynamics→attractor idea
  behind our project, and a sharp contrast. They solve dynamics by **BPTT** and enforce
  attractors by **training-time replay**; we solve an **implicit fixed point** and get
  well-posedness from **normalization/contractivity**. They operate on **graphs**; we on
  **sets** (no given edges). They explicitly report **oscillating orbits instead of
  convergence** — independent evidence that iterated local updates need structure to be
  well-posed (corroborates our normalization finding). They target **geometric** shapes;
  we target **semantic** representations + downstream tasks.
- ⬜ **[inspiration/contrast]** Mordvintsev et al. — *Growing Neural Cellular Automata*,
  Distill 2020. The morphogenesis + **regeneration-after-damage** result. KEY CONTRAST:
  NCA *regenerates* removed parts (remembers & restores via a trained-in attractor) — the
  OPPOSITE of our exact unlearning, which wants removed points to leave no trace. Same
  surface op ("remove, re-run"), opposite intent; a clean way to define erasability by
  negative example.

## H. Theory anchors / inspiration

- 🟡 **[theory]** Ramsauer et al. — *Hopfield Networks is All You Need*, ICLR 2021.
  arXiv:2008.02217. Modern Hopfield update = attention; energy descent to attractors. The
  anchor for "normalized attention DEQ = learned associative memory over the set." Verify
  the exact update correspondence before relying on it.
- 🟡 **[inspiration]** Ainsworth, Hayase, Srinivasa — *Git Re-Basin: Merging Models modulo
  Permutation Symmetries*, 2022. arXiv:2209.04836. Permutation symmetry in weight space
  (advisor-suggested). Tangential — conceptual resonance on permutation invariance, not
  a direct method dependency.

---

### Open positioning questions
1. Primary venue framing: privacy/unlearning (SaTML, unlearning workshops) vs. core ML
   (the implicit-models line)? Affects which of E/F vs A/D we foreground.
2. Confirm no existing "exact unlearning via equilibrium representations" paper — the core
   novelty claim depends on this gap being real.
