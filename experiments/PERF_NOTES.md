# GPU / performance notes for the DEQ training + probe scripts

Measured 2026-07-14 on an **RTX 4050 Laptop GPU** (6 GB) while `curriculum_anchor.py` was training.
Read this before "optimizing" a training run or wondering why the GPU looks idle.

## The workload does NOT fill the GPU (this is expected, not a bug)

During a training stage: GPU util ~33% (choppy 22–45%), **power ~5.5 W** (near idle-warm for this GPU),
**memory 209 / 6141 MiB (3.4%)**. The "33% util" is misleading — near-idle power with a resident kernel a
third of the time means true SM occupancy is low single digits.

Why: the model is tiny (`d=64`, `L≈70`, `bs=64`) and the forward is an **Anderson fixed-point solve**
(`f_max_iter=60`, `f_tol=1e-4`) — up to 60 *sequential* iterations of small matmuls, plus another solve in the
implicit-diff backward. Each kernel is microseconds of math separated by Python/launch overhead, and the
iterations are inherently sequential (iter k+1 needs k). Also `gen_mqar` builds every batch on **CPU** (CPU
generator) then `.to(DEV)`. So wall-clock is bound by **launch latency + CPU data-gen + sequential solver**,
not GPU throughput. A faster/bigger GPU barely helps this shape of work.

## The "GPU drops to 0 for a while" = per-stage `spectrum()` eval

`eval_stage` calls `SeqDEQ.spectrum`, which builds a dense **N×N Jacobian** (`N = L·d`, ≈ 4544 at gap 60 with
the anchor) via `jacrev`, then:

| op | cost @ N≈4544 (this GPU) | device |
|---|---|---|
| `torch.linalg.svdvals(I−J)` → σ_min | **~28 s** | cuda (but slow — dense gesvd) |
| `torch.linalg.eigvals(J)` → ρ | **~17.5 s** | cuda (**not** a CPU fallback — verified) |
| power-iteration for ρ | **~0.10 s**, ~1.8% off exact | cuda |

That ~46 s of near-serial linear algebra, ×9 stages ×2 substrates ≈ **~14 min of stall per full run**, is the
"0 for a while." It is diagnostic logging, not training.

## Levers (and when each is safe)

1. **Concurrency — always comparability-safe, but the EVAL memory caps it on a small card.** The GPU is ~95%
   idle during *training*, so running independent jobs in parallel (e.g. `curriculum_anchorless.py currnp` and
   `... bidir` as two processes) speeds up the training phases. Does not touch batch/recipe/results — timing
   changes, checkpoints don't (deterministic given seed). **BUT** each per-stage `spectrum()` eval needs
   **~5.7 GB** (dense ~4.5k² Jacobian + SVD workspace), which nearly fills a 6 GB card — so concurrent evals
   **cannot overlap**; they serialize (and risk OOM), and high-gap stages balloon. MEASURED: 3 jobs concurrent
   (2 controls + anchor) took **169–224 min each** instead of the naive ~45 min — the training sped up but the
   serialized big-Jacobian evals dominated (a single gap-60 stage hit ~4500–5300 s). So concurrency helps, but
   on this GPU the eval footprint is the real ceiling; either run fewer jobs at once, or cut the eval cost
   (lever 3) before parallelizing. Preferred for comparison-bound runs, in moderation.
2. **Bigger batch — greenfield only.** ~30× VRAM and ~10×+ compute headroom; the batch dim is embarrassingly
   parallel through the solver, so `bs 64→512` raises utilization *and* efficiency. BUT it changes optimization
   dynamics, so it **confounds any comparison against the existing `bs=64` ladders** (all substrate tables,
   `curr*`/`bidir*`/anchor). Use only for genuinely new experiments with no prior ladder to match; retune
   `lr`/`steps` for the larger batch.
3. **`eigvals → power_method` for ρ in `spectrum()` — greenfield / non-load-bearing.** ρ is a rough reach
   diagnostic; power iteration gives it in ~0.10 s vs ~17.5 s at ~2% error (torchdeq ships `power_method`, or a
   3-line power iteration on `J`). Keep **`svdvals` for σ_min** — σ_min is the load-bearing conditioning
   invariant and goes near-singular (~1e-4) at long gap, where the normal-equations shortcut (σ_min² via
   `eigvalsh(JᵀJ)`) loses precision in float32. Don't swap the ρ method *mid-comparison*: it introduces a ~2%
   ρ drift between checkpoints computed the old vs new way.

## Rule of thumb

Within one comparison family, keep `bs`, recipe, and the `spectrum()` method **identical**; get speed from
**concurrency**. Save batch/eval-method changes for fresh experiments, and record them here when you do.
