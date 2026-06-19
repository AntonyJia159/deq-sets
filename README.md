# Set-DEQ: equilibrium representations of sets

Probing **path independence** and **exact unlearning** in Deep Equilibrium
Networks applied to set-structured data (sets of feature vectors in an abstract
semantic space, not 3D geometry).

## Core idea

A set-DEQ iterates a permutation-equivariant update on a latent `Z (N, d)` until
it reaches a fixed point `Z*`. The readout pools over the set, so predictions are
permutation-invariant. The central hypothesis:

> If the fixed point is **unique** (path independent), then `Z*` is a pure
> function of the current set -- independent of insertion/deletion history. This
> yields **exact unlearning** at inference: remove a point, re-solve, and the
> result is identical to never having seen it. Non-uniqueness is precisely the
> failure mode that lets an attacker recover removed data.

## The three-layer plan

1. **Layer 1 (this scaffold):** does a vanilla set-DEQ give these properties for
   free? Synthetic Gaussian-mixture task + three probe metrics
   (path-independence gap, unlearning gap, warm-vs-cold efficiency).
2. **Layer 2:** adversarial evaluation -- membership-inference attack as the
   leakage metric; differential-privacy noise masking of the residual.
3. **Layer 3:** characterization -- Jacobian spectrum across data configurations
   to locate bifurcations (where uniqueness, and thus unlearning, breaks).

## Layout

```
src/
  data.py     Gaussian-mixture set dataset (label = number of clusters)
  solver.py   damped fixed-point solver (torchdeq swapped in later)
  model.py    SetDEQ with DeepSets / attention updates; optional spectral norm
  metrics.py  path_independence_gap, unlearning_gap (+ efficiency)
  train.py    minimal CPU training loop
experiments/
  smoke_test.py   train + run all three probes
```

## Setup & run

Requires a real Python (3.10+) with pip -- the Windows Store stub will not work.

```
pip install -r requirements.txt
python -m experiments.smoke_test
```

Knobs to sweep once the smoke test passes: `update` (`deepsets` vs `attn`),
`spectral` (contractivity on/off), and the dataset's `n_points` (train small /
test large) to see where the free lunch ends.
