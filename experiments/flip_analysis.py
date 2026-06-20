"""Do decision flips track TASK AMBIGUITY or raw MULTISTABILITY?

Thesis under test: attention's latent multistability is ubiquitous (fp_gap > 0 almost
everywhere) but BENIGN -- the readout quotients it out -- EXCEPT where the task itself
is borderline. So decision flips (pred_agreement < 1) should pin to task ambiguity
(low inter-cluster separation / low readout margin), NOT to raw multistability (fp_gap).

Predictions:
  corr(flip, fp_gap)        weak    -- multistability is everywhere; flips are special
  corr(flip, -margin)       strong  -- flips where the model is genuinely uncertain
  corr(flip, -separation)   strong  -- flips where clusters are genuinely borderline

Run:  & "D:\\deq-venv\\Scripts\\python.exe" -m experiments.flip_analysis
"""

import numpy as np
import torch

from src.data import GMMSetDataset
from src.model import SetDEQ
from src.train import train

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K_RANGE = (1, 4)
N = 24
SEEDS = [0, 1, 2, 3, 4]
N_PROBE = 300
N_INITS = 8
SOLVE = dict(max_iter=200)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def predictor_auc(score, flip):
    """AUC of `score` ranking flip=1 above flip=0 (Mann-Whitney, tie-corrected)."""
    score, flip = np.asarray(score, float), np.asarray(flip)
    m = np.isfinite(score)
    score, flip = score[m], flip[m]
    order = score.argsort()
    ranks = np.empty(len(score)); s = score[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos, nneg = int((flip == 1).sum()), int((flip == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[flip == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


@torch.no_grad()
def batched_probe(model, X, n_inits):
    """Batched path-independence probe over a whole stack of sets X (M,N,d).
    Returns per-set fp_gap, flip indicator, and readout margin. Solves all M sets
    at once per init (n_inits batched solves) instead of one set at a time."""
    M, Nn = X.shape[0], X.shape[1]
    Zs, preds = [], []
    for _ in range(n_inits):
        z0 = torch.randn(M, Nn, model.d_latent, device=X.device)
        Z, _ = model.solve(X, z0=z0, **SOLVE)
        Zs.append(Z)
        preds.append(model.readout(model.pool(Z)).argmax(-1))
    Zs = torch.stack(Zs)          # (n_inits, M, N, d)
    preds = torch.stack(preds)    # (n_inits, M)
    fpgap = torch.zeros(M, device=X.device)
    for i in range(n_inits):
        for j in range(i + 1, n_inits):
            d = ((Zs[i] - Zs[j]).flatten(1).norm(dim=1) /
                 (Zs[j].flatten(1).norm(dim=1) + 1e-8))
            fpgap = torch.maximum(fpgap, d)
    agree = torch.tensor([float(preds[:, m].bincount().max()) / n_inits
                          for m in range(M)])
    flip = (agree < 1.0).int().numpy()
    probs = model.readout(model.pool(Zs[0])).softmax(-1)
    top2 = probs.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
    return fpgap.cpu().numpy(), flip, margin


def min_center_sep(x, assign):
    ks = assign.unique()
    if len(ks) < 2:
        return float("nan")  # k=1: no inter-cluster separation to speak of
    centers = torch.stack([x[assign == k].float().mean(0) for k in ks])
    d = torch.cdist(centers, centers) + torch.eye(len(ks), device=x.device) * 1e9
    return float(d.min())


def main():
    train_ds = GMMSetDataset(n_samples=2000, k_range=K_RANGE, n_points=N, d=2,
                             sep=4.0, std=1.0, seed=1)
    flip, fpgap, margin, sep = [], [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SetDEQ(d_in=2, d_latent=32, hidden=64, update="attn",
                       n_classes=train_ds.n_classes, max_iter=150, tol=1e-5)
        train(model, train_ds, epochs=15, batch_size=256, lr=1e-3, seed=seed,
              log_every=0, device=DEV)
        test_ds = GMMSetDataset(n_samples=N_PROBE, k_range=K_RANGE, n_points=N, d=2,
                                sep=4.0, std=1.0, seed=100 + seed)
        X = torch.stack([test_ds.X[i] for i in range(N_PROBE)]).to(DEV)
        fg, fl, mg = batched_probe(model, X, N_INITS)
        fpgap.extend(fg.tolist()); flip.extend(fl.tolist()); margin.extend(mg.tolist())
        sep.extend(min_center_sep(test_ds.X[i].to(DEV), test_ds.assign[i].to(DEV))
                   for i in range(N_PROBE))
        print(f"seed {seed} done  flips so far={int(np.sum(flip))}/{len(flip)}")

    flip = np.array(flip)
    n, nf = len(flip), int(flip.sum())
    print(f"\n{n} sets, {nf} flips ({100*nf/n:.1f}%)")

    def grp(v):
        v = np.asarray(v, float)
        f = v[(flip == 1) & np.isfinite(v)]
        nfl = v[(flip == 0) & np.isfinite(v)]
        return f.mean(), nfl.mean()

    print(f"\n{'variable':<14}{'flip mean':>12}{'no-flip mean':>14}"
          f"{'corr(flip,.)':>14}{'flip-AUC':>11}")
    for name, v, sign in (("fp_gap", fpgap, +1),       # predict weak
                          ("neg_margin", margin, -1),  # low margin -> flip
                          ("neg_separation", sep, -1)):  # low sep -> flip
        fm, nm = grp(v)
        score = sign * np.asarray(v, float)  # higher score should mean flip
        c = pearson(flip, score)
        a = predictor_auc(score, flip)
        print(f"{name:<14}{fm:>12.3f}{nm:>14.3f}{c:>14.3f}{a:>11.3f}")
    print("\n(prediction: fp_gap weak; neg_margin & neg_separation strong "
          "=> flips track task ambiguity, not raw multistability)")


if __name__ == "__main__":
    main()
