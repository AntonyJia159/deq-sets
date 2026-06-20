"""MIA Phase 2 -- train attackers on the cached datasets, report AUC.

Loads each (config, seed) cache and trains, on a 70/30 split of CONVERGED samples:
  latent attacker : permutation-invariant net over Z_post (23x32) conditioned on the
                    candidate c -- the strong, latent-access threat model.
  output attacker : MLP on [mean-pool(Z_post), emb(c)] -- the weak, readout-level
                    threat model (tests whether the channel closes at the readout).

AUC > 0.5 = the post-deletion equilibrium betrays whether c was deleted = leak.
Expected: normdeepsets ~0.5 (control); attn_baseline > 0.5; attn_pi reduced.

Run:  & "D:\\deq-venv\\Scripts\\python.exe" -m experiments.mia_attack
"""

import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "mia_cache")
ATTACKER_SEEDS = [0, 1, 2]
EPOCHS = 40


def auc_score(score, y):
    """Mann-Whitney AUC = P(score_pos > score_neg), tie-corrected via mean ranks."""
    score, y = np.asarray(score), np.asarray(y)
    order = score.argsort()
    ranks = np.empty(len(score), dtype=float)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # average rank, 1-based
        i = j + 1
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r_pos = ranks[y == 1].sum()
    return (r_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def _mlp(sizes, out_act=None):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers = layers[:-1]
    return nn.Sequential(*layers)


class LatentAttacker(nn.Module):
    def __init__(self, d_lat=32, d_in=2, h=64):
        super().__init__()
        self.c_emb = _mlp([d_in, h, d_lat])
        self.row = _mlp([3 * d_lat, h, d_lat])
        self.head = _mlp([d_lat, h, 1])

    def forward(self, Z, c):                      # Z (n,23,32), c (n,2)
        ce = self.c_emb(c).unsqueeze(1).expand(-1, Z.size(1), -1)
        feat = torch.cat([Z, ce, Z * ce], dim=-1)
        g = self.row(feat).mean(dim=1)            # perm-invariant pool
        return self.head(g).squeeze(-1)


class OutputAttacker(nn.Module):
    def __init__(self, d_lat=32, d_in=2, h=64):
        super().__init__()
        self.c_emb = _mlp([d_in, h, d_lat])
        self.head = _mlp([2 * d_lat, h, h, 1])

    def forward(self, Z, c):
        return self.head(torch.cat([Z.mean(dim=1), self.c_emb(c)], dim=-1)).squeeze(-1)


def train_attacker(make, Z, c, y, tid, seed):
    torch.manual_seed(seed)
    # Split by TRIAL, not by sample: a trial's member and non-member samples share
    # the same survivors B (=> ~same Z*(B)) and the same candidate c, differing only
    # by the warm-start residue. A per-sample split would let the attacker memorize a
    # trial's input from one twin and get fooled by the other (opposite label) ->
    # systematic AUC < 0.5. Keeping both twins on the same side leaves only the
    # residue (the real history channel) as signal.
    uniq = torch.unique(tid)
    gperm = torch.randperm(len(uniq), generator=torch.Generator().manual_seed(seed))
    train_ids = uniq[gperm][: int(0.7 * len(uniq))]
    mask_tr = torch.isin(tid, train_ids)
    tr = mask_tr.nonzero(as_tuple=True)[0]
    te = (~mask_tr).nonzero(as_tuple=True)[0]
    n_tr = len(tr)
    Z, c, y = Z.to(DEV), c.to(DEV), y.to(DEV)
    net = make().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        net.train()
        for s in range(0, n_tr, 256):
            idx = tr[s:s + 256]
            opt.zero_grad()
            loss = lossf(net(Z[idx], c[idx]), y[idx])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        score = net(Z[te], c[te]).cpu().numpy()
    return auc_score(score, y[te].cpu().numpy())


def load(path):
    z = np.load(path)
    conv = z["conv"].astype(bool)
    n = len(conv) // 2  # cache layout: [members 0..n-1, non-members 0..n-1]
    tid_full = np.concatenate([np.arange(n), np.arange(n)])
    Z = torch.from_numpy(z["Zpost"][conv].astype(np.float32))
    c = torch.from_numpy(z["cand"][conv])
    y = torch.from_numpy(z["label"][conv].astype(np.float32))
    tid = torch.from_numpy(tid_full[conv])
    return Z, c, y, tid


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    by_cfg = {}
    for f in files:
        m = re.match(r"(.+)_seed(\d+)\.npz", os.path.basename(f))
        by_cfg.setdefault(m.group(1), []).append(f)

    results = {}
    for cfg, fs in by_cfg.items():
        for atk_name, make in (("latent", LatentAttacker), ("output", OutputAttacker)):
            aucs = []
            for f in sorted(fs):
                Z, c, y, tid = load(f)
                seed_aucs = [train_attacker(make, Z, c, y, tid, s) for s in ATTACKER_SEEDS]
                aucs.append(float(np.mean(seed_aucs)))
            arr = np.array(aucs)
            results[f"{cfg}/{atk_name}"] = {"mean": float(arr.mean()),
                                            "std": float(arr.std()),
                                            "per_seed": aucs}
            print(f"{cfg:<14} {atk_name:<7} AUC = {arr.mean():.3f} +/- {arr.std():.3f}  "
                  f"(seeds {[round(a,3) for a in aucs]})")

    out = os.path.join(os.path.dirname(__file__), "results_mia.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
