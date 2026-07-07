"""Cheap check before any insert/delete experiment: is the LEARNED ABSOLUTE positional embedding (posw)
actually load-bearing on the bidirectional checkpoints, or is the per-head relative-position bias (relb,
added for the binding fix) carrying position by itself?

WHY IT MATTERS: an insert/delete under RELATIVE PE + banded attention reduces (in the aligned frame) to a
width-w substitution at the cut — far regions shift uniformly, attention byte-identical. But our h0 still
adds posw[:L] (absolute), which BREAKS that shift-invariance: an insert would smear the response globally
and we'd be measuring the absolute-PE artifact, not the screened shadow. So before an insert experiment we
need to know whether posw can simply be dropped.

TEST: for each bidir checkpoint, measure recall (a) as trained, (b) with posw zeroed at eval. If (b) holds,
position content is vestigial -> rel-bias carries the load -> the substrate is effectively relative-PE and
an insert experiment is a weekend. If (b) collapses, we must retrain the bidir curriculum with posw off
before any insert measurement.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.posw_ablation
"""
import glob
import os

import torch
import experiments.sliding_window_reach as sw

sw.H, sw.dh = 4, sw.d // 4
CKPT_DIR = "checkpoints"


def main():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "bidir*.pt")))
    if not ckpts:
        print(f"No bidir checkpoints in {CKPT_DIR}/"); return
    print(f"device={sw.DEV}  posw ablation on bidir checkpoints "
          f"(recall as-trained vs posw zeroed)\n", flush=True)
    print(f"{'ckpt':<14}{'gap':>4}{'recall':>9}{'recall(no posw)':>18}{'||posw||':>12}{'||emb||':>12}",
          flush=True)
    for path in ckpts:
        ck = torch.load(path, map_location=sw.DEV, weights_only=False)
        sw.REL_BIAS = "relb" in ck["state_dict"]
        sw.READONLY_Q = ck.get("readonly_q", False)
        sw.QUERY_FULL = ck.get("query_full", False)
        sw.BIDIR = ck.get("bidir", False)
        sw.W = ck.get("W", 10)
        gap = ck["stage_gap"]

        m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
        m.load_state_dict(ck["state_dict"]); m.eval()
        ge = torch.Generator().manual_seed(123)
        r_full = sw.recall(m, gap, ge)

        posw_norm = m.posw.detach().norm().item()
        emb_norm = m.emb.weight.detach().norm().item()
        with torch.no_grad():
            saved = m.posw.detach().clone()
            m.posw.zero_()
            ge = torch.Generator().manual_seed(123)
            r_nopos = sw.recall(m, gap, ge)
            m.posw.copy_(saved)

        flag = "  <- holds" if r_nopos > 0.9 * r_full and r_full > 0.5 else \
               ("  <- COLLAPSES" if r_nopos < 0.6 * r_full else "")
        print(f"{os.path.basename(path):<14}{gap:>4}{r_full:>9.3f}{r_nopos:>18.3f}"
              f"{posw_norm:>12.2f}{emb_norm:>12.2f}{flag}", flush=True)
    print("\nREAD: if recall(no posw) tracks recall, posw is vestigial -> substrate is effectively "
          "relative-PE -> insert/delete experiment is cheap (aligned-frame width-w substitution).\n"
          "If it collapses, retrain bidir curriculum with posw disabled before any insert measurement.",
          flush=True)


if __name__ == "__main__":
    main()
