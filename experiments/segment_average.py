"""SEGMENT AVERAGE -- a real-valued, two-sided, select-and-remix consistency task on a stream.

Factored embedding (no value injection): each token is [ mode | value ]. The MODE part (a lookup) marks a
token as a VALUE-carrier or a BOUNDARY; the VALUE part is a real vector payload (zero for boundaries). At each
value position the target is the EXPONENTIAL-DISTANCE-WEIGHTED AVERAGE of the value-parts of the tokens in its
SEGMENT -- the region between the nearest boundary on the left and on the right. Boundaries are content-marked
(mode = a boundary role, recoverable by dot-product with a cardinal direction); the exp-decay exp(-|i-j|/tau) is
the locality kernel.

  select  = find the two bounding boundaries (content) -> a soft two-sided cage
  remix   = exp-weighted average of the in-segment values (real-valued, dense mixing)

WHY: well-conditioned (averaging is contractive -> the clean BVP/near-normal face, opposite of recall-peaking),
piecewise-local (each segment independent -> should length-generalize, maze-like), and the edit-response is a
compact two-sided tent bounded by the segment = a direct check of the resolvent (I-J)^{-1}. Editing a VALUE
shifts its segment's field (exp-tent); editing a BOUNDARY re-scopes the segment (structural).

Self-test (__main__): target == in-segment exp-weighted average; editing one value moves ONLY same-segment
targets (bounded response = the tent), other segments exactly unchanged.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.segment_average
"""
import torch

import experiments.sliding_window_reach as sw

VALUE_MODE = 1                     # mode-token id for a value carrier
BOUNDARY_MODE = 2                  # mode-token id for a boundary
DV = 32                            # value-subspace width (set sw.D_VALUE = DV in the trainer)
TAU = 3.0                          # exp-decay length of the locality kernel


# segment widths deliberately span TIGHT (1-2, isolates a value -> sharp boundary cutoff) to WIDE (8-12,
# long-range averaging that must relay across the window) so training + validation stress the SELECTIVE
# (boundary-detection) and reach properties, not just a typical spacing.
WIDTHS = torch.tensor([1, 1, 2, 2, 3, 4, 6, 9, 12])


def gen_segment_average(batch, L, gen, dv=DV, tau=TAU, widths=WIDTHS):
    """Returns toks (B,L mode ids), values (B,L,dv real), target (B,L,dv), tmask (B,L bool value positions),
    seg_id (B,L) segment index. Segment widths are sampled to SPAN tight..wide; boundaries carry zero value."""
    is_bnd = torch.zeros(batch, L, dtype=torch.bool)
    for b in range(batch):                                              # value-runs of varied width, boundary between
        p = torch.randint(0, 3, (1,), generator=gen).item()
        while p < L - 1:
            p += int(widths[torch.randint(len(widths), (1,), generator=gen)])
            if p < L - 1:
                is_bnd[b, p] = True
                p += 1
    toks = torch.where(is_bnd, BOUNDARY_MODE, VALUE_MODE)               # (B,L) mode ids
    val_mask = ~is_bnd
    values = torch.randn(batch, L, dv, generator=gen) * val_mask[..., None]
    seg_id = torch.cumsum(is_bnd.long(), dim=1)                         # value i,j same segment <=> seg_id equal
    same_seg = seg_id[:, :, None] == seg_id[:, None, :]                 # (B,L,L)
    idx = torch.arange(L)
    dist = (idx[:, None] - idx[None, :]).abs().float()                  # (L,L)
    w = torch.exp(-dist / tau)[None] * same_seg.float() * val_mask[:, None, :].float()   # contributions from values
    target = (w @ values) / (w.sum(-1, keepdim=True) + 1e-9)            # (B,L,dv) in-segment exp-avg
    target = target * val_mask[..., None]                              # boundaries -> 0 target (masked anyway)
    return (toks.to(sw.DEV), values.to(sw.DEV), target.to(sw.DEV),
            val_mask.to(sw.DEV), seg_id.to(sw.DEV))


def _true_target(values_row, is_bnd_row, tau=TAU):
    """Reference ground truth for one row (numpy-free, for the self-test)."""
    L, dv = values_row.shape
    seg = torch.cumsum(is_bnd_row.long(), 0)
    idx = torch.arange(L)
    dist = (idx[:, None] - idx[None, :]).abs().float()
    vm = (~is_bnd_row).float()
    w = torch.exp(-dist / tau) * (seg[:, None] == seg[None, :]).float() * vm[None, :]
    return (w @ values_row) / (w.sum(-1, keepdim=True) + 1e-9) * vm[:, None]


def main():
    g = torch.Generator().manual_seed(0)
    L = 24
    toks, values, target, tmask, seg_id = gen_segment_average(1, L, g)
    is_bnd = (toks[0] == BOUNDARY_MODE).cpu()
    bpos = is_bnd.nonzero().flatten().tolist()
    widths = [bpos[0]] + [bpos[i] - bpos[i - 1] - 1 for i in range(1, len(bpos))]
    print(f"L={L}  boundaries at {bpos}  segment widths (tight..wide) {widths}  seg_id={seg_id[0].cpu().tolist()}",
          flush=True)
    ref = _true_target(values[0].cpu(), is_bnd)
    err = (ref.to(sw.DEV) - target[0]).abs().max().item()
    print(f"target == in-segment exp-weighted average: max|diff|={err:.2e} (should be ~0)", flush=True)

    # edit one value -> only its segment's targets move (bounded tent); other segments EXACTLY unchanged
    vpos = int((toks[0] == VALUE_MODE).nonzero().flatten()[L // 2])
    seg_of_vpos = int(seg_id[0, vpos])
    v2 = values.clone(); v2[0, vpos] += 5.0
    ref2 = _true_target(v2[0].cpu(), is_bnd).to(sw.DEV)
    moved = (ref2 - target[0]).norm(dim=-1) > 1e-6
    same_seg_moved = moved & (seg_id[0] == seg_of_vpos)
    other_seg_moved = moved & (seg_id[0] != seg_of_vpos)
    print(f"edit value at pos {vpos} (segment {seg_of_vpos}): same-segment targets moved "
          f"{int(same_seg_moved.sum())}, OTHER-segment moved {int(other_seg_moved.sum())} (must be 0)",
          flush=True)
    resp = (ref2 - target[0]).norm(dim=-1).cpu()
    print(f"response profile (norm per position): {[round(x,2) for x in resp.tolist()]}", flush=True)
    print("\nREAD: the edit response is a segment-bounded exp tent (zero outside the two boundaries) = the\n"
          "two-sided Green's function the resolvent certificate should reproduce on the well-conditioned face.",
          flush=True)


if __name__ == "__main__":
    main()
