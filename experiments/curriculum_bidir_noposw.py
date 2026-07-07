"""PURE-RELATIVE-PE bidirectional curriculum — the application-narrative viability check.

posw_ablation showed the learned ABSOLUTE positional embedding is load-bearing for the cross-window
relay on the standard bidir checkpoints (recall collapses at gap>0 without it; ||posw|| grows with gap).
Real editing workloads are insert/delete-heavy, and the aligned-frame insert story requires shift
invariance = pure relative position. So the question this run answers is existential for the
application narrative: **can a banded DEQ relay across windows with ONLY relative position (relb),
or does the relay need an absolute coordinate?**

Same two-phase curriculum as curriculum_bidir (window curriculum forms the binding hop, then gap
curriculum with re-banded queries), with sw.NO_POSW=True. Saves checkpoints/bidirnpXX.pt.

If this trains: the pure-relative substrate exists -> insert/delete (v2) unblocked, and the standard
substrate's posw-reliance was a convenience of training, not a necessity.
If it fails at gap>0: genuine finding — the relay leans on an absolute coordinate; the insert story
needs a different position mechanism (e.g. RoPE-style rotation) or must be scoped down.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_bidir_noposw
"""
import experiments.sliding_window_reach as sw
import experiments.curriculum_bidir as cb

sw.NO_POSW = True
cb.PREFIX = "bidirnp"

if __name__ == "__main__":
    cb.main()
