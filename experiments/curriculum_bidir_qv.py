"""QUERY-VISIBLE bidirectional curriculum — the substrate where query-awareness is architecturally POSSIBLE.

The standard bidir substrate keeps queries READONLY (context cannot attend to them), so its context
equilibrium is independent of what the queries ask -> the relay CANNOT be query-aware, and C2-bidir
measured must-carry ~ causal (far/near 0.061 vs 0.068) — exactly what the reader-set principle predicts
for invisible readers. To measure LAZY evaluation (C2t) and test whether query-awareness actually forms
when permitted, we need READONLY_Q *off*: queries attendable within the band.

RISK/HISTORY: round-4 tested readonly-off only in the pre-curriculum era (everything was stuck at the
one-layer ceiling for unrelated reasons); readonly-off + window curriculum is an UNTESTED combination.
Query-identity leakage (query tokens duplicate key tokens) may still poison training — if this fails to
train, that is itself a finding: reader visibility may be in tension with trainability, i.e. must-carry
could be unavoidable in practice for this model class.

Saves checkpoints/bidirqvXX.pt.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_bidir_qv
"""
import experiments.sliding_window_reach as sw
import experiments.curriculum_bidir as cb

sw.READONLY_Q = False        # the one change: queries attendable (cb.main sets BIDIR/REL_BIAS itself)
cb.PREFIX = "bidirqv"

if __name__ == "__main__":
    cb.main()
