"""Pointer-chase-to-root, BIDIRECTIONAL substrate -- the decisive directionality test + the C6 substrate.

Same recipe as pointer_chase_train (relative PE, window+depth curriculum) but sw.BIDIR=True (two-sided band).
Question: does a bidirectional relay lift the content-random multi-hop chase above the causal ~0.70 ceiling?
  - bidir >> causal  -> pointer-chase-to-root IS bidir-natured (causal blocks backward-pointing edges), and
                        this is the C6 substrate.
  - bidir ~= causal  -> the ceiling is model capacity / task difficulty, not directionality.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.pointer_chase_train_bidir
"""
import experiments.sliding_window_reach as sw
import experiments.pointer_chase_train as pct

sw.BIDIR = True                 # two-sided band -- the one change vs the causal trainer
pct.CKPT_NAME = "pcchase_bidir.pt"

if __name__ == "__main__":
    pct.main()
