# Note to Geng — DEQ-transformer edit-locality: five things worth your read (2026-07)

Hi Zhengyang — the direction has moved from graphs to **equilibrium transformers**, per your steer
toward a use-case people want to read (in-context editing / KV maintenance — scenario B). The spine is
measured now. Five bullets, ranked; if you only have time for three, it's 1–3.

**1. The use-case, with a theorem where CacheBlend has a heuristic — and an honest boundary.**
The equilibrium state *is* a KV cache (O(n·d), depth axis collapsed by weight-tying; it is not the
embeddings), and a mid-context edit's invalidation region is a **certified σ_min-ball** — the sound
version of CacheBlend's lossy partial reuse. But we can *prove* edit-locality is **dual to forgetting**
(the same σ_min is the screening length *and* the memory horizon): a causal LM doing its job (long
memory) has non-local edits by necessity, so causal-LM maintenance is self-defeating. The real win is
the **bidirectional local-readout niche** (code-edit-near-cursor, RAG chunk swap). This says exactly
where scenario B is real and where it evaporates.

**2. Your "conditioning isn't the story" steer was right — and sharper than expected: the *scalar*
certificate is vacuous, the *directional* one is the whole content.**
On the causal face every per-hop transfer norm exceeds 1 (up to 25×), so the scalar product bound
predicts *growth* — **764× slack** vs the true decay over 4 hops. Yet the carry subspace is **low-rank
(≈8 of d=64)**, so an a-priori per-edit directional certificate (project δh *before* solving) reproduces
the 3-tier edit taxonomy quantitatively with **zero false containments**. Conditioning-as-a-scalar isn't
the story; conditioning-as-a-*direction* is. (Also: the coarse window-blocked transfer product
reconstructs the exact resolvent to 1e-15 — the product form is not an approximation.)

**3. An apparently undocumented trainability phenomenon in bidirectional equilibrium cells.**
Bidirectional attention-only DEQs **cannot form the two-hop recall circuit** — stuck at the one-layer
ceiling (recall 0.38) across every knob, tied *and* untied, equilibrium *and* unrolled. Rescue = a
**window curriculum** (grow the attention band; at w=2 the binding hop is forced by connectivity) →
recall 1.0. The two faces differ not just in proof family but in **failure mode and trainability**. The
rank-collapse and shortcut-learning halves each exist in the literature; the coupling (bidirectional
mask *blocks* circuit formation) doesn't seem to. Fresh result (this week): making the readers *visible*
to enable selectivity **degrades long-relay trainability** (recall 0.94→0.63 at gap 40) — a
visibility↔trainability tension.

**4. One Jacobian product runs everything — forward is editing, backward is BPTT.**
The edit-response (I−J)⁻¹δh and the BPTT gradient are transposes; σ_min is transpose-invariant, so **one
number certifies edit-locality and gradient conditioning**, and **edit-locality ⟺ vanishing gradients**.
Ties the maintenance object to mature RNN-Lyapunov theory (Vogt et al.) — cited, not claimed.

**5. A clean negative I'd rather show than bury: selective forgetting is *permitted* but doesn't *emerge*.**
I predicted the two faces would split into eager (causal) vs lazy (bidirectional) evaluation — deferred
billing. Measured, it **fails**: editing an unqueried value costs the same iterations whether or not a
reader is present, on *all* substrates including one where I made queries attendable so selectivity was
architecturally possible (write-cost 15.3 vs 15.7). **Must-carry is empirically robust — nothing in the
recall loss rewards forgetting, so it doesn't form** ("emergent, not certified," confirmed negatively).
Worse, making readers visible *hurt* both maintenance cost and trainability (recall 0.94→0.63 at long
gaps). Path-independence held to 1e-7 throughout. I'm keeping this in as an honest boundary on the
selectivity story — it sharpens rather than undercuts the central claim (edit-locality is genuinely
hard to get, and the σ_min envelope bounds it regardless of whether the model is clever).

Scope I'm keeping bright: characterization + measurement, not a systems benchmark; toy-scale (dense
Jacobian is the validation oracle — everything has a matrix-free JVP/Krylov/sketching analog for scale,
at iteration counts set by κ(I−J) itself). Happy to send the blueprint + run digest. — ZJ
