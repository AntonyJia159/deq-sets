# Memo — Federated / Decentralized Set-DEQs (speculative)

**Status:** speculative, not on the current roadmap. Captured 2026-06-20 from a ZJ
intuition: "emergence from local operations; two datasets exchange some latents,
freeze them, interact with their own data, and a global picture emerges."

The claim of this memo: the intuition is correct, and the mechanism that makes it
work is the *same contraction/uniqueness property* that already gives us exact
unlearning. Federation is a third corollary of path-independence, not a new feature
to be bolted on.

---

## 1. Why a set-DEQ is federatable at all

The update is `z_i ← f(z_i, x_i, AGG_j(z_j))`. Points interact **only** through the
aggregate `AGG`. For mean-pool the aggregate is a low-dimensional, *additively
decomposable* sufficient statistic. Split the global set `S = S_A ∪ S_B` across two
parties:

```
mean_all = (n_A · mean_A + n_B · mean_B) / (n_A + n_B)
```

Each party only needs to publish its **local mean of latents and its count** — never
a single raw point. This is exactly the structure secure aggregation in federated
learning already exploits; here it falls straight out of the aggregator.

## 2. The protocol (block-coordinate fixed point)

The global equilibrium couples the two blocks only through the aggregate `a`. So
solve it as a fixed point *in the aggregate*, which is tiny (d-dimensional):

1. Broadcast a current guess of the global aggregate `a`.
2. Each party solves its **own local equilibrium** — its own points, with `a` held
   frozen — and returns its local contribution `(mean_A, n_A)`.
3. Combine into an updated `a` (weighted average).
4. Repeat until `a` converges.

This is Jacobi/Gauss-Seidel block iteration over a consensus variable. Raw data never
leaves a party; only the d-dim aggregate crosses the wire. It is exactly ZJ's
"exchange latents, freeze, re-solve locally" picture, made precise.

## 3. The crux claim — federated = centralized, *exactly*, under contraction

If the global map is contractive (unique fixed point), then **any convergent update
schedule lands on the same point.** Therefore the federated block-iteration converges
to the *identical* global equilibrium as centralized pooling. Consequences:

- **Path-independence ⇒ partition-independence ⇒ schedule-independence.** The global
  answer does not depend on how the data is split across parties or in what order
  they update. This is the same uniqueness that gives exact unlearning — one property,
  now three corollaries (init-, partition-, schedule-independence).
- Federation accuracy is therefore *another probe of well-posedness*: where it
  diverges from centralized, the map is non-unique (attention near a bifurcation),
  exactly where unlearning also leaks.

## 4. Async robustness for free (the part that surprised me)

Classical result (Bertsekas & Tsitsiklis, *Parallel and Distributed Computation*;
Chazan–Miranker): **a contraction iterated asynchronously — parties updating at
different rates, with stale aggregates — still converges to the unique fixed point.**

So contractivity buys straggler-tolerance and stale-communication-tolerance *for
free*. No synchronization barrier needed. This is a strong, citable foundation and a
direct selling point over generic FedAvg, which has no such guarantee.

## 5. The expressiveness ↔ uniqueness ↔ federatability triangle

The mean-pool/attention tradeoff gains a third axis:

| Aggregator | Unique fp | Federatable | Expressive |
|---|---|---|---|
| mean-pool (contractive) | yes | yes (additive sufficient stat) | limited |
| **linear attention** | (likely) | **yes** — sufficient stat `Σ_j φ(k_j) v_jᵀ` is additive | medium |
| softmax attention | no (genuine) | **no** without approximation — needs all keys/values | high |

Softmax attention's aggregate is query-dependent: point `i`'s update needs every
other point's `(k_j, v_j)`, so federation would require exchanging all keys/values
(more leakage, latents not raw data) — *or* approximating with **linear attention**,
whose sufficient statistic IS additive and therefore federatable like the mean. This
gives a clean, possibly-publishable statement: *federatability tracks whether the
aggregator has a low-dimensional additive sufficient statistic*, and it co-varies
with uniqueness.

## 6. Privacy plug-in (ties to the DP thread)

What leaks in the protocol is only the shared aggregate `mean_A` (a mean of latents)
plus a count. Its sensitivity to any one member is `Δ / n_A` — it **shrinks with
party size**. So:

- Larger parties leak less per member.
- The DP machinery from the unlearning discussion plugs in directly: apply the
  Gaussian mechanism to the *shared aggregate*, not to raw latents (this is exactly
  DP-FedAvg). Noise scale = sensitivity = the Jacobian quantity again. Same
  instrument, now a fourth job.

## 7. Honest caveats / why "not yet"

- **Communication rounds.** The block-iteration needs several outer rounds (one per
  fixed-point step) — could be communication-bound. The exchanged object is tiny
  (d-dim), but round *count* matters. Need to measure rounds-to-converge vs. accuracy.
- **Non-IID partitions are the real test.** If party A holds all of cluster 1 and B
  all of cluster 2, does the federated equilibrium still match centralized? For
  contractive maps: yes (uniqueness). For attention: maybe not — and that gap is the
  interesting experiment.
- **Inference-time federation only.** Everything above assumes frozen weights and
  federates the *forward equilibrium*. *Federated training* (learning `f` across
  parties) is a separate, harder layer — it needs federated gradients through the
  implicit function theorem. Flag as a distinct future rung, do not conflate.

## 8. Minimal experiment when we get here

1. Train one contractive set-DEQ centrally (already have this).
2. At inference, split a test set into 2–4 parties; run the block-coordinate protocol
   (§2); measure `‖Z*_federated − Z*_centralized‖` and prediction agreement.
   Prediction: ~0 for contractive, nonzero tail for attention.
3. Sweep IID vs. non-IID (cluster-aligned) partitions — the discriminating case.
4. Async/straggler stress: randomly stale aggregates; confirm contractive still
   converges (Bertsekas–Tsitsiklis prediction), attention destabilizes.
5. (Later) DP-noise the shared aggregate; trace accuracy vs. ε.

## 9. Novelty check (verify before claiming)

- Federated mean-aggregation / secure aggregation: **known** (not our contribution).
- Async iteration of contractions converges: **known** (Bertsekas–Tsitsiklis).
- *Federated equilibrium representation with partition-independence as a corollary of
  the same contraction that gives exact unlearning*, plus the federatability↔
  uniqueness↔expressiveness triangle: **not aware of prior work** — this is the fresh
  framing and the unification is the value. Search FL + DEQ/implicit + set learning
  before asserting.

---

*Bottom line:* the federated angle is real and it is cheap to state because it reuses
the contraction/uniqueness machinery we already have. It is **downstream** of the
Jacobian probe and the MIA — those certify the uniqueness that this whole story
rests on. Revisit after Layer 3.
