# Note to Geng — gut-check on a maintainability framing (and a null result)

Hi Zhengyang — a focused ask. I've spent the last stretch nailing down *what an
equilibrium graph model actually buys*, and I've landed on a thesis I'm fairly
confident in plus one novelty question I'd value your read on before I invest more.

## The setup in one paragraph

Take a graph equilibrium model $z^\star = f(z^\star)$ (one round of frozen-weight,
possibly signed/nonlinear message passing). After a graph edit (node/edge delete),
maintain $z^\star$ by warm-starting from the old fixed point and re-solving. In #7 I
showed the maintainability condition is **not** contraction ($\rho(J)<1$) but the
**conditioning** of $(I-J)$: edit effects decay with graph distance at a screening
length $\xi \approx \sqrt{\kappa(I-J)}/2$, governed by $\sigma_{\min}(I-J)$ (distance
of the spectrum from $+1$), via Demko–Moss–Smith / Benzi–Golub resolvent decay.
Empirically a model with $\rho(J)=0.955$ — barely contractive — still has $\xi\approx 1$
hop, and Broyden reaches genuinely non-contractive ($\rho>1$) but well-conditioned
fixed points that Picard can't.

## The framing I've converged on

The contribution is a **maintainability framework**, not an expressivity win:

1. **Characterization** — $\sigma_{\min}(I-J)$ governs edit-locality of an implicit
   model (non-obvious: a fixed point *looks* global but is provably $\xi$-local).
2. **Algorithm** — warm-start local re-solve maintains nonlinear/signed/interleaved
   equilibria, including the $\rho>1$ regime where no delta-push exists.
3. **Unification** — the same condition + algorithm cover the whole family;
   InstantGNN-style linear PPR is the special case where the resolvent is linear
   (and you get the bonus of exact delta-push).

So this reads as **"InstantGNN's local incremental maintenance, generalized from
linear propagation to nonlinear/interleaved equilibria, with a $\sigma_{\min}$
characterization of when it's valid."**

## The null result I want to be upfront about

I tried hard to also claim *"more expressive than the maintainable incumbents."*
It doesn't hold robustly:

- On controlled local tasks (squared energies of local operators — Laplacian /
  biharmonic / neighbour-variance on a grid), the nonlinear equilibrium **ties** a
  linear-PPR equilibrium with a nonlinear head (±0.05, direction flips with task).
- Going to equilibrium gives **no** expressivity over the same cell unrolled
  $K\approx\text{reach}$ steps — expected, since reach $=$ edit-length $=\xi$.
- The tasks that *would* separate them (interior pre-aggregation squares) are
  exactly the ones our own cell can't learn well either.
- Real heterophily (roman-empire) doesn't rescue it: we beat SGC by +27pt but the
  *graph-free MLP* is only 3.8pt behind us, so propagation's role is suspicious.

I'd rather state this as a tie than oversell it.

## The actual question for you

**Is the bridge genuinely open, or folklore?** Specifically: is
"$\sigma_{\min}(I-J)$ as the edit-locality / well-posedness criterion for
*non-contractive* implicit graph models, + warm-start local re-solve as the
nonlinear generalization of InstantGNN" a real gap, or has the implicit-models /
numerical-LA community effectively said this already (e.g., is it implicit in the
monotone-operator / Anderson-acceleration / IGNN well-posedness lines, or in the
incremental-PPR literature)? My lit pass (GNNDelete/GIF/CEU unlearning; EGNN;
InstantGNN/dynamic-graph; IGNN/IDGNN; DMS/Benzi-Golub/Sherman-Morrison-Woodbury)
didn't find the intersection occupied, but you'd know the implicit-models corner
far better than I do.

Secondary: **given the expressivity tie, is the framework + characterization enough
to stand on its own** as a contribution, or does it need a sharper practical hook
(e.g., the ring-truncation cost win on a large high-diameter graph, or a setting
where the linear-push incumbents genuinely fail and ours doesn't)?

Happy to send the two short reports (#7 theory, #8 framework + null) if useful.
Thanks — ZJ
