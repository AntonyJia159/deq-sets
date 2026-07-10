# Memo (SPECULATIVE) — Coupled equilibria as a substrate for multimodal / agentic / neurosymbolic systems

*2026-07-10. ZJ's end-of-day idea, parked for later. Speculative / systems-flavored — NOT for the current paper's
spine; the current paper is the potential theoretical enabler. Revisit when the core certificate paper is out.*

## The vision (ZJ)

Entities in an environment — images, text, other inputs — each carry **state**, possibly produced by different
algorithms/models. When an agent comes into **contact** with an entity, a cross-attention (or similar) mechanism
forces a **consistency iteration** between the two: they **co-reach a joint equilibrium**, and predictions/actions
are made downstream of that settled state. The system is a **"lingua franca of equilibria"** built in **patches**
(not one monolith); everything steps **asynchronously**, event-driven by the interaction sequence, and **never
globally converges** — it tracks a nonstationary world. This is an affordance the **feedforward** regime does not
imply.

## The sharp claim (why feedforward can't do it)

- Feedforward composition is **procedural** (A→B→C, fixed pipeline); equilibrium composition is **relational** (A
  and B settle into a mutually-consistent joint state, order-independent under well-posedness).
- Contact is **two-way**: each side pushes back until consistent. That "pushing back" exists **only because there
  is a fixed point** — a feedforward net computes input→output and the input never hears back. Same root as "the
  edit-response is an inverse only because there's an equilibrium," generalized from one model to a network.
- So: **equilibrium = negotiation / mutual consistency; feedforward = one-way computation.**

## Lineage (grounded, not vapor — novelty is the combination)

- Co-reach on contact = **predictive coding** (Rao–Ballard), **energy-based / Hopfield consensus**, **equilibrium
  propagation**.
- Async / event-driven / no global convergence = **asynchronous fixed-point iteration / chaotic relaxation**
  (Chazan–Miranker 1969; Bertsekas–Tsitsiklis, *Parallel and Distributed Computation*). Real theorem: async
  iteration in ANY order → the SAME fixed point, **provided** the well-posedness / diagonal-dominance condition
  holds. Nonstationary env → tracks a moving equilibrium instead of settling.
- Actions from the settled state = **active inference** (act to make the world match settled predictions).
- Novel part = **heterogeneous patch-built modules interoperating through equilibria-as-interface**, event-driven,
  with a **locality guarantee**.

## Where OUR certificate is the keystone

Tractability rests on: a local contact must induce a **local** update that does **not** require global
reconvergence (else every interaction re-solves the whole coupled system → dead on arrival). That IS our
edit-locality certificate at system scale:
- **σ_min of the joint operator** = can two entities consistently co-settle (well-posed consensus) or not.
- **reach envelope** = the **blast radius of a contact** (how far it must propagate).
- **reader-set** = *which* modules a contact must update.

⇒ The current paper is the **theoretical license** for "patches + async + no global convergence": it certifies
contact-induced updates stay bounded. Reframes the certificate from a characterization curiosity into the
**enabling primitive** ("not just for KV-caches — it's what lets equilibria compose asynchronously at all").

## Through-line to what we measured

Async event-order-independence = the **path-independence** thread, holding iff the joint fixed point is unique
(σ_min>0). So **multistability is the system-scale failure mode**: two independently-trained modules with no
unique joint state → order-dependent, conflicting, forked beliefs = **curr40 writ large**. Our machinery already
carries both the **enabler** (locality) and the **failure diagnosis** (multistable coupling).

## Honest hard parts

1. **Joint well-posedness isn't free** — two heterogeneous modules may share no consistent fixed point (conflict)
   or many (ambiguity/multistability). The "lingua franca" must be learned/enforced; this is the crux.
2. **Credit assignment** across async, coupled, nonstationary equilibria (implicit-diff over the whole web,
   asynchronously) is unsolved; RL on top compounds it.
3. **Testability** — systems vision, hard to falsify cleanly.

## Minimal testable version (a natural C-series extension)

Two **heterogeneous** DEQ modules (different arch/training), coupled by cross-attention, co-reaching a joint fixed
point. Measure with existing tools: (i) does a unique joint equilibrium exist (σ_min of the coupled operator)?
(ii) edit module A's input → how far into B does it propagate (cross-interface edit-locality / reach envelope)?
(iii) does async (event-order) relaxation reach the same joint state as synchronous? Turns "lingua franca of
equilibria" into a measured claim.

## Concrete home: graphics / light transport / drag-and-drop asset integration (2026-07-10)

The cleanest real instance — physical consistency instead of a learned lingua franca:
- **The equilibrium is PHYSICAL, not imposed.** Light transport / radiosity = `(I−T)⁻¹` (a resolvent, ours);
  physics sim = energy equilibrium; SDF = eikonal PDE. DEQs are native here (INRs, implicit/equilibrium rendering),
  not retrofitted.
- **`G` = the light-transport operator; `ρ(G)` = albedo / bounce-decay.** Our reach envelope `ρ(G)^d` is literally
  **global-illumination falloff** — edit one surface, the light change decays with bounce distance. Abstract
  edit-locality → a physical, observable quantity.
- **Multi-representation ecosystem = the vision, concretely.** NeRF / mesh / point cloud / voxel / SDF / splat /
  INR are different *languages for the same scene field*, coupled by **consistency**. Editing one → the others must
  re-equilibrate. Physics is the consistency constraint (cleaner test bed than the abstract multimodal case).
- **Persona / product (ZJ): "jpg/png for 3D" — drag-and-drop scene assembly.** A game/interior designer downloads
  heterogeneous assets (voxels, point clouds, meshes, NeRFs). Today DL tools **convert** between formats (one-way,
  lossy, context-blind). The DEQ move: **don't convert to a canonical format — couple by consistency in a shared
  field** (the equilibrium *is* the lingua franca, sidestepping "need one format"). Drag-and-drop = a new element
  **contacts** the scene → a consistency iteration integrates it: it **conforms** (relights, respects geometry) AND
  **changes** the scene (casts shadows, occludes, bounces light). That two-way adjustment is a fixed point —
  **feedforward converters can only do one direction.**
- **Our certificate = the interactivity engine (a NON-safety value prop).** Drop an object → don't re-solve global
  illumination; certify the **local blast radius** (nearby relighting/shadows), Woodbury-predict the transport
  update (moving one object = low-rank `δT`), cache the far field. Here the certificate buys **instant-and-correct
  interactivity**, not safety — a distinct customer (games / virtual production / digital twins / asset marketplaces,
  where heterogeneous formats are a real pain).
- **Honest hard parts:** heterogeneous reps may share no consistent joint equilibrium (a low-poly mesh vs a
  high-detail NeRF disagree at fine scale → ill-posed or multistable = σ_min-of-joint again); you need per-pair
  consistency operators or a shared field they all embed into; real-time GI is already hard.

## Status

Parked. Prereq = the core certificate paper. Related: [federated direction memo], the async-convergence owed check
(flagged for per-window early-stop), path-independence (2D time×depth recurrence discussion).
