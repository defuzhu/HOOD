# Area-weighted relaxed contact potential (experimental)

The current neural rollout can produce negative signed distances. A strict log
barrier is undefined there. Replacing negative distances by a positive constant
inside log would hide penetration and kill its distance gradient. V2 therefore
implements an explicitly **relaxed** barrier-inspired penalty, not a strict
interior-point method and not a nonintersection guarantee.

## Formula and units

The model follows the book's `(s-1) log(s)` density (not the squared IPC variant):

https://phys-sim-book.github.io/boundary-treatments/distance-barrier-for-nonpenetration/dist_barrier_formulation.html

Let `s = d/dhat`, `tau = 0.1`, and `phi(s) = (s-1) log(s)` for `0 < s < 1`.
Use zero above 1 and the following quadratic below tau:

```
phi_relaxed(s) = phi(tau) + phi'(tau)*(s-tau) + 0.5*phi''(tau)*(s-tau)^2
phi'(tau) = log(tau) + 1 - 1/tau
phi''(tau) = 1/tau + 1/tau^2
P = sum_i A_i * dhat * kappa(t)/2 * phi_relaxed(d_i/dhat)
A_i = sum_{triangles incident to i} area_rest(triangle)/3
kappa(t) = kappa_start * weight(t)/weight_start
```

The join at tau is C2. The activation at s=1 is C1 (the second derivative jumps).
For d<=0, energy and gradients stay finite and descent pushes outward. It does
not become infinite at zero and has no strict barrier guarantee. The distance
geometry itself still has nearest-feature/medial-axis nonsmoothness.

`A_i` is rest-surface quadrature area in m^2, not a measurement of the instantaneous
contact patch. Rest areas avoid a predicted area change becoming a way to lower
the loss. `dhat` is in m and `kappa` is in Pa, giving energy in J. All cloth
vertices are included, as in the original loss; handles are prescribed, so their
energy need not provide useful learning gradients.

`dhat=0.002 m` retains the original clearance. The extension starts at 0.2 mm.
The initial kappa is 94.82279102794561 Pa. This is only an energy-scale matching
initialization: the reference T-shirt has total rest area 0.6730958656354848 m^2
and 4424 vertices, giving `A_ref=0.00015214644340765933 m^2`. Equate the energy of
one average vertex at `d=dhat/2` with the original cubic at initial weight 5000:

```
kappa_start = weight_start * dhat^2 / (2*A_ref*log(2))
```

This does not match derivatives or calibrate physical contact stiffness, and
does not establish an optimal hyperparameter. Keep A_ref fixed when changing
mesh density. Recomputing it for each remesh would undo resolution invariance.

## Four controlled variants

| Config | Geometry | Potential |
|---|---|---|
| baseline/postcvpr_seed57 | legacy plane | original vertex cubic |
| contact/geometry_v1_seed57 | signed mesh | original vertex cubic |
| contact/area_cubic_v2_seed57 | signed mesh | sum A_i/A_ref * cubic |
| contact/relaxed_barrier_v2_seed57 | signed mesh | area relaxed log density |

Example fresh training (after a clean commit, with the environment activated):

```bash
python train_baseline.py config=contact/relaxed_barrier_v2_seed57
```

Each config keeps the same architecture, material distribution, seed, split,
training budget and fixed validation. It writes an independent run directory,
CSV, TensorBoard, checkpoints and provenance. Friction, preprocessing, graph
contact edges, self-contact and CCD are not changed by these variants.

## Implementation acceptance versus quality evidence

Analytic tests verify positive energies, outward descent at positive/zero/
negative gaps, finite gradients, finite-difference agreement, continuity at the
extension, correct weight ramp, and invariant area-integrated energy when a
constant-gap triangle is subdivided. The original criterion regression stays
exact. These establish implementation properties, not training superiority.

Before a long run, execute a short real-data optimizer/rollout smoke test on a
working GPU: finite losses/gradients, checkpoint save/load, fixed validation,
and acceptable step time/memory. CPU geometry tests cannot replace this gate.

For training comparison:

1. Train all four variants from scratch at matched budgets; first paired seed
   57, then at least 3 (preferably 5) paired seeds. A baseline fine-tune is a
   separate experiment and must not be presented as the same training budget.
2. Use exactly the same signed-mesh evaluator for every checkpoint, including
   baseline. Re-evaluate old saved rollouts if needed. Historical `approx_*`
   metrics and pre-cc1a4f9 `mesh_*` evaluations are not valid common comparisons.
3. Compare penetration fraction and area-weighted fraction, depth distributions
   and worst cases, stretch/strain, contact jitter, floating clearance, failures,
   and timing. Existing `mesh_*` fields are vertex samples; triangle intersections,
   area-weighted metrics and jitter need additional common evaluation before
   making corresponding claims. Watch identical rollout videos side by side.
4. Current fixed validation is only three T-shirt motions with at most 120
   frames. Expand held-out shapes/garments and longer sequences before claiming
   generalization. Tune coefficients on development validation, then freeze them
   and report on untouched test sequences; never select test-best settings.
5. Predeclare meaningful noninferiority margins per metric, in physical units.
   For lower-is-better metrics, evaluate paired (new-old) differences with an
   upper confidence bound: below 0 supports improvement; below a predefined
   positive margin supports noninferiority. Account for seed/sequence clustering,
   not millions of correlated vertices/frames as independent samples. With few
   seeds/sequences, report uncertainty and inconclusive outcomes honestly.

Total/contact loss values across these variants use different objectives and
cannot prove superiority. Better penetration accompanied by excessive stretching,
floating or slower inference is a tradeoff, not unconditional improvement.

## What a strict barrier would require

A feasible initial state has all relevant distances strictly positive. A real
barrier solver keeps optimization iterates in that domain, typically by checking
a proposed displacement with CCD and bounding position-space line-search steps.
The optimizer here changes network weights and does not do that position solve.

A contact-correction layer would take a raw predicted configuration and solve a
small constrained correction (with fixed handles, moving-body contact, consistent
velocity update, and a defined gradient through the correction). Merely moving
each penetrating vertex along one normal does not guarantee feasibility for
triangles, multiple contacts or a moving obstacle. Feasibility must be checked
after correction and failure handled. Endpoint correction alone is not CCD.

This additional solver is a separate architectural experiment. Until it exists,
V2 intentionally makes only a relaxed-penalty claim.
