# Contact geometry v1

This experiment changes only the normal cloth-to-body collision distance. The
default `legacy_plane` mode remains available and the existing cubic penalty,
weight ramp, clearance `eps`, inertia/stretch/bend/gravity terms, and friction
criterion are unchanged.

## Distance and penalty

For every free garment vertex `x`, query the globally closest point `q` on all
triangles of the next-frame body mesh. The query includes triangle interiors,
edges and vertices. The sign is negative inside the closed body and positive
outside, classified with the generalized winding number. Thus

```
d = signed_distance(x, body_next)
loss_contact = weight(iter) * sum(clamp(eps - d, min=0) ** 3)
```

This is a point-to-triangle signed-distance replacement, **not** an area-weighted
contact integral and **not** a barrier. The contact loss value and its gradient
will change because `d` changes, even though the formula applied to `d` is the
same. `eps`, the cubic, the vertex-wise summation and the iteration weight ramp
remain fixed for this ablation.

Run with `conda activate hood_clean`, then:

```
export HOOD_PROJECT=/home/zhudf/projects/HOOD
export HOOD_DATA=/home/zhudf/data/hood_data/hood_data
taskset -c 0 python -u train_baseline.py config=contact/geometry_v1_seed57
```

The contact config copies the seed-57 baseline and selects
`criterions.postcvpr.collision_penalty.contact_geometry=signed_mesh`. A clean
Git checkout is required; each run writes a distinct experiment directory with
the resolved YAML and source/data provenance. No long run is started as part of
the geometry unit check.

## How to judge the change

Geometry correctness and training benefit are different questions.

* Unit cases compare against analytic point-triangle distances for face, edge
  and vertex regions; inside/outside signs; a case where nearest face center is
  not the nearest triangle; orientation reversal; finite-difference gradients;
  and repulsion direction at zero clearance.
* The legacy criterion test confirms the default mode produces the same cubic
  penalty value as before. The signed mode test confirms that it uses the
  target body mesh, preserves the cubic formula and backpropagates a finite
  gradient.
* The actual SMPL female topology was checked as closed and consistently
  oriented; points just outside and inside one body face gave signed distances
  of approximately `+1e-4 m` and `-1e-4 m`.
* Fixed validation records mesh-distance penetration fraction, mean/max depth,
  minimum clearance, and free-vertex penetration. Compare these metrics and
  rollout videos against the baseline at equal training steps and on the same
  fixed sequences. Total losses are not directly comparable because the
  collision distance has changed.

This verifies endpoint vertex-to-body distance. It does not test every point on
the cloth triangles, cloth self-collision, or the continuous path between two
frames. The winding sign assumes a closed, consistently oriented body surface;
the implementation rejects open or nonmanifold topologies.

## Scope of contact handling

The original `collision_penalty` and `CollisionPreprocessor` handle garment to
body contact. They do not implement garment self-contact. The friction criterion
is also garment to body, but still selects body triangle centers and its own
proximity mask. The learned body-edge graph is another cloth/body interaction
signal and is unchanged. This experiment replaces the geometry in the cubic
normal collision penalty only; it does not claim to replace every contact
approximation in HOOD.

## Barrier, CCD, and filtered line search

The cubic penalty is finite when `d <= 0`, so it encourages recovery from
penetration but cannot enforce strict nonpenetration. A distance barrier would
change the energy model and requires a feasible positive-distance starting
state. Contact-area quadrature weights would also change how the energy scales
with mesh resolution. Neither is part of geometry v1.

CCD answers a separate temporal question: whether the path from the current to
the proposed next state crosses the body, even if both endpoints are outside.
Filtered line search uses a CCD-derived step-size bound while an optimizer
searches positions for a step that reduces an incremental potential. HOOD's
training optimizer updates network weights; its predicted cloth state is not
obtained from a position-space Newton solve with a line search. Therefore this
distance replacement adds no CCD or filtered line search and does not prevent
tunneling on its own.
