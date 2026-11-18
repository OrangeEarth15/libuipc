# Half Plane

A half plane is the set of points $\mathbf{x}$ with
$(\mathbf{x} - \mathbf{P}) \cdot \mathbf{N} \ge 0$. Each instance holds the
point $\mathbf{P}$ and the normal $\mathbf{N}$ in the instance attributes `P`
and `N`; `ground(height, N)` and `halfplane(P, N)` create them.

Distances to the plane are measured along the unit normal: the backend and the
sanity check normalize `N` when they read the scene, so its length does not
matter. A zero `N` has no direction, and the sanity check rejects a scene that
has one.
