from dataclasses import dataclass, replace

import numpy as np
import toppra as ta
from toppra.constraint import ConstraintType


@dataclass(frozen=True)
class StageConstraints:
    """Per-stage admissible set Omega_i: a[i] * u + b[i] * x <= c[i] and 0 <= x <= xmax[i]."""

    gridpoints: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    xmax: np.ndarray

    @property
    def num_stages(self) -> int:
        return len(self.gridpoints) - 1

    @property
    def deltas(self) -> np.ndarray:
        return np.diff(self.gridpoints)

    def with_rows(self, a, b, c) -> "StageConstraints":
        return replace(
            self,
            a=np.concatenate([self.a, a], axis=1),
            b=np.concatenate([self.b, b], axis=1),
            c=np.concatenate([self.c, c], axis=1),
        )


def build_stage_constraints(path, gridpoints, constraints) -> StageConstraints:
    """Collect toppra canonical-linear constraints in collocation form."""
    gridpoints = np.asarray(gridpoints, dtype=float)
    n_points = len(gridpoints)
    a_rows, b_rows, c_rows = [], [], []
    xmax = np.full(n_points, np.inf)
    for constraint in constraints:
        if constraint.get_constraint_type() != ConstraintType.CanonicalLinear:
            raise NotImplementedError("only canonical linear constraints are supported")
        a, b, c, F, g, ubound, xbound = constraint.compute_constraint_params(path, gridpoints)
        if xbound is not None:
            xmax = np.minimum(xmax, xbound[:, 1])
        if ubound is not None and np.isfinite(ubound).any():
            raise NotImplementedError("u bounds are not used by joint velocity/acceleration limits")
        if a is None:
            continue
        F = np.broadcast_to(F, (n_points,) + np.shape(F)[-2:])
        g = np.broadcast_to(g, (n_points, F.shape[1]))
        a_rows.append(np.einsum("nrk,nk->nr", F, a))
        b_rows.append(np.einsum("nrk,nk->nr", F, b))
        c_rows.append(g - np.einsum("nrk,nk->nr", F, c))
    if not a_rows:
        empty = np.zeros((n_points, 0))
        return StageConstraints(gridpoints, empty, empty.copy(), empty.copy(), xmax)
    return StageConstraints(
        gridpoints,
        np.concatenate(a_rows, axis=1),
        np.concatenate(b_rows, axis=1),
        np.concatenate(c_rows, axis=1),
        xmax,
    )


def joint_limit_constraints(path, gridpoints, vlim, alim) -> StageConstraints:
    vlim = np.asarray(vlim, dtype=float)
    alim = np.asarray(alim, dtype=float)
    velocity = ta.constraint.JointVelocityConstraint(np.stack([-vlim, vlim], axis=1))
    acceleration = ta.constraint.JointAccelerationConstraint(np.stack([-alim, alim], axis=1))
    return build_stage_constraints(path, gridpoints, [velocity, acceleration])
