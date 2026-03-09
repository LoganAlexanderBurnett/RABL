"""Finite-difference utilities for branchpoint/uncertainty analysis."""

from __future__ import annotations

import numpy as np


def finite_difference(x_t: np.ndarray, *, order: int = 4, dt: float = 1.0) -> np.ndarray:
    """Compute first derivatives for a generic 2D trajectory array ``x(t)``.

    Parameters
    ----------
    x_t:
        Array with shape ``(n_steps, n_variables)``.
    order:
        Finite-difference order, either ``2`` or ``4``. Defaults to ``4``.
    dt:
        Timestep spacing. Defaults to ``1.0``.

    Returns
    -------
    np.ndarray
        Derivative array with the same shape as ``x_t``.
    """
    x_t = np.asarray(x_t, dtype=np.float64)
    if x_t.ndim != 2:
        raise ValueError(f"Expected x_t to be 2D (n_steps, n_variables), got shape {x_t.shape}.")
    if order not in (2, 4):
        raise ValueError(f"Unsupported order={order}. Expected 2 or 4.")
    if dt <= 0:
        raise ValueError(f"dt must be positive; got {dt}.")

    n_steps, _ = x_t.shape
    if n_steps < 2:
        raise ValueError("Need at least 2 timesteps to compute a derivative.")

    dx_dt = np.empty_like(x_t, dtype=np.float64)

    if n_steps == 2:
        slope = (x_t[1] - x_t[0]) / dt
        dx_dt[0] = slope
        dx_dt[1] = slope
        return dx_dt

    # 2nd-order one-sided stencils at both ends.
    dx_dt[0] = (-3.0 * x_t[0] + 4.0 * x_t[1] - x_t[2]) / (2.0 * dt)
    dx_dt[-1] = (3.0 * x_t[-1] - 4.0 * x_t[-2] + x_t[-3]) / (2.0 * dt)

    if order == 2 or n_steps < 5:
        # 2nd-order central stencil for interior points.
        for i in range(1, n_steps - 1):
            dx_dt[i] = (x_t[i + 1] - x_t[i - 1]) / (2.0 * dt)
        return dx_dt

    # order == 4
    # Near boundaries: 2nd-order central fallback.
    dx_dt[1] = (x_t[2] - x_t[0]) / (2.0 * dt)
    dx_dt[-2] = (x_t[-1] - x_t[-3]) / (2.0 * dt)

    # Interior: 4th-order central stencil.
    for i in range(2, n_steps - 2):
        dx_dt[i] = (-x_t[i + 2] + 8.0 * x_t[i + 1] - 8.0 * x_t[i - 1] + x_t[i - 2]) / (12.0 * dt)

    return dx_dt
