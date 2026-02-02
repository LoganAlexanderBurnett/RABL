import csv
from dataclasses import dataclass
from statistics import NormalDist
from typing import Optional, List, Tuple
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat


# =============================================================================
# Matérn correlation kernels (for VELOCITY)
# =============================================================================

def matern32_corr(h, ell):
    """Matérn ν=3/2 correlation ρ(h); ρ(0)=1."""
    h = np.asarray(h, dtype=float)
    r = np.sqrt(3.0) * np.abs(h) / ell
    return (1.0 + r) * np.exp(-r)

def matern52_corr(h, ell):
    """Matérn ν=5/2 correlation ρ(h); ρ(0)=1."""
    h = np.asarray(h, dtype=float)
    r = np.sqrt(5.0) * np.abs(h) / ell
    return (1.0 + r + (r**2) / 3.0) * np.exp(-r)


# =============================================================================
# Variogram parameter helpers
# =============================================================================
def _A_of_ell(kernel: str, delta_t: float, N: int, ell: float) -> float:
    """
    Return A(ell) for the final-node variance under a uniform grid.

    Notes:
        - N is the number of time nodes in the grid; M = N - 1 velocity points.
        - delta_t is the uniform spacing between time nodes.
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    if delta_t <= 0:
        raise ValueError("delta_t must be > 0")
    if ell <= 0:
        raise ValueError("ell must be > 0")

    if kernel == "matern32":
        corr = matern32_corr
    elif kernel == "matern52":
        corr = matern52_corr
    else:
        raise ValueError("kernel must be 'matern32' or 'matern52'")

    M = N - 1
    k = np.arange(1, M, dtype=float)
    rho = corr(k * delta_t, ell)

    A = M * 1.0 + 2.0 * np.sum((M - k) * rho)
    return float(A)


def sill_for_sigma_theta_target(
    kernel: str,
    delta_t: float,
    N: int,
    ell: float,
    sigma_theta_target: float,
    nugget: float = 0.0,
) -> float:
    """
    Solve for the velocity sill that matches a target sigma at the final node.

    Notes:
        - N is the number of time nodes in the grid; M = N - 1 velocity points.
        - delta_t is the uniform spacing between time nodes.
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    if delta_t <= 0:
        raise ValueError("delta_t must be > 0")
    if ell <= 0:
        raise ValueError("ell must be > 0")
    if sigma_theta_target <= 0:
        raise ValueError("sigma_theta_target must be > 0")
    if nugget < 0:
        raise ValueError("nugget must be >= 0")

    M = N - 1
    A = _A_of_ell(kernel, delta_t, N, ell)
    var = sigma_theta_target**2
    sill = ((var / (delta_t**2)) - nugget * M) / A
    if sill < 0:
        raise ValueError("Computed sill must be >= 0 for the target sigma.")
    return sill


def vmax_for_sill(sill: float, p_all: float, N: int) -> float:
    """
    Conservative estimate of vmax such that max_i |v_i| <= vmax with probability ~p_all,
    using the same union-bound logic as sill_for_vmax.
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    if sill < 0:
        raise ValueError("sill must be >= 0")
    if not (0.0 < p_all < 1.0):
        raise ValueError("p_all must be in (0, 1)")

    M = N - 1
    p_pt = 1.0 - (1.0 - p_all) / M
    z = NormalDist().inv_cdf((p_pt + 1.0) / 2.0)
    return z * math.sqrt(sill)


def sigma_theta_end(
    kernel: str,
    delta_t: float,
    N: int,
    ell: float,
    sill: float,
    nugget: float = 0.0,
) -> float:
    """
    Return the std dev of theta at the final node.

    Notes:
        - N is the number of time nodes in the grid; M = N - 1 velocity points.
        - delta_t is the uniform spacing between time nodes.
    """
    if sill < 0 or nugget < 0:
        raise ValueError("sill and nugget must be >= 0")
    M = N - 1
    A = _A_of_ell(kernel, delta_t, N, ell)
    var = (delta_t**2) * (sill * A + nugget * M)
    return math.sqrt(var)


# =============================================================================
# Data container for one profile
# =============================================================================
@dataclass
class DrumProfile:
    t: np.ndarray
    theta_deg: np.ndarray
    v_deg_s: np.ndarray
    a_deg_s2: np.ndarray

    # -----------------------------
    # I/O helpers
    # -----------------------------
    @staticmethod
    def _as_1d(arr: np.ndarray, name: str) -> np.ndarray:
        a = np.asarray(arr).squeeze()
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1D after squeeze; got shape {a.shape}")
        return a.astype(float, copy=False)

    @classmethod
    def _require_len(cls, arr: np.ndarray, n: int, name: str) -> np.ndarray:
        a = cls._as_1d(arr, name)
        if a.size != n:
            raise ValueError(f"{name} must have length {n}, got {a.size}")
        return a

    @classmethod
    def _pad_short(cls, arr: np.ndarray, n: int, name: str, fill: float = np.nan) -> np.ndarray:
        a = cls._as_1d(arr, name)
        if a.size > n:
            raise ValueError(f"{name} length {a.size} exceeds t length {n}")
        if a.size == n:
            return a
        out = np.full(n, fill, dtype=float)
        out[: a.size] = a
        return out

    def save_csv(self, path: str, missing: str = "nan") -> None:
        """Save an N-row CSV (N = len(t)) even if v/a are shorter.

        Rules:
            - t and theta must be length N.
            - v and a may be length <= N; if shorter, they are padded to N with NaN.
            - If missing == "blank", NaN values are written as empty cells.
              If missing == "nan", NaN values are written as the string 'nan' (via float(np.nan)).
        """
        if missing not in {"nan", "blank"}:
            raise ValueError("missing must be 'nan' or 'blank'")

        t = self._require_len(self.t, len(self.t), "t")
        N = t.size
        theta = self._require_len(self.theta_deg, N, "theta_deg")
        v = self._pad_short(self.v_deg_s, N, "v_deg_s", fill=np.nan)
        a = self._pad_short(self.a_deg_s2, N, "a_deg_s2", fill=np.nan)

        header = ["Time(s)", "Drum_Angle(deg)", "Drum_Velocity(deg/s)", "Drum_Acceleration(deg/s^2)"]

        def cell(x: float):
            if missing == "blank" and (x is None or math.isnan(float(x))):
                return ""
            return float(x)

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(N):
                w.writerow([cell(t[i]), cell(theta[i]), cell(v[i]), cell(a[i])])

    def save_mat(self, path: str) -> None:
        """Save a MAT file with an N×4 numeric table in 'profile'.

        Rules match save_csv:
            - t and theta must be length N.
            - v and a may be shorter and will be padded with NaN to length N.
        """
        t = self._require_len(self.t, len(self.t), "t")
        N = t.size
        theta = self._require_len(self.theta_deg, N, "theta_deg")
        v = self._pad_short(self.v_deg_s, N, "v_deg_s", fill=np.nan)
        a = self._pad_short(self.a_deg_s2, N, "a_deg_s2", fill=np.nan)

        table = np.column_stack([t, theta, v, a])
        savemat(
            path,
            {"profile": table},
            format="4",
        )


# =============================================================================
# Generator class (Gaussian-consistent branching on PAST VELOCITIES)
# =============================================================================
class DrumProfileGenerator:
    """
    Prior is defined on VELOCITY v(t) as a zero-mean GP with Matérn correlation.
    Generate ANGLE theta(t) by linear integration:
        theta = theta0*1 + B v
    Therefore theta is Gaussian:
        theta ~ N(theta0*1,  Sigma_theta),  Sigma_theta = B K_v B^T
    Branching conditions on past INTERVAL VELOCITIES (Gaussian-consistent):
        v_future | v_past is Gaussian, then integrate to get theta.

    Notes on outputs in this file:
        - For *generated* profiles (generate()), velocity is treated as the primary
          sampled quantity and is defined on the N-1 *intervals* between time nodes.
          We condition on v[0]=0 on that interval grid, keep the sampled interval velocities,
          and compute theta by integration (theta = theta0*1 + B v_intervals).
          For output, we pad v and a to length N by appending a trailing NaN at index N-1.
        - For *velocity-conditioned branching* (branch/branch_N_times), we keep the past
          interval velocities fixed, sample future interval velocities from the conditional GP,
          integrate to get the branched angle, compute acceleration from velocity differences,
          and pad v/a with a trailing NaN for output.
    """

    def __init__(
        self,
        kernel: str = "matern52",
        ell: float = 5.0,
        sill_v_deg2_s2: float = 0.1,
        nugget_v_deg2_s2: float = 0.0,
        jitter_frac: float = 1e-10,
        cond_jitter: float = 1e-10,
    ):
        self.kernel = kernel
        self.ell = float(ell)
        self.sill_v = float(sill_v_deg2_s2)
        self.nugget_v = float(nugget_v_deg2_s2)
        self.jitter_frac = float(jitter_frac)
        self.cond_jitter = float(cond_jitter)

        # cache for last t-grid matrices
        self._cache_key = None
        self._cache = None

    @staticmethod
    def _theta_within_bounds(theta: np.ndarray, min_deg: float = 0.0, max_deg: float = 180.0) -> bool:
        theta = np.asarray(theta, float)
        return bool(np.all((theta >= min_deg) & (theta <= max_deg)))

    # -----------------------------
    # Velocity correlation ρ(|Δt|)
    # -----------------------------
    def _rho(self, dt):
        if self.kernel == "matern32":
            return matern32_corr(dt, self.ell)
        elif self.kernel == "matern52":
            return matern52_corr(dt, self.ell)
        else:
            raise ValueError("kernel must be 'matern32' or 'matern52'")

    # -----------------------------
    # Velocity covariance K_v
    # -----------------------------
    def build_velocity_cov(self, t: np.ndarray) -> np.ndarray:
        """
        K_v(i,j) = sill_v * ρ(|t_i - t_j|) + nugget_v * δ_ij + jitter
        Units: (deg/s)^2
        """
        t = np.asarray(t, dtype=float)
        dt = np.abs(t[:, None] - t[None, :])
        rho = self._rho(dt)

        Kv = self.sill_v * rho + self.nugget_v * np.eye(len(t))
        Kv += (self.sill_v + self.nugget_v) * self.jitter_frac * np.eye(len(t))
        return Kv

    # -----------------------------
    # Integration matrix B for theta = theta0*1 + B v
    # -----------------------------
    @staticmethod
    def build_integration_matrix(t: np.ndarray) -> np.ndarray:
        """
        Forward-Euler integration on nodes using left-endpoint *interval* velocity:
            theta[i] = theta0 + sum_{j=0}^{i-1} v_int[j] * dt[j]
        with dt[j] = t[j+1]-t[j] and v_int[j] defined on the interval [t[j], t[j+1]).

        This is linear: theta = theta0*1 + B v_int

        Shapes:
            - N = len(t) time nodes
            - M = N-1 interval velocities
            - B is (N, M)
        """
        t = np.asarray(t, float)
        N = len(t)
        dt = np.diff(t)  # length M = N-1
        M = N - 1

        B = np.zeros((N, M), dtype=float)
        # row 0 is all zeros (theta[0] = theta0)
        for i in range(1, N):
            # theta[i] depends on v[0..i-1] with weights dt[0..i-1]
            B[i, :i] = dt[:i]
        return B

    # -----------------------------
    # Build Sigma_theta = B K_v B^T (cached)
    # -----------------------------
    def _get_theta_gaussian_mats(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (Kv, B, Sigma_theta) with caching for the given t-grid.
        """
        t = np.asarray(t, float)
        key = (len(t), float(t[0]), float(t[-1]), float(np.mean(np.diff(t))))
        if self._cache_key == key and self._cache is not None:
            return self._cache

        # Interval-velocity grid: left endpoints of each interval.
        # M = N-1 velocities live on [t[i], t[i+1]) and are indexed by i=0..N-2.
        t_v = t[:-1]

        Kv = self.build_velocity_cov(t_v)          # (M, M)
        B = self.build_integration_matrix(t)       # (N, M)
        Sigma_theta = B @ Kv @ B.T

        # Stabilize (Sigma_theta can be ill-conditioned due to integration)
        diag_scale = np.mean(np.diag(Sigma_theta)) if Sigma_theta.size else 1.0
        Sigma_theta += max(diag_scale, 1.0) * self.jitter_frac * np.eye(len(t))

        # Symmetrize for numerical safety
        Sigma_theta = 0.5 * (Sigma_theta + Sigma_theta.T)

        self._cache_key = key
        self._cache = (Kv, B, Sigma_theta)
        return Kv, B, Sigma_theta

    # -----------------------------
    # Compute a on the interval grid from interval velocities
    # -----------------------------
    @staticmethod
    def accel_intervals_from_v(t: np.ndarray, v_intervals: np.ndarray) -> np.ndarray:
        """Compute interval accelerations from interval velocities.

        Here v_intervals[j] is defined on [t[j], t[j+1]) for j=0..M-1, where M=N-1.
        We return a_intervals of length M, aligned to the same left-endpoint grid (t[:-1]).

        We use forward differences where possible and a backward difference at the end:
            a[j]   = (v[j+1] - v[j]) / dt[j]          for j = 0..M-2
            a[M-1] = (v[M-1] - v[M-2]) / dt[M-2]      (backward at the end)

        This keeps a_intervals fully defined (no internal NaNs). The final *node*
        acceleration at t[N-1] is undefined and should be padded as NaN separately.
        """
        t = np.asarray(t, float)
        v = np.asarray(v_intervals, float)
        if t.ndim != 1 or v.ndim != 1:
            raise ValueError("t and v_intervals must be 1D")
        if t.size < 2:
            raise ValueError("t must have at least 2 points")

        dt = np.diff(t)
        M = t.size - 1
        if v.size != M:
            raise ValueError(f"v_intervals must have length N-1={M}; got {v.size}")
        if np.any(dt <= 0):
            raise ValueError("t must be strictly increasing")

        a = np.empty_like(v)
        if M == 1:
            a[0] = 0.0
            return a

        a[:-1] = np.diff(v) / dt[:-1]
        a[-1] = (v[-1] - v[-2]) / dt[-2]
        return a

    # -----------------------------
    # Sample a full angle path (Gaussian) via v then integrate
    # -----------------------------

    def sample_v_conditioned_v0(self, Kv: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample v ~ N(0, Kv) conditioned on v[0] == 0.

        Exact Gaussian conditioning identity:
            If v ~ N(0, K), then
                v_cond = v - (K[:,0] / K[0,0]) * v[0]
            has the distribution of v | (v[0]=0), and satisfies v_cond[0]=0.

        We add a small denominator jitter (cond_jitter) for numerical stability.
        """
        Kv = np.asarray(Kv, float)
        if Kv.ndim != 2 or Kv.shape[0] != Kv.shape[1]:
            raise ValueError(f"Kv must be square; got shape {Kv.shape}")
        N = Kv.shape[0]
        v = rng.multivariate_normal(mean=np.zeros(N), cov=Kv)
        denom = float(Kv[0, 0]) + float(self.cond_jitter)
        if denom == 0.0:
            raise ValueError("Kv[0,0] + cond_jitter is zero; cannot condition on v[0]=0")
        v_cond = v - (Kv[:, 0] / denom) * v[0]
        v_cond[0] = 0.0  # enforce exactly
        return v_cond

    def sample_velocity_and_theta_conditioned_v0(
        self, t: np.ndarray, theta0: float, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample (v_intervals, theta) with the constraint v_intervals[0] == 0.

        Notes:
            - v_intervals has length M = N-1 and lives on the interval grid t_v = t[:-1].
            - The conditioning is applied directly to the M×M interval-velocity covariance Kv.
        """
        Kv, B, _ = self._get_theta_gaussian_mats(t)
        v_int = self.sample_v_conditioned_v0(Kv, rng)
        # Sanity: v_int is interval-based (M=N-1) and conditioned to start at 0.
        if v_int.size != t.size - 1:
            raise RuntimeError(f"Expected interval velocity length N-1={t.size-1}, got {v_int.size}")
        # sample_v_conditioned_v0 enforces v_int[0]=0 exactly.
        if v_int.size > 0 and v_int[0] != 0.0:
            raise RuntimeError(f"Conditioning failed: v_int[0]={v_int[0]}")
        theta = theta0 + B @ v_int
        return v_int, theta

    def sample_velocity_and_theta(
        self, t: np.ndarray, theta0: float, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample (v_intervals, theta) where v_intervals ~ N(0, K_v) and theta = theta0*1 + B v_intervals."""
        Kv, B, _ = self._get_theta_gaussian_mats(t)
        v_int = rng.multivariate_normal(mean=np.zeros(Kv.shape[0]), cov=Kv)
        theta = theta0 + B @ v_int
        return v_int, theta

    def sample_theta(self, t: np.ndarray, theta0: float, rng: np.random.Generator) -> np.ndarray:
        """Back-compat: sample theta only (still via sampling v then integrating)."""
        _, theta = self.sample_velocity_and_theta(t, theta0, rng)
        return theta

    # -----------------------------
    # Generate full profiles
    # -----------------------------
    def generate(
        self,
        t_grid: np.ndarray,
        n_realizations: int,
        baseline_angle_deg: float = 45.0,
        seed: Optional[int] = None,
    ) -> List[DrumProfile]:
        t = np.asarray(t_grid, float)
        rng = np.random.default_rng(seed)

        profiles: List[DrumProfile] = []
        attempts = 0
        max_attempts = max(100, 100 * n_realizations)
        while len(profiles) < n_realizations:
            v_int, theta = self.sample_velocity_and_theta_conditioned_v0(t, baseline_angle_deg, rng)
            # Sanity checks for interval velocities and conditioning
            if v_int.size != t.size - 1:
                raise RuntimeError(f"Expected v_int length N-1={t.size-1}, got {v_int.size}")
            if v_int.size > 0 and v_int[0] != 0.0:
                raise RuntimeError(f"Expected conditioned v_int[0]=0, got {v_int[0]}")
            attempts += 1
            if not self._theta_within_bounds(theta):
                if attempts >= max_attempts:
                    raise RuntimeError(
                        "Failed to generate a bounded profile within the attempt limit. "
                        "Consider relaxing constraints or increasing max_attempts."
                    )
                continue

            # Treat velocity as the primary quantity: keep sampled *interval* velocities.
            # Compute interval acceleration from interval velocity differences.
            a_int = self.accel_intervals_from_v(t, v_int)

            # Output schema is node-aligned length N, with trailing NaN at index N-1.
            v_nodes = np.concatenate([v_int, [np.nan]])
            a_nodes = np.concatenate([a_int, [np.nan]])

            profiles.append(DrumProfile(t=t, theta_deg=theta, v_deg_s=v_nodes, a_deg_s2=a_nodes))
            print(f"Profiles generated: [{len(profiles)}/{n_realizations}]")
        return profiles

    def solve_params_for_sigma_theta(
        self,
        t_grid: np.ndarray,
        sigma_theta_target: float,
        ell: float,
        nugget: float = 0.0,
        *,
        update_instance: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Solve (ell, sill, nugget) that match sigma_theta_target at the final node.

        Notes:
            - N is the number of time nodes in the grid; M = N - 1 velocity points.
            - delta_t is the uniform spacing between time nodes.
            - Returns (ell, sill, nugget). Optionally updates the generator instance.
        """
        t_grid = np.asarray(t_grid, float)
        if t_grid.size < 2:
            raise ValueError("t_grid must have at least 2 points")

        delta_t = float(np.mean(np.diff(t_grid)))
        N = t_grid.size
        sill = sill_for_sigma_theta_target(
            kernel=self.kernel,
            delta_t=delta_t,
            N=N,
            ell=ell,
            sigma_theta_target=sigma_theta_target,
            nugget=nugget,
        )
        if update_instance:
            self.ell = float(ell)
            self.sill_v = float(sill)
            self.nugget_v = float(nugget)
        return ell, sill, nugget

    # -----------------------------
    # Conditional future interval velocities given past interval velocities
    # -----------------------------
    def conditional_future_params_from_past_velocities(
        self,
        t: np.ndarray,
        n_past_intervals: int,
        v_past: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Condition the interval-velocity GP on past interval velocities.

        Definitions:
            - N = len(t) time nodes.
            - M = N-1 interval velocities v[0..M-1] live on intervals [t[i], t[i+1]).
            - n_past_intervals = P is how many *leading* interval velocities are fixed/observed.
              The future velocities are v[P:] of length F = M - P.

        This returns (mu_cond, Sigma_cond) for v_future = v[P:].

        Notes:
            - The GP prior is v ~ N(0, Kv) on the interval grid t_v = t[:-1].
        """
        t = np.asarray(t, float)
        if t.ndim != 1:
            raise ValueError("t must be 1D")
        N = t.size
        if N < 2:
            raise ValueError("t must have at least 2 points")
        M = N - 1

        P = int(n_past_intervals)
        if P < 0 or P > M:
            raise ValueError(f"n_past_intervals must be in [0, {M}], got {P}")

        v_past = np.asarray(v_past, float).squeeze()
        if P == 0:
            if v_past.size != 0:
                raise ValueError("v_past must be empty when n_past_intervals=0")
        else:
            if v_past.ndim != 1 or v_past.size != P:
                raise ValueError(f"v_past must have length {P}, got shape {v_past.shape}")

        # No future velocities to sample.
        if P == M:
            return np.zeros((0,), float), np.zeros((0, 0), float)

        Kv, _, _ = self._get_theta_gaussian_mats(t)
        if Kv.shape != (M, M):
            raise RuntimeError(f"Expected Kv shape {(M, M)}, got {Kv.shape}")

        # If P==0, future is just the full prior.
        if P == 0:
            mu_cond = np.zeros((M,), float)
            Sigma_cond = Kv.copy()
            Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)
            Sigma_cond = Sigma_cond + self.cond_jitter * np.eye(M)
            return mu_cond, Sigma_cond

        # Partition into past/future blocks
        Sigma_pp = Kv[:P, :P]
        Sigma_pf = Kv[:P, P:]
        Sigma_fp = Kv[P:, :P]
        Sigma_ff = Kv[P:, P:]

        # Stabilize Sigma_pp for solve
        Sigma_pp = 0.5 * (Sigma_pp + Sigma_pp.T)
        Sigma_pp = Sigma_pp + self.cond_jitter * np.eye(P)

        # mu_cond = Sigma_fp Sigma_pp^{-1} v_past   (prior mean is 0)
        alpha = np.linalg.solve(Sigma_pp, v_past)
        mu_cond = Sigma_fp @ alpha

        # Sigma_cond = Sigma_ff - Sigma_fp Sigma_pp^{-1} Sigma_pf
        A = np.linalg.solve(Sigma_pp, Sigma_pf)
        Sigma_cond = Sigma_ff - Sigma_fp @ A
        Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)

        # Stabilize conditional covariance
        if Sigma_cond.shape[0] > 0:
            Sigma_cond = Sigma_cond + self.cond_jitter * np.eye(Sigma_cond.shape[0])

        return mu_cond, Sigma_cond

    # -----------------------------
    # Branch once: condition on past INTERVAL VELOCITIES, sample future velocities, then integrate
    # -----------------------------
    def branch(self, original: DrumProfile, t_branch: float, seed: Optional[int] = None) -> DrumProfile:
        t = np.asarray(original.t, float)
        idx = int(np.argmin(np.abs(t - t_branch)))
        if not np.isclose(t[idx], t_branch, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"t_branch={t_branch} must coincide with a time node in the grid. "
                f"Closest node is t[{idx}]={t[idx]}."
            )

        N = t.size
        M = N - 1
        if idx >= N - 1:
            # no future intervals; nothing to branch
            return original

        # Extract original interval velocities (length M). Stored v is node-aligned with a trailing NaN.
        v_orig_nodes = np.asarray(original.v_deg_s, float).squeeze()
        if v_orig_nodes.ndim != 1 or v_orig_nodes.size != N:
            raise ValueError(f"original.v_deg_s must be length N={N}")
        v_orig_int = v_orig_nodes[:M].copy()
        if np.any(~np.isfinite(v_orig_int)):
            raise ValueError("original interval velocities contain NaN/inf in the first N-1 entries")

        theta0 = float(original.theta_deg[0])

        # Past interval count is idx (intervals 0..idx-1). For idx==0, treat v[0]=0 as the only 'past' constraint.
        if idx == 0:
            P = 1
            v_past = np.array([0.0], dtype=float)
        else:
            P = idx
            v_past = v_orig_int[:P].copy()

        mu_cond, Sigma_cond = self.conditional_future_params_from_past_velocities(
            t=t, n_past_intervals=P, v_past=v_past
        )
        n_fut = mu_cond.size
        # Assemble past+future into a full interval-velocity vector of length M.
        rng = np.random.default_rng(seed)

        Kv, B, _ = self._get_theta_gaussian_mats(t)
        attempts = 0
        max_attempts = 1000
        while True:
            if n_fut == 0:
                v_new_int = v_past.copy()
            else:
                v_future = rng.multivariate_normal(mean=mu_cond, cov=Sigma_cond)
                v_new_int = np.concatenate([v_past, v_future])

            if v_new_int.size != M:
                raise RuntimeError(f"Expected v_new_int length M={M}, got {v_new_int.size}")

            theta_new = theta0 + B @ v_new_int
            attempts += 1
            if self._theta_within_bounds(theta_new):
                a_int = self.accel_intervals_from_v(t, v_new_int)
                v_nodes = np.concatenate([v_new_int, [np.nan]])
                a_nodes = np.concatenate([a_int, [np.nan]])
                return DrumProfile(t=t, theta_deg=theta_new, v_deg_s=v_nodes, a_deg_s2=a_nodes)

            if attempts >= max_attempts:
                raise RuntimeError(
                    "Failed to generate a bounded branched profile within the attempt limit. "
                    "Consider relaxing constraints or increasing max_attempts."
                )

    # -----------------------------
    # Branch N times efficiently (velocity-space)
    # -----------------------------
    def branch_N_times(
        self,
        original: DrumProfile,
        t_branch: float,
        n_branches: int,
        seed: Optional[int] = None,
    ) -> List[DrumProfile]:
        if n_branches <= 0:
            return []

        t = np.asarray(original.t, float)
        idx = int(np.argmin(np.abs(t - t_branch)))
        if not np.isclose(t[idx], t_branch, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"t_branch={t_branch} must coincide with a time node in the grid. "
                f"Closest node is t[{idx}]={t[idx]}."
            )

        N = t.size
        M = N - 1
        if idx >= N - 1:
            return [original for _ in range(n_branches)]

        # Extract original interval velocities
        v_orig_nodes = np.asarray(original.v_deg_s, float).squeeze()
        if v_orig_nodes.ndim != 1 or v_orig_nodes.size != N:
            raise ValueError(f"original.v_deg_s must be length N={N}")
        v_orig_int = v_orig_nodes[:M].copy()
        if np.any(~np.isfinite(v_orig_int)):
            raise ValueError("original interval velocities contain NaN/inf in the first N-1 entries")

        theta0 = float(original.theta_deg[0])

        # Past interval count is idx; for idx==0, enforce v[0]=0 as the only constraint.
        if idx == 0:
            P = 1
            v_past = np.array([0.0], dtype=float)
        else:
            P = idx
            v_past = v_orig_int[:P].copy()

        mu_cond, Sigma_cond = self.conditional_future_params_from_past_velocities(
            t=t, n_past_intervals=P, v_past=v_past
        )
        n_fut = mu_cond.size
        if n_fut == 0:
            # All velocities are fixed by the past; just return identical copies.
            out: List[DrumProfile] = []
            Kv, B, _ = self._get_theta_gaussian_mats(t)
            theta_new = theta0 + B @ v_past
            a_int = self.accel_intervals_from_v(t, v_past)
            v_nodes = np.concatenate([v_past, [np.nan]])
            a_nodes = np.concatenate([a_int, [np.nan]])
            prof = DrumProfile(t=t, theta_deg=theta_new, v_deg_s=v_nodes, a_deg_s2=a_nodes)
            return [prof for _ in range(n_branches)]

        # Cholesky once for fast sampling
        try:
            L = np.linalg.cholesky(Sigma_cond)
        except np.linalg.LinAlgError:
            Sigma_cond2 = Sigma_cond + (10.0 * self.cond_jitter) * np.eye(n_fut)
            L = np.linalg.cholesky(Sigma_cond2)

        Kv, B, _ = self._get_theta_gaussian_mats(t)
        rng = np.random.default_rng(seed)

        branched: List[DrumProfile] = []
        attempts = 0
        max_attempts = max(1000, 1000 * n_branches)

        while len(branched) < n_branches:
            z = rng.standard_normal(size=n_fut)
            v_future = mu_cond + L @ z
            v_new_int = np.concatenate([v_past, v_future])

            theta_new = theta0 + B @ v_new_int
            attempts += 1
            if not self._theta_within_bounds(theta_new):
                if attempts >= max_attempts:
                    raise RuntimeError(
                        "Failed to generate bounded branches within the attempt limit. "
                        "Consider relaxing constraints or increasing max_attempts."
                    )
                continue

            a_int = self.accel_intervals_from_v(t, v_new_int)
            v_nodes = np.concatenate([v_new_int, [np.nan]])
            a_nodes = np.concatenate([a_int, [np.nan]])
            branched.append(DrumProfile(t=t, theta_deg=theta_new, v_deg_s=v_nodes, a_deg_s2=a_nodes))
            print(f"Branches generated: [{len(branched)}/{n_branches}]")

        return branched

# -----------------------------
    # Plot: base vs branches (stacked panels)
    # -----------------------------
    @staticmethod
    def plot_base_vs_branched(base: DrumProfile, branched_list: List[DrumProfile], t_branch: float):
        t = base.t
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

        # Angle
        for i, bp in enumerate(branched_list):
            axes[0].plot(t, bp.theta_deg, alpha=0.6)
        axes[0].plot(t, base.theta_deg, label="base", linewidth=1.0, color='k')
        axes[0].axvline(t_branch, linestyle="--", label="branch time")
        axes[0].set_ylabel("Angle [deg]")
        axes[0].set_title(f"Base vs {len(branched_list)} Branched Profiles (Gaussian conditioning on past velocities)")
        axes[0].grid(True)
        axes[0].legend(ncol=2, fontsize=9)

        # Velocity
        for bp in branched_list:
            axes[1].plot(t, bp.v_deg_s, alpha=0.6)
        axes[1].plot(t, base.v_deg_s, label="base", linewidth=1.0, color='k')
        axes[1].axvline(t_branch, linestyle="--")
        axes[1].set_ylabel("Velocity [deg/s]")
        axes[1].grid(True)

        # Acceleration
        for bp in branched_list:
            axes[2].plot(t, bp.a_deg_s2, alpha=0.6)
        axes[2].plot(t, base.a_deg_s2, label="base", linewidth=1.0, color='k')
        axes[2].axvline(t_branch, linestyle="--")
        axes[2].set_ylabel("Acceleration [deg/s²]")
        axes[2].set_xlabel("Time [s]")
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Plot: multiple generated profiles (stacked panels)
    # -----------------------------
    @staticmethod
    def plot_profiles(profiles: List[DrumProfile], title: str = "Generated Drum Profiles"):
        if not profiles:
            raise ValueError("profiles list is empty")

        t = profiles[0].t
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

        for profile in profiles:
            axes[0].plot(t, profile.theta_deg, alpha=0.5)
        axes[0].set_ylabel("Angle [deg]")
        axes[0].set_title(title)
        axes[0].grid(True)

        for profile in profiles:
            axes[1].plot(t, profile.v_deg_s, alpha=0.5)
        axes[1].set_ylabel("Velocity [deg/s]")
        axes[1].grid(True)

        for profile in profiles:
            axes[2].plot(t, profile.a_deg_s2, alpha=0.5)
        axes[2].set_ylabel("Acceleration [deg/s²]")
        axes[2].set_xlabel("Time [s]")
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and plot drum profile variography samples.")
    parser.add_argument("--kernel", default="matern52", choices=("matern32", "matern52"))
    parser.add_argument("--ell", type=float, default=5.0)
    parser.add_argument("--sill", type=float, default=0.1)
    parser.add_argument("--nug", type=float, default=0.0)
    parser.add_argument("--n_profiles", type=int, default=10)
    parser.add_argument("--n_branches", type=int, default=10)
    parser.add_argument(
        "--sigma_theta_target",
        type=float,
        default=None,
        help="If set, overrides --sill and solves sill from the target sigma and --ell.",
    )
    parser.add_argument("--p_all", type=float, default=0.999)
    return parser


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    t_grid = np.linspace(0.0, 200.0, 2001)

    ell = args.ell
    sill = args.sill
    if args.sigma_theta_target is not None:
        N = t_grid.size
        delta_t = float(np.mean(np.diff(t_grid)))
        sill = sill_for_sigma_theta_target(
            kernel=args.kernel,
            delta_t=delta_t,
            N=N,
            ell=ell,
            sigma_theta_target=args.sigma_theta_target,
            nugget=args.nug,
        )
        sigma_v = math.sqrt(sill)
        sigma_check = sigma_theta_end(
            kernel=args.kernel,
            delta_t=delta_t,
            N=N,
            ell=ell,
            sill=sill,
            nugget=args.nug,
        )
        vmax_est = vmax_for_sill(sill, args.p_all, N)
        print("=== Parameter solution ===")
        print(f"kernel              = {args.kernel}")
        print(f"ell                 = {args.ell:g} s")
        print(f"delta_t             = {delta_t:g} s")
        print(f"N (time nodes)      = {len(t_grid)}")
        print(f"T                  ~= {delta_t * (len(t_grid) - 1):g} s")
        print(f"sigma_theta_target  = {args.sigma_theta_target:.6f} deg")
        print(f"computed sill_v     = {sill:.6g} (deg/s)^2")
        print(f"computed sigma_v    = {sigma_v:.6g} deg/s (per-sample std dev)")
        print(f"estimated vmax      = {vmax_est:.6g} deg/s (~{args.p_all:.2%} confidence)")
        print(f"sigma_theta_end chk = {sigma_check:.6f} deg")

    gen = DrumProfileGenerator(
        kernel=args.kernel,
        ell=ell,
        sill_v_deg2_s2=sill,
        nugget_v_deg2_s2=args.nug,
        jitter_frac=1e-10,
        cond_jitter=1e-10,
    )

    equilibrium_drum_angle = 45.0
    profiles = gen.generate(t_grid, n_realizations=args.n_profiles, baseline_angle_deg=equilibrium_drum_angle, seed=999)
    base = profiles[0]

    branched_profiles = gen.branch_N_times(
        base,
        t_branch=80.0,
        n_branches=args.n_branches,
        seed=123,
    )

    gen.plot_profiles(profiles, title="Generated Drum Profiles")
    gen.plot_base_vs_branched(base, branched_profiles, t_branch=80.0)
