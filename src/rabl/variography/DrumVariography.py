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
def sill_for_vmax(vmax: float, p_all: float, N: int) -> float:
    """
    Return the velocity sill needed so |v| <= vmax with probability p_all.

    Notes:
        - N is the number of time nodes in the grid; M = N - 1 velocity points.
        - Uses a conservative union-bound to convert p_all to a per-point bound.
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    if vmax <= 0:
        raise ValueError("vmax must be > 0")
    if not (0.0 < p_all < 1.0):
        raise ValueError("p_all must be in (0, 1)")

    M = N - 1
    p_pt = 1.0 - (1.0 - p_all) / M
    z = NormalDist().inv_cdf((p_pt + 1.0) / 2.0)
    sigma_v = vmax / z
    return sigma_v**2


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


def solve_ell_for_sigma_theta(
    kernel: str,
    delta_t: float,
    T: float,
    N: int,
    sigma_theta_target: float,
    *,
    sill: float,
    nugget: float = 0.0,
    ell_low: float = 1e-6,
    ell_high: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> float:
    """
    Solve for ell such that sigma_theta_end(...) == sigma_theta_target.

    Notes:
        - N is the number of time nodes in the grid; M = N - 1 velocity points.
        - delta_t is the uniform spacing between time nodes.
        - T is only used as a sanity check / default upper bracket.
    """
    if sigma_theta_target <= 0:
        raise ValueError("sigma_theta_target must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")
    if N < 2:
        raise ValueError("N must be >= 2")
    if delta_t <= 0:
        raise ValueError("delta_t must be > 0")
    if sill < 0 or nugget < 0:
        raise ValueError("sill and nugget must be >= 0")

    implied_T = delta_t * (N - 1)
    if abs(implied_T - T) > 1e-6 * max(1.0, T):
        pass

    M = N - 1
    sigma_min = delta_t * math.sqrt((sill + nugget) * M)
    sigma_max = delta_t * math.sqrt(sill * (M**2) + nugget * M)

    if not (sigma_min <= sigma_theta_target <= sigma_max):
        raise ValueError(
            "Target sigma not achievable with given sill/nugget/grid. "
            f"sigma_min≈{sigma_min:.6g}, sigma_max≈{sigma_max:.6g}, "
            f"target={sigma_theta_target:.6g}."
        )

    if ell_high is None:
        ell_high = 100.0 * max(T, delta_t)

    def f(ell: float) -> float:
        return sigma_theta_end(kernel, delta_t, N, ell, sill, nugget) - sigma_theta_target

    lo, hi = ell_low, ell_high
    flo, fhi = f(lo), f(hi)

    expand_count = 0
    while fhi < 0 and expand_count < 50:
        hi *= 2.0
        fhi = f(hi)
        expand_count += 1

    if flo > 0:
        raise ValueError("ell_low already gives sigma above target; decrease ell_low.")
    if fhi < 0:
        raise ValueError("Failed to bracket solution; increase ell_high.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if fmid > 0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


# =============================================================================
# Data container for one profile
# =============================================================================
@dataclass
class DrumProfile:
    t: np.ndarray
    theta_deg: np.ndarray
    v_deg_s: np.ndarray
    a_deg_s2: np.ndarray

    def save_csv(self, path: str) -> None:
        header = ["Time(s)", "Drum_Angle(deg)", "Drum_Velocity(deg/s)", "Drum_Acceleration(deg/s^2)"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for ti, th, vi, ai in zip(self.t, self.theta_deg, self.v_deg_s, self.a_deg_s2):
                w.writerow([float(ti), float(th), float(vi), float(ai)])

    def save_mat(self, path: str) -> None:
        t = np.asarray(self.t).squeeze()
        theta = np.asarray(self.theta_deg).squeeze()
        v = np.asarray(self.v_deg_s).squeeze()
        a = np.asarray(self.a_deg_s2).squeeze()

        table = np.column_stack([t, theta, v, a])
        savemat(
            path,
            {"profile": table},
            format="4",
        )


# =============================================================================
# Generator class (Gaussian-consistent branching on PAST ANGLES)
# =============================================================================
class DrumProfileGenerator:
    """
    Prior is defined on VELOCITY v(t) as a zero-mean GP with Matérn correlation.
    We generate ANGLE theta(t) by linear integration:
        theta = theta0*1 + B v
    Therefore theta is Gaussian:
        theta ~ N(theta0*1,  Sigma_theta),  Sigma_theta = B K_v B^T
    Branching conditions on past ANGLES (Gaussian-consistent):
        theta_future | theta_past  is Gaussian.
    Velocity and acceleration are then computed by finite differences of theta.
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
        Forward-Euler integration on nodes using left-endpoint velocity:
            theta[i] = theta0 + sum_{j=0}^{i-1} v[j] * dt[j]
        with dt[j] = t[j+1]-t[j].

        This is linear: theta = theta0*1 + B v

        Note: v[N-1] is unused by this discretization (last column of B is zeros).
        """
        t = np.asarray(t, float)
        N = len(t)
        dt = np.diff(t)  # length N-1

        B = np.zeros((N, N), dtype=float)
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

        Kv = self.build_velocity_cov(t)
        B = self.build_integration_matrix(t)
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
    # Compute v and a from theta via finite differences (node-wise)
    # -----------------------------
    @staticmethod
    def velocity_and_accel_from_theta(t: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Node-wise backward differences:

          v[0] = 0
          v[i] = (theta[i] - theta[i-1]) / dt[i-1]   for i>=1

          a[0] = 0
          a[i] = (v[i] - v[i-1]) / dt[i-1]           for i>=1

        This is linear in theta (hence Gaussian if theta is Gaussian).
        """
        t = np.asarray(t, float)
        theta = np.asarray(theta, float)
        dt = np.diff(t)

        v = np.empty_like(theta)
        v[0] = 0.0
        v[1:] = np.diff(theta) / dt

        a = np.empty_like(theta)
        a[0] = 0.0
        a[1:] = np.diff(v) / dt

        return v, a

    # -----------------------------
    # Sample a full angle path (Gaussian) via v then integrate
    # -----------------------------
    def sample_theta(self, t: np.ndarray, theta0: float, rng: np.random.Generator) -> np.ndarray:
        """
        Sample v ~ N(0, K_v), then theta = theta0*1 + B v.
        """
        Kv, B, _ = self._get_theta_gaussian_mats(t)
        v = rng.multivariate_normal(mean=np.zeros(len(t)), cov=Kv)
        theta = theta0 + B @ v
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
        for i in range(n_realizations):
            theta = self.sample_theta(t, baseline_angle_deg, rng)
            v, a = self.velocity_and_accel_from_theta(t, theta)
            profiles.append(DrumProfile(t=t, theta_deg=theta, v_deg_s=v, a_deg_s2=a))
            print(f"Profiles generated: [{i+1}/{n_realizations}]")
        return profiles

    def solve_params_for_sigma_theta(
        self,
        t_grid: np.ndarray,
        sigma_theta_target: float,
        v_max: float = 1.0,
        p_all: float = 0.999,
        nugget: float = 0.0,
        *,
        ell_bounds: Tuple[float, float | None] = (1e-6, None),
    ) -> Tuple[float, float]:
        """
        Solve (ell, sill) that match sigma_theta_target at the final node.

        Notes:
            - N is the number of time nodes in the grid; M = N - 1 velocity points.
            - delta_t is the uniform spacing between time nodes.
            - Returns (ell, sill). The caller can assign these to the generator.
        """
        t_grid = np.asarray(t_grid, float)
        if t_grid.size < 2:
            raise ValueError("t_grid must have at least 2 points")

        delta_t = float(np.mean(np.diff(t_grid)))
        N = t_grid.size
        sill = sill_for_vmax(v_max, p_all, N)
        ell_low, ell_high = ell_bounds
        ell = solve_ell_for_sigma_theta(
            kernel=self.kernel,
            delta_t=delta_t,
            T=float(t_grid[-1] - t_grid[0]),
            N=N,
            sigma_theta_target=sigma_theta_target,
            sill=sill,
            nugget=nugget,
            ell_low=ell_low,
            ell_high=ell_high,
        )
        return ell, sill

    # -----------------------------
    # Conditional future angles given past angles
    # -----------------------------
    def conditional_future_params_from_past_angles(
        self,
        t: np.ndarray,
        idx_branch: int,
        theta_past: np.ndarray,
        theta0: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Condition on past ANGLES (indices 0..idx_branch inclusive).
        Return conditional mean/cov for future angles (idx_branch+1..N-1).

        theta ~ N(mu, Sigma_theta), mu = theta0 * 1
        """
        t = np.asarray(t, float)
        theta_past = np.asarray(theta_past, float)

        N = len(t)
        S = idx_branch + 1
        if S <= 0 or S > N:
            raise ValueError("idx_branch out of range")
        if theta_past.shape[0] != S:
            raise ValueError("theta_past length must be idx_branch+1")

        if S == N:
            return np.zeros((0,), float), np.zeros((0, 0), float)

        _, _, Sigma_theta = self._get_theta_gaussian_mats(t)

        Sigma_pp = Sigma_theta[:S, :S]
        Sigma_pf = Sigma_theta[:S, S:]
        Sigma_fp = Sigma_theta[S:, :S]
        Sigma_ff = Sigma_theta[S:, S:]

        mu_p = theta0 * np.ones(S)
        mu_f = theta0 * np.ones(N - S)

        # mu_cond = mu_f + Sigma_fp Sigma_pp^{-1} (theta_past - mu_p)
        alpha = np.linalg.solve(Sigma_pp, (theta_past - mu_p))
        mu_cond = mu_f + Sigma_fp @ alpha

        # Sigma_cond = Sigma_ff - Sigma_fp Sigma_pp^{-1} Sigma_pf
        A = np.linalg.solve(Sigma_pp, Sigma_pf)
        Sigma_cond = Sigma_ff - Sigma_fp @ A
        Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)

        # stabilize
        if Sigma_cond.shape[0] > 0:
            Sigma_cond = Sigma_cond + self.cond_jitter * np.eye(Sigma_cond.shape[0])

        return mu_cond, Sigma_cond

    # -----------------------------
    # Branch once: condition on entire past ANGLE history
    # -----------------------------
    def branch(self, original: DrumProfile, t_branch: float, seed: Optional[int] = None) -> DrumProfile:
        t = original.t
        idx = int(np.argmin(np.abs(t - t_branch)))
        if not np.isclose(t[idx], t_branch, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"t_branch={t_branch} must coincide with a time node in the grid. "
                f"Closest node is t[{idx}]={t[idx]}."
            )

        N = len(t)
        if idx >= N - 1:
            return original

        S = idx + 1
        theta_past = original.theta_deg[:S].copy()
        theta0 = float(original.theta_deg[0])

        mu_cond, Sigma_cond = self.conditional_future_params_from_past_angles(
            t=t, idx_branch=idx, theta_past=theta_past, theta0=theta0
        )

        rng = np.random.default_rng(seed)
        theta_fut = rng.multivariate_normal(mean=mu_cond, cov=Sigma_cond)

        theta_full = original.theta_deg.copy()
        theta_full[:S] = theta_past
        theta_full[S:] = theta_fut

        v_full, a_full = self.velocity_and_accel_from_theta(t, theta_full)
        return DrumProfile(t=t, theta_deg=theta_full, v_deg_s=v_full, a_deg_s2=a_full)

    # -----------------------------
    # Branch N times efficiently
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

        t = original.t
        idx = int(np.argmin(np.abs(t - t_branch)))
        if not np.isclose(t[idx], t_branch, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"t_branch={t_branch} must coincide with a time node in the grid. "
                f"Closest node is t[{idx}]={t[idx]}."
            )

        if idx >= len(t) - 1:
            return [original for _ in range(n_branches)]

        S = idx + 1
        theta_past = original.theta_deg[:S].copy()
        theta0 = float(original.theta_deg[0])

        mu_cond, Sigma_cond = self.conditional_future_params_from_past_angles(
            t=t, idx_branch=idx, theta_past=theta_past, theta0=theta0
        )
        n_fut = mu_cond.size
        if n_fut == 0:
            return [original for _ in range(n_branches)]

        # Cholesky once
        try:
            L = np.linalg.cholesky(Sigma_cond)
        except np.linalg.LinAlgError:
            Sigma_cond2 = Sigma_cond + (10.0 * self.cond_jitter) * np.eye(n_fut)
            L = np.linalg.cholesky(Sigma_cond2)

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(size=(n_branches, n_fut))
        theta_fut_draws = mu_cond[None, :] + Z @ L.T

        branched: List[DrumProfile] = []
        for k in range(n_branches):
            theta_full = original.theta_deg.copy()
            theta_full[:S] = theta_past
            theta_full[S:] = theta_fut_draws[k]
            v_full, a_full = self.velocity_and_accel_from_theta(t, theta_full)
            branched.append(DrumProfile(t=t, theta_deg=theta_full, v_deg_s=v_full, a_deg_s2=a_full))
            print(f"Branches generated: [{k+1}/{n_branches}]")

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
        axes[0].set_title(f"Base vs {len(branched_list)} Branched Profiles (Gaussian conditioning on past angles)")
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
        axes[2].set_ylim(-0.3, 0.3)
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
        help="If set, overrides --ell/--sill and solves parameters from the target sigma.",
    )
    parser.add_argument("--vmax", type=float, default=1.0)
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
        sill = sill_for_vmax(args.vmax, args.p_all, N)
        ell = solve_ell_for_sigma_theta(
            kernel=args.kernel,
            delta_t=delta_t,
            T=float(t_grid[-1] - t_grid[0]),
            N=N,
            sigma_theta_target=args.sigma_theta_target,
            sill=sill,
            nugget=args.nug,
        )

    gen = DrumProfileGenerator(
        kernel=args.kernel,
        ell=ell,
        sill_v_deg2_s2=sill,
        nugget_v_deg2_s2=args.nug,
        jitter_frac=1e-10,
        cond_jitter=1e-10,
    )

    profiles = gen.generate(t_grid, n_realizations=args.n_profiles, baseline_angle_deg=45.0, seed=999)
    base = profiles[0]

    branched_profiles = gen.branch_N_times(
        base,
        t_branch=80.0,
        n_branches=args.n_branches,
        seed=123,
    )

    gen.plot_profiles(profiles, title="Generated Drum Profiles")
    gen.plot_base_vs_branched(base, branched_profiles, t_branch=80.0)
