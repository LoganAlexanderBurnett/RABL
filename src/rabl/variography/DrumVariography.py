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
        attempts = 0
        max_attempts = max(100, 100 * n_realizations)
        while len(profiles) < n_realizations:
            theta = self.sample_theta(t, baseline_angle_deg, rng)
            attempts += 1
            if not self._theta_within_bounds(theta):
                if attempts >= max_attempts:
                    raise RuntimeError(
                        "Failed to generate a bounded profile within the attempt limit. "
                        "Consider relaxing constraints or increasing max_attempts."
                    )
                continue
            v, a = self.velocity_and_accel_from_theta(t, theta)
            profiles.append(DrumProfile(t=t, theta_deg=theta, v_deg_s=v, a_deg_s2=a))
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
        attempts = 0
        max_attempts = 1000
        while True:
            theta_fut = rng.multivariate_normal(mean=mu_cond, cov=Sigma_cond)
            theta_full = original.theta_deg.copy()
            theta_full[:S] = theta_past
            theta_full[S:] = theta_fut
            attempts += 1
            if self._theta_within_bounds(theta_full):
                v_full, a_full = self.velocity_and_accel_from_theta(t, theta_full)
                return DrumProfile(t=t, theta_deg=theta_full, v_deg_s=v_full, a_deg_s2=a_full)
            if attempts >= max_attempts:
                raise RuntimeError(
                    "Failed to generate a bounded branched profile within the attempt limit. "
                    "Consider relaxing constraints or increasing max_attempts."
                )

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
        branched: List[DrumProfile] = []
        attempts = 0
        max_attempts = max(1000, 1000 * n_branches)
        while len(branched) < n_branches:
            z = rng.standard_normal(size=n_fut)
            theta_fut = mu_cond + L @ z
            theta_full = original.theta_deg.copy()
            theta_full[:S] = theta_past
            theta_full[S:] = theta_fut
            attempts += 1
            if not self._theta_within_bounds(theta_full):
                if attempts >= max_attempts:
                    raise RuntimeError(
                        "Failed to generate bounded branches within the attempt limit. "
                        "Consider relaxing constraints or increasing max_attempts."
                    )
                continue
            v_full, a_full = self.velocity_and_accel_from_theta(t, theta_full)
            branched.append(DrumProfile(t=t, theta_deg=theta_full, v_deg_s=v_full, a_deg_s2=a_full))
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
        print(f"sigma_theta_target  = {args.sigma_theta_target:g} deg")
        print(f"computed sill_v     = {sill:.6g} (deg/s)^2")
        print(f"computed sigma_v    = {sigma_v:.6g} deg/s (pointwise)")
        print(f"estimated vmax      = {vmax_est:.6g} deg/s (p_all={args.p_all:g})")
        print(f"sigma_theta_end chk = {sigma_check:.6g} deg")

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
