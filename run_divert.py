import numpy as np
import scipy.io
import OMSimulator as oms
import matplotlib.pyplot as plt


def _derive_vel_acc(t: np.ndarray, angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive vel/acc from angle using numerical differentiation."""
    if t.ndim != 1 or angle_deg.ndim != 1 or len(t) != len(angle_deg):
        raise ValueError("t and angle_deg must be 1D arrays of equal length.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time vector must be strictly increasing for differentiation.")
    vel = np.gradient(angle_deg, t, edge_order=2)
    acc = np.gradient(vel, t, edge_order=2)
    return vel, acc


def _validate_divert(f: np.ndarray) -> None:
    if np.any(~np.isfinite(f)):
        raise ValueError("f_divert contains NaN/inf.")
    fmin = float(np.min(f))
    fmax = float(np.max(f))
    if fmin < 0.0 or fmax > 1.0:
        raise ValueError(f"f_divert must be in [0,1]. Got min={fmin:.6g}, max={fmax:.6g}.")


def generate_profile_5col(
    in_mat: str,
    out_mat: str = "divert_profile.mat",
    mode: str = "linear",
    seed: int = 42,
) -> str:
    """Generate a 5-column MAT profile: [time, angle, vel, acc, f_divert].

    - If the input profile has >=4 columns (time, angle, vel, acc), we reuse vel/acc.
    - Otherwise we derive vel/acc numerically from the angle.
    - We do NOT clip/clamp f_divert; we validate and hard-fail if outside [0,1].
    """
    og = scipy.io.loadmat(in_mat)
    if "profile" not in og:
        raise KeyError(f"Missing 'profile' variable in {in_mat}")

    profile = np.asarray(og["profile"], dtype=float)
    if profile.ndim != 2 or profile.shape[1] < 2:
        raise ValueError(f"'profile' must be an NxM matrix with at least 2 columns. Got shape {profile.shape}.")

    t = profile[:, 0]
    angle = profile[:, 1]

    if profile.shape[1] >= 4:
        vel = profile[:, 2]
        acc = profile[:, 3]
    else:
        vel, acc = _derive_vel_acc(t, angle)

    if mode == "linear":
        # Simple piecewise schedule: 0 -> 0.1 -> 0.3 -> 0.5 with 10s ramps
        f = np.zeros_like(t)
        values = [0.0, 0.1, 0.3, 0.5]
        change_times = [50.0, 100.0, 150.0]
        ramp_dur = 10.0

        current = values[0]
        f[:] = current
        for k, next_val in enumerate(values[1:]):
            t0 = change_times[k]
            t1 = t0 + ramp_dur

            ramp_mask = (t >= t0) & (t <= t1)
            hold_mask = (t > t1) & (t <= (change_times[k + 1] if k + 1 < len(change_times) else t[-1]))

            if np.any(ramp_mask):
                alpha = (t[ramp_mask] - t0) / (t1 - t0)
                f[ramp_mask] = (1 - alpha) * current + alpha * next_val
            if np.any(hold_mask):
                f[hold_mask] = next_val

            current = next_val

    elif mode == "gp":
        rng = np.random.default_rng(seed)

        # Minimal GP-like smooth random signal (not a full covariance build; avoids heavy deps)
        # This is intentionally simple; replace with your preferred GP generator if needed.
        white = rng.standard_normal(len(t))
        # Smooth with a Gaussian kernel in time index space
        sigma = 30  # smoothing width in samples (tuned by eye)
        x = np.arange(len(t))
        kern = np.exp(-0.5 * ((x - x[len(t)//2]) / sigma) ** 2)
        kern /= np.sum(kern)
        smooth = np.convolve(white, kern, mode="same")

        # Scale/shift into a reasonable range without clipping:
        # Map smooth to mean 0.2, std 0.1, then *validate*.
        f = 0.2 + 0.1 * (smooth / (np.std(smooth) + 1e-12))

    else:
        raise ValueError("mode must be 'linear' or 'gp'.")

    _validate_divert(f)

    out = np.column_stack((t, angle, vel, acc, f))
    scipy.io.savemat(out_mat, {"profile": out}, format="4")
    return out_mat


def quick_plot(mat_file: str) -> None:
    d = scipy.io.loadmat(mat_file)
    prof = d["profile"]
    t = prof[:, 0]
    angle = prof[:, 1]
    f = prof[:, 4]

    plt.figure()
    plt.plot(t, angle)
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("angle [deg]")
    plt.title("Angle profile")

    plt.figure()
    plt.plot(t, f)
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("f_divert [-]")
    plt.title("Diversion fraction profile")
    plt.show()


if __name__ == "__main__":
    # Example:
    #   python run_divert_updated.py
    in_mat = "../drumv4out.mat"
    out_mat = generate_profile_5col(in_mat, out_mat="divert_profile.mat", mode="linear")
    print(f"Wrote {out_mat}")
    quick_plot(out_mat)
