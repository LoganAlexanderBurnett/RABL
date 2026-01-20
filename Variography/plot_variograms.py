import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, List, Tuple, Optional


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


def _pair_ell_sill(
    ell_list: Iterable[float],
    sill_list: Iterable[float]
) -> List[Tuple[float, float]]:
    """
    If lengths match -> zip them.
    Otherwise -> Cartesian product.
    """
    ell_list = list(ell_list)
    sill_list = list(sill_list)

    if len(ell_list) == len(sill_list):
        return list(zip(ell_list, sill_list))

    return [(ell, sill) for ell in ell_list for sill in sill_list]


def _rho_and_gamma(
    kernel: str,
    h: np.ndarray,
    ell: float,
    sill: float,
    nugget: float
) -> Tuple[np.ndarray, np.ndarray]:
    if kernel == "matern32":
        rho = matern32_corr(h, ell)
    elif kernel == "matern52":
        rho = matern52_corr(h, ell)
    else:
        raise ValueError("kernel must be 'matern32' or 'matern52'")

    gamma = sill * (1.0 - rho) + nugget
    return rho, gamma


def _unique_preserve_order(values):
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _plot_kernels_overlay(
    ax_rho,
    ax_gamma,
    kernels: List[str],
    h: np.ndarray,
    ell_sill_pairs: List[Tuple[float, float]],
    nugget: float
) -> None:
    # Kernel linestyle
    ls_map = {"matern32": "-", "matern52": "--"}
    name_map = {
        "matern32": r"Matérn $\nu=\frac{3}{2}$",
        "matern52": r"Matérn $\nu=\frac{5}{2}$"
    }

    # Extract unique ell values (correlation does not depend on sill)
    ell_only = _unique_preserve_order([ell for ell, _ in ell_sill_pairs])

    # ------------------------------------------------------------
    # 1) Correlation plot: ONLY (kernel, ell)
    # ------------------------------------------------------------
    # Record colors used so gamma curves for the same (kernel, ell)
    # can reuse the same color for visual consistency.
    color_map = {}  # (kernel, ell) -> color

    for kernel in kernels:
        for ell in ell_only:
            rho = matern32_corr(h, ell) if kernel == "matern32" else matern52_corr(h, ell)
            label = f"{name_map[kernel]} | ell={ell:g}"
            line = ax_rho.plot(h, rho, linestyle=ls_map[kernel], label=label)[0]
            color_map[(kernel, ell)] = line.get_color()

    ax_rho.set_title("Matérn correlation")
    ax_rho.set_xlabel("h")
    ax_rho.set_ylabel(r"$\rho(h)$")
    ax_rho.grid(True, linestyle="--", alpha=0.5)
    ax_rho.legend(fontsize=9)

    # ------------------------------------------------------------
    # 2) Variogram plot: (kernel, ell, sill)
    # ------------------------------------------------------------
    for kernel in kernels:
        for ell, sill in ell_sill_pairs:
            rho, gamma = _rho_and_gamma(kernel, h, ell, sill, nugget)
            label = f"{name_map[kernel]} | ell={ell:g}, sill={sill:g}"

            ax_gamma.plot(
                h,
                gamma,
                linestyle=ls_map[kernel],
                color=color_map.get((kernel, ell), None),  # reuse rho color
                label=label,
                alpha=0.9
            )

    ax_gamma.set_title("Matérn variogram")
    ax_gamma.set_xlabel("h")
    ax_gamma.set_ylabel(r"$\gamma(h)$")
    ax_gamma.grid(True, linestyle="--", alpha=0.5)
    ax_gamma.legend(fontsize=9)

    # ------------------------------------------------------------
    # Equation annotations
    # ------------------------------------------------------------
    corr_eq = (
        r"$\rho_{3/2}(h)=(1+r_{3/2})e^{-r_{3/2}},\quad r_{3/2}=\sqrt{3}\frac{|h|}{\ell}$"
        "\n"
        r"$\rho_{5/2}(h)=\left(1+r_{5/2}+\frac{r_{5/2}^2}{3}\right)e^{-r_{5/2}},\quad r_{5/2}=\sqrt{5}\frac{|h|}{\ell}$"
    )

    vario_eq = (
        r"$\gamma(h)=\mathrm{sill}\,\left(1-\rho(h)\right)+\mathrm{nugget}$"
        "\n"
        r"(same form for both Matérn kernels; only $\rho(h)$ changes)"
    )

    ax_rho.text(
        0.02, 0.02, corr_eq,
        transform=ax_rho.transAxes,
        fontsize=10,
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6")
    )

    ax_gamma.text(
        0.6, 0.02, vario_eq,
        transform=ax_gamma.transAxes,
        fontsize=10,
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6")
    )



def plot_matern_curves(
    kernel: str = "both",                 # "matern32", "matern52", or "both"
    ell_list: Iterable[float] = (3.0, 4.0, 5.0),
    sill_list: Iterable[float] = (0.1, 0.2, 0.3),
    num_points: int = 200,
    nugget: float = 0.0,
    output: Optional[str] = None,         # e.g. "matern.png" to save
    h_max_multiplier: float = 4.0,        # plot h in [0, multiplier * max(ell)]
    figsize: Optional[Tuple[float, float]] = None,
):
    ell_list = list(ell_list)
    sill_list = list(sill_list)
    if not ell_list:
        raise ValueError("ell_list cannot be empty")
    if not sill_list:
        raise ValueError("sill_list cannot be empty")

    if kernel not in ("matern32", "matern52", "both"):
        raise ValueError("kernel must be 'matern32', 'matern52', or 'both'")

    ell_sill_pairs = _pair_ell_sill(ell_list, sill_list)

    max_ell = max(ell_list)
    h = np.linspace(0.0, h_max_multiplier * max_ell, num_points)

    kernels = ["matern32", "matern52"] if kernel == "both" else [kernel]

    if figsize is None:
        figsize = (14, 14)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)
    ax_rho, ax_gamma = axes

    _plot_kernels_overlay(ax_rho, ax_gamma, kernels, h, ell_sill_pairs, nugget)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved figure to: {output}")

    plt.show()
    return fig, axes



def main() -> None:
    plot_matern_curves(
    kernel="both",
    ell_list=[3.0, 5.0, 7.0],
    sill_list=[0.1, 0.15, 0.20],
    num_points=200,
    nugget=0.0,
    )

if __name__ == "__main__":
    main()
