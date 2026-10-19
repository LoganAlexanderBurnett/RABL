"""Display-only units for manuscript figures.

Native ML outputs remain the dimensionless neutron-power multiplier n.
The underlying microreactor reference thermal power is 6 MW. The physical
power curve (and its conformal interval endpoints) is P [MW] = 6 n.
Do not apply this conversion when reading simulator P_MW or calculating
native normalized forecast/UQ accuracy metrics.
"""
POWER_MW_PER_N = 6.0


def plot_index_and_scale(key, names):
    """Native model target index and factor to display the requested quantity."""
    if key in ("P_MW", "n"):
        return names.index("n"), POWER_MW_PER_N
    return names.index(key), 1.0
