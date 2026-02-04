# VARIOGRAPHY AND SIMULATION PARAMETERS

# Change for each batch as necessary
NUM_PROFILES       = 10
ELL                = 10.0
SIGMA_THETA_TARGET = 2.5
BATCH_NUMBER       = 1
SEED               = 123    # avoid duplicate drum profiles across batches, but ensure reproducibility
PLOT_BATCH_NUMBERS = [1]

# Should never need to change
NUGGET_V_DEG2_S2   = 0.0
KERNEL             = "matern52"
T_GRID_DURATION    = 200.0
T_GRID_INTERVALS   = 2000
BASELINE_ANGLE_DEG = 45.0

STEADY_STATE = {
    "TN2": 878.1500000000001,
    "Tm": 1150.15,
    "Thp": 1073.15,
    "Tf": 1173.15,
    "c[1]": 111.29100352191402,
    "c[2]": 176.12853827610243,
    "c[3]": 49.22970537728693,
    "c[4]": 45.4634608511303,
    "c[5]": 5.533226788012196,
    "c[6]": 0.7900465760468193,
    "n": 1.0,
    "rho_dollars": 0.0,
    "Q_to_steam": 6000000.0,
    "drumAngleDeg": 45.0,
}
