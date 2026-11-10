"""
Every tunable parameter of the model lives here, so a run is fully described by this file.

Dimensionless variables (Siversky & Zharkova 2009):

    tau = t * 2 pi e^4 n0 lnLambda / sqrt(2 me E0^3)      time
    s   = xi * pi e^4 lnLambda / E0^2 = xi / xi0           column depth
    z   = E / E0                                           electron energy
    mu  = cos(theta)                                       pitch-angle cosine
    eps = E_field * E0 / (2 pi e^3 n0 lnLambda)            return-current electric field
    n   = n_p / n0                                         ambient density

Units are CGS throughout.

SINGLE CONFIGURATION
--------------------
The solver, the pipeline and the figures all use this one file: the pipeline and the
figures read the distribution f saved by scripts/run_solver.py, so everything comes
from the same Fokker-Planck solution. Values follow Siversky & Zharkova (2009).
"""

# CGS constants
e_esu = 4.80320471257e-10        # electron charge [esu]
me_g = 9.1093837015e-28          # electron mass [g]
k_B = 1.380649e-16               # Boltzmann constant [erg/K]
erg_per_keV = 1.602176634e-9     # 1 keV in erg
lnLambda = 20.0                  # Coulomb logarithm

# Constants for the bremsstrahlung cross sections
MEC2_KEV = 510.99895             # electron rest energy [keV]
R0_CM = 2.8179403262e-13         # classical electron radius [cm]
ALPHA_FINE = 1.0 / 137.035999084 # fine-structure constant
Z_ION = 1.0                      # target ion charge (hydrogen)


# Normalization scales (Siversky & Zharkova 2009)

E0_keV = 12.0                    # energy scale [keV]; z = E / E0
n0 = 1.0e10                      # density scale [cm^-3]
F0 = 1.0e10                      # injected energy-flux scale used in the conductivity

zmin = 0.1                       # lowest z  (E = 1.2 keV)
zmax = 100.0                     # highest z (E = 1200 keV)

xi_min = 2.29e17                 # top of the column-depth grid [cm^-2]
xi_max = 1.01e23                 # bottom of the column-depth grid [cm^-2]


# Grid Size
NS = 100                         # points in column depth s (log-spaced)
NZ = 60                           # points in energy z (log-spaced)
NMU = 80                         # points in pitch-angle cosine mu (linear in [-1, 1])

XI_DIAG = 1.0e18                 # column depth [cm^-2] used for convergence checks and diagnostics


# Numerics
DTAU = 2.0e-5                    # pseudo-time step
N_ITER = 2600                    # maximum number of iterations
TOL = 4.0e-5                     # convergence tolerance on the relative change of f at XI_DIAG
CHECK_EVERY = 200                # check convergence every this many iterations
MIN_ITER = 500                   # never stop before this iteration

EPS_EVERY = 2                    # update the electric field every this many iterations
OMEGA_EPS = 0.35                 # under-relaxation of the electric field update

# Scale factor on the conductivity (Eq. 13). Kept because the atmosphere is an approximate
# tanh profile, not the exact hydrodynamic table of the paper.
#   If C+E stays too close to C, lower it (e.g. 5e-7).
#   If C+E dominates too much, raise it (e.g. 2e-6).
SIGMA_FACTOR = 1.0e-6


# Injected beam (eq. 15)
DELTA_LOW = 10.0                 # low-energy index that smooths the turnover of the power law
DELTA_MU = 0.2                   # width of the injected pitch-angle distribution around mu = 1


# MAGNETIC FIELD (hybrid model of the paper)
# d ln B / d xi = ln(B_RATIO) / XI_C_MAG for xi < XI_C_MAG, and 0 deeper.
B_RATIO = 3.0                    # B at XI_C_MAG divided by B at the top
XI_C_MAG = 1.0e20                # column depth where the convergence stops [cm^-2]


# Approx. astmosphere (tanh transition region)

XI_TR = 2.0e19                   # column depth of the transition region [cm^-2]
W_TR = 0.45                      # width of the transition in decades of xi

T_cor = 2.0e7                    # coronal temperature [K]
T_chr = 1.0e4                    # chromospheric temperature [K]

n_cor = 1.0                      # coronal density in units of n0
n_chr = 1.0e3                    # chromospheric density in units of n0


# Bremsstrahlung emission and polarization
NPHI_BREMS = 16                  # quadrature points in gyrophase phi (use 6-8 to debug, 16-24 for accuracy)
SIGNO_STOKES = +1.0              # sign convention for Q (Komarov et al. 2016, Eq. 19)
NZ_FINE = 300                    # refined energy grid used only to integrate the emission
NMU_FINE = 160                   # refined mu grid used only to integrate the emission


# Models
# C = Coulomb collisions, E = return-current electric field, B = magnetic convergence
MODELS = ["C", "CE", "CB", "CEB"]
GAMMAS = [3, 7]                  # spectral indices of the injected beam
