"""
grid.py
=======

Builds, from config.py, everything that depends only on position and not on the
electron distribution:

    s, z, mu          the solver grid (dimensionless)
    xi, E_keV         the same grid in physical units (cm^-2, keV)
    n_dimless         ambient density n_p / n0 along the loop
    T_arr             temperature [K] along the loop
    sigma_dimless     conductivity (Eq. 13), times SIGMA_FACTOR
    alpha_B           d ln B / d s  (magnetic convergence)
    A_cross           B0 / B(s), the flux-tube cross-section factor used in Eq. 22

Other modules import these arrays from here, so there is a single grid per run.
"""

import numpy as np

import config as C

# Column depth scale [cm^-2] corresponding to s = 1.0 (dimensionless)
E0_erg = C.E0_keV * C.erg_per_keV

# s = xi * pi e^4 lnLambda / E0^2  =  xi / xi0
xi0 = E0_erg**2 / (np.pi * C.e_esu**4 * C.lnLambda)
smin = C.xi_min / xi0
smax = C.xi_max / xi0


# Grid
s = np.logspace(np.log10(smin), np.log10(smax), C.NS)
z = np.logspace(np.log10(C.zmin), np.log10(C.zmax), C.NZ)
mu = np.linspace(-1.0, 1.0, C.NMU)

xi = s * xi0                  # column depth [cm^-2]
E_keV = z * C.E0_keV          # electron energy [keV]

ds_log = np.diff(np.log(s))   # spacing in ln s
dz_log = np.diff(np.log(z))   # spacing in ln z
dmu = mu[1] - mu[0]

IS_DIAG = int(np.argmin(np.abs(xi - C.XI_DIAG)))   # index of the diagnostic depth


# approximate the chromosphere-corona transition as a tanh function
# Temperature drops and density rises across a tanh transition at XI_TR.
T_arr = C.T_chr + (C.T_cor - C.T_chr) * 0.5 * (
    1.0 - np.tanh(np.log10(xi / C.XI_TR) / C.W_TR)
)

n_dimless = C.n_cor + (C.n_chr - C.n_cor) * 0.5 * (
    1.0 + np.tanh(np.log10(xi / C.XI_TR) / C.W_TR)
)

# Conductivity, Eq. (13): sigma = C n0 (kB T)^(3/2) / sqrt(me F0)
sigma_base = (
    (1.97 * np.sqrt(2.0 * np.pi) / 3.0)
    * C.n0
    * (C.k_B * T_arr)**1.5
    / np.sqrt(C.me_g * C.F0)
)
sigma_dimless = C.SIGMA_FACTOR * sigma_base


# Magnetic field
# alpha_B = d ln B / d s = xi0 * d ln B / d xi, constant above XI_C_MAG and zero below.
alpha_B = np.zeros_like(s)
alpha_val = xi0 * np.log(C.B_RATIO) / C.XI_C_MAG
alpha_B[xi < C.XI_C_MAG] = alpha_val

# A(s) = B0 / B(s), used in Eq. (22).
# ln(B/B0) = integral of alpha_B ds (trapezoidal rule).
lnB_over_B0 = np.zeros_like(s)
for i in range(1, len(s)):
    ds_i = s[i] - s[i - 1]
    lnB_over_B0[i] = lnB_over_B0[i - 1] + 0.5 * (alpha_B[i] + alpha_B[i - 1]) * ds_i
A_cross = np.exp(-lnB_over_B0)


# Layer thickness
def layer_thickness():
    """
    Thickness d(xi) [cm^-2] of each solver layer, used to integrate over depth (Eq. 7).

    Uses the real differences in xi (central differences inside, one-sided at the edges),
    not the layer index. The top layer also takes the column above the grid, xi[0].
    """
    dxi = np.empty(C.NS)
    dxi[1:-1] = 0.5 * (xi[2:] - xi[:-2])
    dxi[0] = 0.5 * (xi[1] - xi[0]) + xi[0]
    dxi[-1] = 0.5 * (xi[-1] - xi[-2])
    return dxi


def describe():
    """Print the grid and the atmosphere at the diagnostic depth."""
    print(f"Grid: NS={C.NS}, NZ={C.NZ}, NMU={C.NMU}")
    print(f"xi0={xi0:.3e}, smin={smin:.3e}, smax={smax:.3e}")
    print(f"xi_diag={xi[IS_DIAG]:.3e}, s_diag={s[IS_DIAG]:.3e}")
    print(f"n_diag={n_dimless[IS_DIAG]:.3e}, T_diag={T_arr[IS_DIAG]:.3e} K")
    print(f"sigma_diag={sigma_dimless[IS_DIAG]:.3e}, SIGMA_FACTOR={C.SIGMA_FACTOR:.1e}")
    print(f"alpha_diag={alpha_B[IS_DIAG]:.3e}, B_RATIO={C.B_RATIO}, XI_C_MAG={C.XI_C_MAG:.2e}")
    print(f"A_cross_diag={A_cross[IS_DIAG]:.3e}")
