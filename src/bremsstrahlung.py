"""
Polarized electron-ion bremsstrahlung from the electron distribution f(z, mu) at one depth.

Cross sections: relativistic, polarized Gluckstern-Hull form, as written in
Komarov et al. (2016), Appendix B. For each electron energy E, photon energy eps and
angle psi between the electron velocity and the photon direction, the kernel returns
sigma_par and sigma_perp (photon polarization parallel / perpendicular to the emission plane).

Stokes parameters, Eq. (6) of the paper, integrating over electron energy, pitch angle
and gyrophase phi:

    I = sum  w (sigma_perp + sigma_par)
    Q = sum  w (sigma_perp - sigma_par) cos(2 chi)
    U = sum  w (sigma_perp - sigma_par) sin(2 chi)

where chi is the angle between the emission plane and the reference plane (B, line of sight),
and w = v(E) * sqrt(z) * f * dE dmu dphi (sqrt(z) f is the number of electrons per unit energy
in the solver variables).

Q reference: perpendicular to the plane containing B and the line of sight. Negative Q/I means
polarization parallel to the sky projection of B. By gyrophase symmetry U = 0 to numerical
precision, so Q/I is the full linear polarization degree.

No clipping or smoothing is applied: the result is the raw integral.

NOTE: E0_keV, NPHI_BREMS and SIGNO_STOKES are read as globals inside numba functions with
cache=True. If you change them in config.py, delete the __pycache__ folders so numba recompiles.
"""

import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator

import config as C

MEC2_KEV = C.MEC2_KEV
R0_CM = C.R0_CM
ALPHA_FINE = C.ALPHA_FINE
Z_ION = C.Z_ION
E0_keV = C.E0_keV

NPHI_BREMS = C.NPHI_BREMS
PHI_BREMS = (np.arange(NPHI_BREMS) + 0.5) * (2.0 * np.pi / NPHI_BREMS)   # midpoint rule in phi
SIGNO_STOKES = C.SIGNO_STOKES


# Geometry
@njit(cache=True)
def geometria(mu0, phi0, theta):
    """
    Angles between one electron and the observed photon.

    Electron direction:  p = (sin(theta0) cos(phi0), sin(theta0) sin(phi0), mu0)   (B along z)
    Photon direction:    k = (sin(theta), 0, cos(theta))                           (viewing angle theta to B)

    Returns
    -------
    cospsi_k : cos(psi) = p . k, clipped to [-0.999, 0.999] only for evaluating the cross section
               (avoids psi = 0 or pi exactly).
    cos2chi, sin2chi : orientation of the emission plane relative to the (B, line of sight) plane.
    """
    sin0 = np.sqrt(max(1.0 - mu0 * mu0, 0.0))
    sinT = np.sin(theta)
    cosT = np.cos(theta)

    cospsi = cosT * mu0 + sinT * sin0 * np.cos(phi0)
    sinpsi = np.sqrt(max(1.0 - cospsi * cospsi, 1.0e-12))

    coschi = (sinT * mu0 - cosT * sin0 * np.cos(phi0)) / sinpsi
    sinchi = -sin0 * np.sin(phi0) / sinpsi

    coschi = min(max(coschi, -1.0), 1.0)
    sinchi = min(max(sinchi, -1.0), 1.0)

    cospsi_k = min(max(cospsi, -0.999), 0.999)

    cos2chi = 2.0 * coschi * coschi - 1.0
    sin2chi = 2.0 * sinchi * coschi
    return cospsi_k, cos2chi, sin2chi


# GLUCKSTERN-HULL Cross Sections (Komarov et al. 2016, App. B)

@njit(cache=True)
def secciones_gh(eps_keV, E_loc_keV, cospsi):
    """
    Polarized bremsstrahlung cross sections, Komarov et al. (2016), Appendix B.

    Parameters
    ----------
    eps_keV : photon energy [keV]
    E_loc_keV : electron kinetic energy [keV]
    cospsi : cosine of the angle between electron velocity and photon direction

    Returns
    -------
    (sigma_par, sigma_perp), unclipped. Returns (0, 0) if emission is impossible
    (eps >= E) or the inputs are degenerate.

    Notation: k = eps / mc^2, G0 and G1 initial and final Lorentz factors,
    p0 and p1 initial and final momenta in units of mc.
    """
    if E_loc_keV <= eps_keV:
        return 0.0, 0.0

    k = eps_keV / MEC2_KEV
    G0 = 1.0 + E_loc_keV / MEC2_KEV
    G1 = G0 - k

    if k <= 0.0 or G1 <= 1.0:
        return 0.0, 0.0

    p0sq = max(G0 * G0 - 1.0, 1e-300)
    p1sq = max(G1 * G1 - 1.0, 1e-300)
    p0 = np.sqrt(p0sq)
    p1 = np.sqrt(p1sq)

    sin2psi = max(1.0 - cospsi * cospsi, 1e-14)

    Delta = G0 - p0 * cospsi
    if abs(Delta) < 1e-6:
        Delta = 1e-6 if Delta >= 0.0 else -1e-6

    T2 = max(p0sq + k * k - 2.0 * p0 * k * cospsi, 1e-300)
    T = np.sqrt(T2)

    # L uses a SINGLE ln. Eq. (B16) of the paper shows "2 ln", which is one factor too many:
    # checked by requiring sigma >= 0 and sigma_perp = sigma_par when psi -> 0.
    aL = G0 * G1 + p0 * p1 - 1.0
    bL = G0 * G1 - p0 * p1 - 1.0
    if abs(aL) < 1e-300:
        aL = 1e-300
    if abs(bL) < 1e-300:
        bL = 1e-300
    L = np.log(abs(aL / bL))

    # epsilon_T = ln((T + p1) / (T - p1))
    aT = T + p1
    bT = T - p1
    if abs(aT) < 1e-300:
        aT = 1e-300
    if abs(bT) < 1e-300:
        bT = 1e-300
    epsT = np.log(abs(aT / bT))

    # epsilon = ln((G1 + p1) / (G1 - p1))
    ae = G1 + p1
    be = G1 - p1
    if abs(ae) < 1e-300:
        ae = 1e-300
    if abs(be) < 1e-300:
        be = 1e-300
    epslog = np.log(abs(ae / be))

    D2 = Delta * Delta
    D4 = D2 * D2

    pref = (Z_ION * Z_ION) * R0_CM * R0_CM * ALPHA_FINE / (8.0 * np.pi)
    pref *= p1 / max(p0, 1e-300)
    pref *= 1.0 / max(k, 1e-300)

    # ---------- parallel (B2, B5, B6, B8, B9, B11) ----------
    Bpar = (
        8.0 * sin2psi * (2.0 * G0 * G0 + 1.0) / (p0sq * D4)
        - (5.0 * G0 * G0 + 2.0 * G0 * G1 + 5.0) / (p0sq * D2)
        - (p0sq - k * k) / (T2 * D2)
        + 2.0 * (G0 + G1) / (p0sq * Delta)
    )

    # Cpar: the "+2" sits outside the parenthesis of 9G0^2 - 4G0G1 + G1^2 (B8)
    Cpar = (
        4.0 * G0 * sin2psi * (3.0 * k - p0sq * G1) / (p0sq * D4)
        + (2.0 * G0 * G0 * (G0 * G0 + G1 * G1)
           - (9.0 * G0 * G0 - 4.0 * G0 * G1 + G1 * G1)
           + 2.0) / (p0sq * D2)
        + k * (G0 * G0 + G0 * G1) / (p0sq * Delta)
    )

    Dpar = (
        4.0 / D2
        - 7.0 * k / Delta
        - k * (p0sq - k * k) / (T2 * Delta)
        - 4.0
    )

    # E term, common to both polarizations (B12)
    Ecommon = (
        2.0 * L / max(p0 * p1, 1e-300) * (2.0 * G0 * G0 - G0 * G1 - 1.0 - k / Delta)
        - 4.0 * epsT * (Delta - G1) * (Delta - G1) / max(p1 * T, 1e-300)
        - 2.0 * epslog * (Delta - G1) / max(p1, 1e-300)
    )

    sig_par = pref * (
        Bpar
        + L / max(p0 * p1, 1e-300) * Cpar
        + epsT / max(p1 * T, 1e-300) * Dpar
        - 4.0 * epslog / max(p1 * Delta, 1e-300)
        + Ecommon / (p0sq * sin2psi)
    )

    # ---------- perpendicular (B1, B4, B7, B10) ----------
    Bperp = (
        -(5.0 * G0 * G0 + 2.0 * G0 * G1 + 1.0) / (p0sq * D2)
        - (p0sq - k * k) / (T2 * D2)
        - 2.0 * k / (p0sq * Delta)
    )

    Cperp = (
        2.0 * G0 * G0 * (G0 * G0 + G1 * G1) / (p0sq * D2)
        - (5.0 * G0 * G0 - 2.0 * G0 * G1 + G1 * G1) / (p0sq * D2)
        + k * (G0 * G0 + G0 * G1 - 2.0) / (p0sq * Delta)
    )

    Dperp = (
        k / Delta
        - k * (p0sq - k * k) / (T2 * Delta)
        + 4.0
    )

    sig_perp = pref * (
        Bperp
        + L / max(p0 * p1, 1e-300) * Cperp
        + epsT / max(p1 * T, 1e-300) * Dperp
        - Ecommon / (p0sq * sin2psi)
    )

    return sig_par, sig_perp


@njit(cache=True)
def beta_E(E_loc_keV):
    """Relativistic speed v/c of an electron of kinetic energy E [keV]; the v(E) of Eq. (6)."""
    g = 1.0 + E_loc_keV / MEC2_KEV
    return np.sqrt(max(1.0 - 1.0 / (g * g), 0.0))


# Stokes parameters of the HXR emission from the distribution at one depth
@njit(cache=True)
def stokes_hxr(f2, fotones_keV, theta_deg, E_keV, mu, parte):
    """
    Stokes I, Q, U of the HXR emission from the distribution at one depth, Eq. (6).

    Parameters
    ----------
    f2 : ndarray (nE, nmu)
        Distribution f(z, mu) at a single xi (usually refined with refinar_f2).
    fotones_keV : ndarray
        Photon energies [keV].
    theta_deg : float
        Viewing angle between B and the line of sight [deg].
    E_keV, mu : ndarray
        Electron energy [keV] and pitch-angle grids of f2.
    parte : int
        0 total, 1 only mu > 0 (downward electrons), -1 only mu < 0 (upward electrons).

    Returns
    -------
    I, Q, U : ndarray, one value per photon energy (code units)
    n_malos : int
        Number of cross-section evaluations that were negative or not finite (should be 0).
    """
    theta = theta_deg * np.pi / 180.0
    nE = fotones_keV.size
    I = np.zeros(nE)
    Q = np.zeros(nE)
    U = np.zeros(nE)
    dphi = 2.0 * np.pi / NPHI_BREMS
    n_malos = 0

    for ie in range(nE):
        eps = fotones_keV[ie]
        accI = 0.0
        accQ = 0.0
        accU = 0.0

        for j in range(E_keV.size - 1):
            E1 = E_keV[j]
            E2 = E_keV[j + 1]
            Ec = 0.5 * (E1 + E2)
            if Ec <= eps:
                continue
            dE = E2 - E1
            # weight v(E) * sqrt(z): number of electrons per unit energy times their speed
            peso_E = beta_E(Ec) * np.sqrt(Ec / E0_keV)

            for k in range(mu.size - 1):
                mu0 = 0.5 * (mu[k] + mu[k + 1])
                if parte == 1 and mu0 <= 0.0:
                    continue
                if parte == -1 and mu0 >= 0.0:
                    continue

                dm = mu[k + 1] - mu[k]
                fc = 0.25 * (f2[j, k] + f2[j + 1, k] + f2[j, k + 1] + f2[j + 1, k + 1])
                if fc <= 0.0:
                    continue

                for iphi in range(NPHI_BREMS):
                    cospsi, cos2chi, sin2chi = geometria(mu0, PHI_BREMS[iphi], theta)
                    sig_par, sig_perp = secciones_gh(eps, Ec, cospsi)

                    if (not np.isfinite(sig_par)) or (not np.isfinite(sig_perp)):
                        n_malos += 1
                        continue
                    if sig_par < 0.0 or sig_perp < 0.0:
                        n_malos += 1

                    w = fc * peso_E * dE * dm * dphi
                    s0 = sig_perp + sig_par
                    s1 = sig_perp - sig_par

                    accI += w * s0
                    accQ += SIGNO_STOKES * w * s1 * cos2chi
                    accU += SIGNO_STOKES * w * s1 * sin2chi

        I[ie] = accI
        Q[ie] = accQ
        U[ie] = accU

    return I, Q, U, n_malos


# Refined Grid for the emission integral

def refinar_f2(f2, z, mu, nz_f=C.NZ_FINE, nmu_f=C.NMU_FINE):
    """
    Interpolate log(f) in (log z, mu) onto a much finer grid, only to integrate the emission.

    With the finer grid, the edge of each electron-energy cell passes close to the photon
    energy eps without large jumps in the integrand.

    Returns (f2_fine, E_fine_keV, mu_fine).
    """
    z_f = np.logspace(np.log10(z[0]), np.log10(z[-1]), nz_f)
    mu_f = np.linspace(mu[0], mu[-1], nmu_f)
    interp = RegularGridInterpolator((np.log(z), mu), np.log(np.maximum(f2, 1e-300)),
                                     method="linear")
    ZZ, MM = np.meshgrid(np.log(z_f), mu_f, indexing="ij")
    puntos = np.stack([ZZ.ravel(), MM.ravel()], axis=-1)
    f2_f = np.ascontiguousarray(np.exp(interp(puntos).reshape(nz_f, nmu_f)))
    return f2_f, z_f * E0_keV, mu_f


def single_electron_polarization(E_keV_el, ratio, cospsi):
    """
    Polarization degree of a single electron, P = SIGNO_STOKES (sigma_perp - sigma_par) / (sigma_perp + sigma_par),
    for photon energies eps = ratio * E. Used to reproduce Komarov et al. (2016) Fig. 1.
    """
    P = np.empty_like(ratio)
    for i, r in enumerate(ratio):
        sig_par, sig_perp = secciones_gh(r * E_keV_el, E_keV_el, cospsi)
        P[i] = SIGNO_STOKES * (sig_perp - sig_par) / (sig_perp + sig_par)
    return P
