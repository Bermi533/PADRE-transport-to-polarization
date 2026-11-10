"""
From the full distribution f(s, z, mu) to observable HXR spectra and polarization:

    f_at_depth            f(z, mu) interpolated in log(xi) at an exact column depth
    stokes_at_depth       I, Q, U at one depth (Eq. 6)
    stokes_per_layer      I, Q, U for every solver layer
    depth_integrated      I, Q, U summed over all layers with weight d(xi) (Eq. 7),
                          which is what SHARP sees since it does not resolve the layers
    angle_scan            Q/I versus viewing angle, integrated and at one depth
    fig5_data             spectra and Q/I by depth, model and angle for the Fig. 5 figure

All intensities are in arbitrary code units: they are only comparable between results
of the same run (same gamma, model, theta and photon energies).
"""

import time
import numpy as np

from grid import z, mu, xi, layer_thickness
from bremsstrahlung import stokes_hxr, refinar_f2


def f_at_depth(f_all, xi_obj):
    """
    Distribution f(z, mu) at column depth xi_obj [cm^-2], linearly interpolated in log(xi)
    between the two neighbouring solver layers.

    Returns (f2, i0, i1, w): the interpolated slice, the two layer indices and the weight of i1.
    Raises ValueError if xi_obj is outside the grid.
    """
    if xi_obj < xi[0] or xi_obj > xi[-1]:
        raise ValueError(f"xi={xi_obj:.1e} is outside the grid [{xi[0]:.2e}, {xi[-1]:.2e}]")
    i1 = int(np.searchsorted(xi, xi_obj))
    i0 = max(i1 - 1, 0)
    if i1 == i0:
        i1 = i0 + 1
    w = (np.log(xi_obj) - np.log(xi[i0])) / (np.log(xi[i1]) - np.log(xi[i0]))
    f2 = np.ascontiguousarray((1.0 - w) * f_all[i0] + w * f_all[i1])
    return f2, i0, i1, w


def stokes_at_depth(f2, photon_keV, theta_deg, part=0):
    """I, Q, U and the bad-cross-section count for one f(z, mu) slice, on the refined grid."""
    f2_f, E_f, mu_f = refinar_f2(f2, z, mu)
    return stokes_hxr(f2_f, photon_keV, theta_deg, E_f, mu_f, part)


def refine_all_layers(f_all):
    """Refine every layer once (the refinement does not depend on theta or photon energy)."""
    return [refinar_f2(np.ascontiguousarray(f_all[i]), z, mu) for i in range(f_all.shape[0])]


def stokes_per_layer(f_all, photon_keV, theta_deg, refined=None, verbose=True):
    """
    I, Q, U for every solver layer.

    Returns (I_layer, Q_layer, U_layer, n_bad), arrays of shape (NS, n_photon).
    Pass `refined` (from refine_all_layers) to avoid refining again.
    """
    ns = f_all.shape[0]
    I_layer = np.zeros((ns, photon_keV.size))
    Q_layer = np.zeros_like(I_layer)
    U_layer = np.zeros_like(I_layer)
    n_bad = 0
    t0 = time.time()
    for i in range(ns):
        if refined is None:
            f2_f, E_f, mu_f = refinar_f2(np.ascontiguousarray(f_all[i]), z, mu)
        else:
            f2_f, E_f, mu_f = refined[i]
        I_i, Q_i, U_i, bad = stokes_hxr(f2_f, photon_keV, theta_deg, E_f, mu_f, 0)
        I_layer[i] = I_i
        Q_layer[i] = Q_i
        U_layer[i] = U_i
        n_bad += bad
        if verbose and i % 20 == 0:
            print(f"layer {i + 1}/{ns}  xi={xi[i]:.2e}  ({time.time() - t0:.0f}s)")
    return I_layer, Q_layer, U_layer, n_bad


def depth_integrated(I_layer, Q_layer, U_layer):
    """
    Sum over all layers with weight d(xi), Eq. (7). Returns (I_tot, Q_tot, U_tot).
    """
    dxi = layer_thickness()
    I_tot = np.sum(I_layer * dxi[:, None], axis=0)
    Q_tot = np.sum(Q_layer * dxi[:, None], axis=0)
    U_tot = np.sum(U_layer * dxi[:, None], axis=0)
    return I_tot, Q_tot, U_tot


def integrated_intensity(f_all, photon_keV, thetas, refined=None):
    """
    Depth-integrated I(eps) for each viewing angle. Returns an array of shape (n_theta, n_photon).
    """
    dxi = layer_thickness()
    if refined is None:
        refined = refine_all_layers(f_all)
    out = np.zeros((len(thetas), photon_keV.size))
    for it, th in enumerate(thetas):
        for i in range(len(refined)):
            f2_f, E_f, mu_f = refined[i]
            I_i, Q_i, U_i, _ = stokes_hxr(f2_f, photon_keV, th, E_f, mu_f, 0)
            out[it] += I_i * dxi[i]
    return out


def angle_scan(f_all, photon_keV, thetas, xi_single=1.0e18, refined=None):
    """
    Q/I versus viewing angle theta.

    Returns
    -------
    QI_int : depth-integrated Q/I, shape (n_theta, n_photon)
    QI_single : Q/I of the solver layer closest to xi_single, same shape
    maxU : largest |U|/I found (linearity check, should be ~0)
    """
    dxi = layer_thickness()
    if refined is None:
        refined = refine_all_layers(f_all)
    i_single = int(np.argmin(np.abs(np.log(xi) - np.log(xi_single))))

    QI_int = np.zeros((len(thetas), photon_keV.size))
    QI_single = np.zeros_like(QI_int)
    maxU = 0.0

    for it, th in enumerate(thetas):
        I_tot = np.zeros(photon_keV.size)
        Q_tot = np.zeros(photon_keV.size)
        U_tot = np.zeros(photon_keV.size)
        for i in range(len(refined)):
            f2_f, E_f, mu_f = refined[i]
            I_i, Q_i, U_i, _ = stokes_hxr(f2_f, photon_keV, th, E_f, mu_f, 0)
            I_tot += I_i * dxi[i]
            Q_tot += Q_i * dxi[i]
            U_tot += U_i * dxi[i]
            if i == i_single:
                QI_single[it] = Q_i / I_i
        QI_int[it] = Q_tot / I_tot
        maxU = max(maxU, np.max(np.abs(U_tot / I_tot)))

    return QI_int, QI_single, maxU


def fig5_data(results, photon_keV, rows, thetas=(0.0, 90.0, 180.0), theta_pol=90.0, gammas=(3, 7)):
    """
    Everything the Fig. 5 figure needs (HXR spectra and polarization across depth,
    like Fig. 5 of Zharkova et al. 2010).

    Parameters
    ----------
    results : dict {(gamma, model): f}, the saved Fokker-Planck solutions
    photon_keV : photon energies [keV]
    rows : list of (row_label, model, xi_target). model = "boundary" uses the injected
           distribution (Eq. 15) instead of the solver, i.e. xi = 0.
    thetas : viewing angles of the intensity columns [deg]
    theta_pol : viewing angle of the polarization column [deg]

    Returns
    -------
    data : dict keyed by (row_index, gamma) with
        "I"[theta][part]  intensity for part = 0 (total), 1 (downward, mu > 0), -1 (upward, mu < 0)
        "P"[part]         Q/I at theta_pol for the same three parts
    """
    from fokker_planck import make_injection_bc

    data = {}
    for r, (row_label, model, xi_target) in enumerate(rows):
        for gamma in gammas:
            if model == "boundary":
                f2 = np.ascontiguousarray(make_injection_bc(gamma))
            else:
                f2, _, _, _ = f_at_depth(results[(gamma, model)], xi_target)

            f2_f, E_f, mu_f = refinar_f2(f2, z, mu)
            entry = {"I": {}, "P": {}}
            for th in set(list(thetas) + [theta_pol]):
                entry["I"][th] = {}
                for part in (0, 1, -1):
                    I, Q, U, _ = stokes_hxr(f2_f, photon_keV, th, E_f, mu_f, part)
                    entry["I"][th][part] = I
                    if th == theta_pol:
                        entry["P"][part] = np.divide(Q, I, out=np.zeros_like(Q), where=I > 0)
            data[(r, gamma)] = entry
            print(f"fig5: row {r} ({row_label.replace(chr(10), ' ')}), gamma={gamma} done")
    return data
