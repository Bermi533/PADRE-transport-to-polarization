"""
Step 2. The HXR polarization pipeline 

Reads the saved distribution f for one (gamma, model) and writes 3-column CSVs
(photon energy [keV], normalized flux, Q/I) to results/polarization/:

    hxr_integrado_<grid>.csv        sum over all layers with weight d(xi), Eq. (7): what SHARP sees
    hxr_xi_1e18_<grid>.csv, ...     one CSV per column depth, Eq. (6)

Each flux column is normalized by its value at the lowest photon energy (first row = 1).
The normalization factors are printed and saved to normalization_factors_<grid>.json, so the
absolute (code-unit) flux can be recovered as  flux = flux_normalized * factor.

It also checks that U ~ 0 (so Q/I is the full linear polarization) and |P| <= 1.

Usage
-----
    python scripts/run_pipeline.py                                   # gamma 3, CEB, theta 90
    python scripts/run_pipeline.py --gamma 7 --model CE --theta 60
    python scripts/run_pipeline.py --emin 15 --emax 100 --npts 30    # restrict to the SHARP band
"""

import argparse
import json
import time

import numpy as np


# The model functions live in ../src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C
from grid import xi
from data_io import load_f, write_csv, POL_DIR, grid_tag
from polarization import (
    f_at_depth, stokes_at_depth, stokes_per_layer, depth_integrated,
)


def print_parameters(fname, description, n_bad, U_over_I, P, photon_keV, theta, gamma, model):
    """Print every parameter that defines one CSV, plus the checks."""
    print("=" * 64)
    print(f"File: {fname}")
    print(f"Content: {description}")
    print("=" * 64)
    print("Output")
    print("  columns: photon energy (keV), normalized flux, Q/I")
    print(f"  flux normalized by its value at {photon_keV[0]:.1f} keV (first row = 1)")
    print(f"  photon energy: {photon_keV[0]:.1f} to {photon_keV[-1]:.1f} keV, {photon_keV.size} points (log)")
    print(f"  viewing angle theta (B - line of sight): {theta:.1f} deg")
    print("Injected beam")
    print(f"  gamma = {gamma}   model = {model}   (C=collisions, E=electric field, B=magnetic convergence)")
    print(f"  DELTA_LOW = {C.DELTA_LOW}   DELTA_MU = {C.DELTA_MU}   E0 = {C.E0_keV} keV   F0 = {C.F0:.1e}   n0 = {C.n0:.1e}")
    print(f"  z range: {C.zmin} to {C.zmax}  (E = {C.zmin * C.E0_keV:.2f} to {C.zmax * C.E0_keV:.0f} keV)")
    print("Solver")
    print(f"  grid NS x NZ x NMU = {C.NS} x {C.NZ} x {C.NMU}")
    print(f"  DTAU = {C.DTAU:.1e}   N_ITER = {C.N_ITER}   TOL = {C.TOL:.1e}")
    print(f"  OMEGA_EPS = {C.OMEGA_EPS}   EPS_EVERY = {C.EPS_EVERY}   SIGMA_FACTOR = {C.SIGMA_FACTOR:.1e}")
    print("Atmosphere and magnetic field")
    print(f"  tanh atmosphere: XI_TR = {C.XI_TR:.1e}  W_TR = {C.W_TR}  T_cor = {C.T_cor:.1e} K  T_chr = {C.T_chr:.1e} K")
    print(f"  n_cor = {C.n_cor}  n_chr = {C.n_chr:.0e}  (units of n0)")
    print(f"  magnetic field: B_RATIO = {C.B_RATIO}  XI_C_MAG = {C.XI_C_MAG:.1e} (step)")
    print("Emission")
    print("  cross sections: Gluckstern-Hull, Komarov et al. (2016) App. B, L with a single ln")
    print(f"  energy weight: v(E)*sqrt(z)*f (Eq. 6)   sign of Q: {C.SIGNO_STOKES:+.0f} (Eq. 19)")
    print(f"  phi integration: {C.NPHI_BREMS} points   f refined to {C.NZ_FINE} x {C.NMU_FINE} (energy x mu)")
    print("Checks")
    print(f"  negative or non-finite sigma evaluations: {n_bad}")
    print(f"  max |U|/I = {np.max(np.abs(U_over_I)):.1e}   max |Q/I| = {np.max(np.abs(P)):.3f}")
    print("=" * 64 + "\n")


def main():
    parser = argparse.ArgumentParser(description="HXR polarization pipeline: CSVs of energy, flux, Q/I.")
    parser.add_argument("--gamma", type=float, default=3)
    parser.add_argument("--model", default="CEB", choices=C.MODELS)
    parser.add_argument("--theta", type=float, default=90.0, help="viewing angle [deg]")
    parser.add_argument("--emin", type=float, default=15.0, help="lowest photon energy [keV]")
    parser.add_argument("--emax", type=float, default=500.0, help="highest photon energy [keV]")
    parser.add_argument("--npts", type=int, default=50, help="number of photon energies (log-spaced)")
    parser.add_argument("--depths", type=float, nargs="*", default=[1.0e18, 1.0e19, 1.0e20, 1.0e21],
                        help="column depths [cm^-2], one CSV each")
    parser.add_argument("--no-integrated", action="store_true", help="skip the depth-integrated CSV")
    args = parser.parse_args()

    gamma = int(args.gamma) if args.gamma.is_integer() else args.gamma
    model, theta = args.model, args.theta
    photon_keV = np.logspace(np.log10(args.emin), np.log10(args.emax), args.npts)
    tag = grid_tag()

    f_all = load_f(gamma, model)
    factors = {}

    # ---- 1. Depth-integrated CSV (Eq. 7) ----
    if not args.no_integrated:
        t0 = time.time()
        I_l, Q_l, U_l, n_bad = stokes_per_layer(f_all, photon_keV, theta)
        I_tot, Q_tot, U_tot = depth_integrated(I_l, Q_l, U_l)
        if np.any(I_tot <= 0.0) or not np.all(np.isfinite(I_tot)):
            raise ValueError("The integrated I has values <= 0 or not finite: CSV not written.")
        P_tot = Q_tot / I_tot

        fname = POL_DIR / f"hxr_integrado_{tag}.csv"
        write_csv(fname, photon_keV, I_tot / I_tot[0], P_tot)
        print_parameters(fname.name,
                         f"sum of all layers, xi = {xi[0]:.2e} to {xi[-1]:.2e} cm^-2, weight d(xi) (Eq. 7)",
                         n_bad, U_tot / I_tot, P_tot, photon_keV, theta, gamma, model)
        factors["integrated"] = float(I_tot[0])
        print(f"({time.time() - t0:.0f}s)")

    # ---- 2. One CSV per depth (Eq. 6) ----
    for xi_obj in args.depths:
        try:
            f2, i0, i1, w = f_at_depth(f_all, xi_obj)
        except ValueError as err:
            print(f"WARNING: {err}; skipped.")
            continue

        I_i, Q_i, U_i, n_bad = stokes_at_depth(f2, photon_keV, theta)
        if np.any(I_i <= 0.0) or not np.all(np.isfinite(I_i)):
            print(f"WARNING: xi={xi_obj:.1e} has I <= 0 or not finite; CSV not written.")
            continue

        fname = POL_DIR / (f"hxr_xi_{xi_obj:.0e}".replace("+", "") + f"_{tag}.csv")
        write_csv(fname, photon_keV, I_i / I_i[0], Q_i / I_i)
        print_parameters(fname.name,
                         f"single depth, xi = {xi_obj:.3e} cm^-2 "
                         f"(interpolated between {xi[i0]:.3e} and {xi[i1]:.3e}, weight {w:.2f}) (Eq. 6)",
                         n_bad, U_i / I_i, Q_i / I_i, photon_keV, theta, gamma, model)
        factors[f"{xi_obj:.0e}".replace("+", "")] = float(I_i[0])

    # ---- 3. Normalization factors ----
    print("=" * 64)
    print(f"NORMALIZATION FACTORS (value at {photon_keV[0]:.1f} keV, the one that equals 1 in each CSV)")
    print("To recover the unnormalized flux: flux = flux_normalized * factor")
    print("=" * 64)
    if "integrated" in factors:
        print(f"  integrated:  factor = {factors['integrated']:.4e}   (units: I * cm^-2)")
    print("\n  Per depth (units: I per cm^-2 of column):")
    print("   xi_cm2      factor        light_fraction_per_decade_of_xi")
    for key, fac in factors.items():
        if key == "integrated":
            continue
        xi_val = float(key)
        line = f"  {xi_val:8.1e}   {fac:.4e}"
        if "integrated" in factors:
            frac = fac * xi_val * np.log(10.0) / factors["integrated"]
            line += f"   {frac:.3f}"
        print(line)
    print("\nlight_fraction_per_decade_of_xi: approximate share of the integrated light at that energy")
    print("from one decade of column around that depth (uses xi*ln(10) as the thickness).")
    print("Arbitrary units: only comparable between files of the same run (same gamma, model, theta, energies).")

    meta = {"gamma": gamma, "model": model, "theta_deg": theta, "grid": tag,
            "photon_keV": [float(photon_keV[0]), float(photon_keV[-1]), int(photon_keV.size)],
            "factors": factors}
    json_path = POL_DIR / f"normalization_factors_{tag}.json"
    with open(json_path, "w") as fj:
        json.dump(meta, fj, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
