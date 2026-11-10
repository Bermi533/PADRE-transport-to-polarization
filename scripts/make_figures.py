"""
Step 3. Make the figures from results already saved by run_solver.py and run_pipeline.py.

Usage
-----
    python scripts/make_figures.py                  # all figures
    python scripts/make_figures.py --only fig4 heatmaps
    python scripts/make_figures.py --list           # show the available names


Fast (only read f or the CSVs):   fig4, fig1_v6, heatmaps, komarov,
                                  pol_depth, spectrum_depth, spectrum_rescaled
Slow (recompute Stokes from f):   angle, intensity, fig5

Output folders:
    results/fokker_planck/   fig4, fig1_v6, heatmaps
    results/polarization/    komarov, pol_depth, spectrum_depth, spectrum_rescaled, angle, intensity, fig5
"""

import argparse
import json

import numpy as np


# The model functions live in ../src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C
import plotting as pl
from grid import s, z, mu
from data_io import load_f, load_results, FP_DIR, POL_DIR, grid_tag
from polarization import (
    angle_scan, fig5_data, refine_all_layers, integrated_intensity,
)

# Base case used by the polarization figures
GAMMA = 3
MODEL = "CEB"
DEPTHS = [1.0e18, 1.0e19, 1.0e20, 1.0e21]


def depth_csvs():
    """{legend label: csv path} for the per-depth CSVs written by run_pipeline.py."""
    tag = grid_tag()
    out = {}
    for d in DEPTHS:
        exp = int(np.log10(d))
        key = f"{d:.0e}".replace("+", "")
        out[rf"$\xi = 10^{{{exp}}}$ cm$^{{-2}}$"] = POL_DIR / f"hxr_xi_{key}_{tag}.csv"
    return out


def integrated_csv():
    return POL_DIR / f"hxr_integrado_{grid_tag()}.csv"


# One function per fig.
def fig_fig4():
    pl.make_fig4(load_results([3, 7], C.MODELS), FP_DIR / "fig4_beam_density.png")


def fig_fig1_v6():
    pl.make_fig1_v6(load_results([3], C.MODELS), FP_DIR / "fig1_v6_depths_en.png")


def fig_heatmaps():
    f = load_f(GAMMA, MODEL)
    Ns = f.shape[0]
    s_indices = [int(0.2 * Ns), int(0.5 * Ns), int(0.8 * Ns)]
    pl.plot_E_mu_heatmaps_subplots(f, s, z, mu, s_indices,
                                   outname=FP_DIR / f"heatmaps_E_mu_gamma{GAMMA}_{MODEL}.png",
                                   title=rf"$f(E,\mu)$, $\gamma={GAMMA}$, {MODEL}")


def fig_komarov():
    pl.plot_komarov_fig1(POL_DIR / "komarov_fig1_check.png")


def fig_pol_depth():
    pl.plot_polarization_by_depth(depth_csvs(), integrated_csv(), POL_DIR / "polarization_by_depth.png")


def fig_spectrum_depth():
    pl.plot_spectrum_by_depth(depth_csvs(), integrated_csv(), POL_DIR / "spectrum_by_depth.png")


def fig_spectrum_rescaled():
    # Normalization factors saved by run_pipeline.py (replaces the values pasted by hand)
    with open(POL_DIR / f"normalization_factors_{grid_tag()}.json") as fj:
        factors = json.load(fj)["factors"]
    files = depth_csvs()
    I0 = {}
    for label, d in zip(files, DEPTHS):
        I0[label] = factors[f"{d:.0e}".replace("+", "")]
    pl.plot_spectrum_rescaled(files, I0, mode="brightness", outname=POL_DIR / "spectrum_by_depth_rescaled.png")
    pl.plot_spectrum_rescaled(files, I0, mode="E2", outname=POL_DIR / "spectrum_by_depth_E2.png")


def fig_angle():
    photon = np.array([15.0, 60.0, 200.0])
    thetas = np.arange(0.0, 181.0, 15.0)
    QI_int, QI_single, maxU = angle_scan(load_f(GAMMA, MODEL), photon, thetas)
    print("Q/I (integrated) at theta = 0 deg  :", QI_int[0])
    print("Q/I (integrated) at theta = 180 deg:", QI_int[-1])
    print("max |U|/I over all theta (linearity check):", maxU)
    pl.plot_polarization_vs_angle(thetas, photon, QI_int, QI_single, POL_DIR / "polarization_vs_angle.png")


def fig_intensity():
    photon = np.logspace(np.log10(15.0), np.log10(500.0), 25)
    I_a = {}
    for gam in [3, 7]:
        for model in ["CE", "CEB"]:
            I_a[(float(gam), model)] = integrated_intensity(load_f(gam, model), photon, [90.0])[0]
            print("done (a): gamma =", gam, " model =", model)
    thetas_b = [0.0, 45.0, 90.0, 135.0, 180.0]
    I_b = integrated_intensity(load_f(GAMMA, MODEL), photon, thetas_b)
    print("done (b)")
    pl.plot_intensity_panels(photon, I_a, I_b, thetas_b, GAMMA, MODEL, outname=POL_DIR / "intensity_panels.png")


def fig_fig5():
    # Rows as in the original Fig. 5 script: injected beam (xi = 0), then each model at xi = 1e18,
    # and C+E+B deep at xi = 1e21.
    rows = [
        ("xi=0", "boundary", 0.0),
        ("C\nxi=1e18", "C", 1.0e18),
        ("C+E\nxi=1e18", "CE", 1.0e18),
        ("C+B\nxi=1e18", "CB", 1.0e18),
        ("C+E+B\nxi=1e18", "CEB", 1.0e18),
        ("C+E+B\nxi=1e21", "CEB", 1.0e21),
    ]
    photon = np.logspace(np.log10(15.0), np.log10(700.0), 70)
    results = load_results([3, 7], C.MODELS)
    data = fig5_data(results, photon, rows)
    pl.make_fig5(data, rows, photon, outname=POL_DIR / "fig5_hxr_spectra_polarization.png")


FIGURES = {
    "fig4": fig_fig4,
    "fig1_v6": fig_fig1_v6,
    "heatmaps": fig_heatmaps,
    "komarov": fig_komarov,
    "pol_depth": fig_pol_depth,
    "spectrum_depth": fig_spectrum_depth,
    "spectrum_rescaled": fig_spectrum_rescaled,
    "angle": fig_angle,
    "intensity": fig_intensity,
    "fig5": fig_fig5,
}


def main():
    parser = argparse.ArgumentParser(description="Make the project figures from saved results.")
    parser.add_argument("--only", nargs="*", choices=list(FIGURES), help="figures to make (default: all)")
    parser.add_argument("--list", action="store_true", help="list the figure names and exit")
    args = parser.parse_args()

    if args.list:
        print("\n".join(FIGURES))
        return

    FP_DIR.mkdir(parents=True, exist_ok=True)
    POL_DIR.mkdir(parents=True, exist_ok=True)
    for name in (args.only or FIGURES):
        print(f"\n--- {name} ---")
        try:
            FIGURES[name]()
        except FileNotFoundError as err:
            print(f"Skipped {name}: {err}")


if __name__ == "__main__":
    main()
