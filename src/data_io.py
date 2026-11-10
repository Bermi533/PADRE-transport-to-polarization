"""
Saving and loading solver output, and writing the 3-column CSVs.

The solver is the slow step, so its output f(s, z, mu) is saved once to
results/fokker_planck/ and every later step (pipeline, figures) reads it from there.
"""

from pathlib import Path
import numpy as np

import config as C

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

# Every output goes to the folder of the step it belongs to:
FP_DIR = RESULTS_DIR / "fokker_planck"    # f (.npz, not in git), summary/profiles CSVs, Fig. 4, Fig. 1 v6, heatmaps
POL_DIR = RESULTS_DIR / "polarization"    # HXR CSVs (energy, flux, Q/I), normalization factors, polarization and spectra figures

CSV_HEADER = "energia_keV,flujo_normalizado,polarizacion_QI"


def grid_tag():
    """Grid size as a string, e.g. '150x90x120', used in file names."""
    return f"{C.NS}x{C.NZ}x{C.NMU}"


def f_path(gamma, model):
    """Path of the saved distribution for one (gamma, model)."""
    return FP_DIR / f"f_gamma{gamma:g}_{model}_{grid_tag()}.npz"


def save_f(gamma, model, f, eps, integ, diag, runtime):
    """Save the steady-state distribution and its diagnostics to results/fokker_planck/."""
    FP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f_path(gamma, model),
        f=f, eps=eps, integ=integ, diag=np.array(diag), runtime=runtime,
    )


def load_f(gamma, model):
    """Load f(s, z, mu) for one (gamma, model). Raises FileNotFoundError if the solver has not been run."""
    p = f_path(gamma, model)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run first:  python scripts/run_solver.py --gamma {gamma:g} --model {model}"
        )
    return np.load(p)["f"]


def load_results(gammas, models):
    """Load every available (gamma, model) into a dict {(gamma, model): f}."""
    return {(g, m): load_f(g, m) for g in gammas for m in models}


def write_csv(path, photon_keV, flux_norm, qi):
    """Write the 3-column CSV: photon energy [keV], normalized flux, Q/I."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([photon_keV, flux_norm, qi]),
               delimiter=",", header=CSV_HEADER, comments="")


def read_csv(path):
    """Read a 3-column CSV. Returns (E, flux, P)."""
    return np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
