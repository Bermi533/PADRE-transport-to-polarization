"""
Step 1. Solve the Fokker-Planck equation and save f(s, z, mu) to results/fokker_planck/.

This is the slow step. Run it once; the pipeline and the figures read the saved files.

Usage
-----
    python scripts/run_solver.py                         # all gammas (3, 7) and models (C, CE, CB, CEB)
    python scripts/run_solver.py --gamma 3 --model CEB   # a single case
    python scripts/run_solver.py --skip-existing         # do not recompute cases already saved

It also writes results/fokker_planck/summary_<grid>.csv with the diagnostics of each case and
results/fokker_planck/profiles_<grid>.csv with the beam density profiles of Eq. (22).
"""

import argparse
import csv
import time


# The model functions live in ../src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C
import grid
from fokker_planck import run_model, local_ratio, density_mu_sign_eq22
from data_io import save_f, f_path, load_f, FP_DIR, grid_tag


def main():
    parser = argparse.ArgumentParser(description="Run the Fokker-Planck solver and save f.")
    parser.add_argument("--gamma", type=float, nargs="*", default=C.GAMMAS)
    parser.add_argument("--model", nargs="*", default=C.MODELS, choices=C.MODELS)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    print("Normalized Solver")
    grid.describe()

    summary_rows = []
    for gamma in args.gamma:
        gamma = int(gamma) if float(gamma).is_integer() else gamma
        for model in args.model:
            if args.skip_existing and f_path(gamma, model).exists():
                print(f"Already saved, skipping: {f_path(gamma, model).name}")
                continue

            print("\n" + "-" * 72)
            print(f"Running gamma={gamma}, model={model}")
            t0 = time.time()
            f, eps, integ, diag = run_model(gamma, model)
            runtime = time.time() - t0

            save_f(gamma, model, f, eps, integ, diag, runtime)
            print(f"Saved: {f_path(gamma, model)}")

            ratio = local_ratio(f, grid.z, grid.mu, grid.A_cross, grid.IS_DIAG)
            summary_rows.append([
                gamma, model, ratio, diag[2], integ[grid.IS_DIAG],
                diag[0], diag[1], diag[3], diag[4],
                diag[5], diag[6], diag[7], runtime,
            ])
            print(
                f"FINAL gamma={gamma}, model={model}: "
                f"up/down={ratio:.4e}, eps_diag={diag[2]:.3e}, "
                f"zE/zC={diag[5]:.3e}, muE/muC={diag[6]:.3e}, "
                f"muB/muC={diag[7]:.3e}, runtime={runtime:.1f}s"
            )

    FP_DIR.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        summary_path = FP_DIR / f"summary_{grid_tag()}.csv"
        with open(summary_path, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "gamma", "model", "up_down_diag", "eps_diag", "int_diag",
                "n_diag", "sigma_diag", "eps_over_n", "alpha_diag",
                "zE_over_zC", "muE_over_muC", "muB_over_muC", "runtime_s",
            ])
            writer.writerows(summary_rows)
        print(f"Saved: {summary_path}")

    # Beam density profiles (Eq. 22) for every case that exists on disk
    profiles_path = FP_DIR / f"profiles_{grid_tag()}.csv"
    with open(profiles_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["gamma", "model", "xi", "down", "up", "total", "A_cross"])
        for gamma in C.GAMMAS:
            for model in C.MODELS:
                if not f_path(gamma, model).exists():
                    continue
                down, up = density_mu_sign_eq22(load_f(gamma, model), grid.z, grid.mu, grid.A_cross)
                total = down + up
                for i in range(len(grid.xi)):
                    writer.writerow([gamma, model, grid.xi[i], down[i], up[i], total[i], grid.A_cross[i]])
    print(f"Saved: {profiles_path}")


if __name__ == "__main__":
    main()
