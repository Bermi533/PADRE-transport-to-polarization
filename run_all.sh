#!/usr/bin/env bash
# Runs the whole project in order. Each step reads what the previous one saved.
#
#   bash run_all.sh            # everything
#   bash run_all.sh figures    # only the figures (solver and pipeline already done)
#
set -e
cd "$(dirname "$0")"

if [ "$1" != "figures" ]; then
    echo "=== 1/3 Fokker-Planck solver (slow) ==="
    python scripts/run_solver.py --skip-existing

    echo "=== 2/3 HXR polarization pipeline (CSV) ==="
    python scripts/run_pipeline.py --gamma 3 --model CEB --theta 90
fi

echo "=== 3/3 Figures ==="
python scripts/make_figures.py
