#!/usr/bin/env bash
# Regenerate models/dubins_car/dynamics_1000regions.nc in a pinned, throwaway Python environment.
# Run from the Examples directory. Requires git and uv.
set -euo pipefail

NBF_COMMIT=9d1c29f556aa58e49788f4e11df6caa311f1bc16
BOUND_PROPAGATION_COMMIT=e08b2497d97e906a7e593b2aca99ac1dfb00ca71

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${1:-models/dubins_car/dynamics_1000regions.nc}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

git clone -q https://github.com/DAI-Lab-HERALD/neural-barrier-functions.git "$WORKDIR/neural-barrier-functions"
git -C "$WORKDIR/neural-barrier-functions" checkout -q "$NBF_COMMIT"

uv venv -q --python 3.11 "$WORKDIR/venv"
VIRTUAL_ENV="$WORKDIR/venv" uv pip install -q \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    "torch==2.3.1+cpu" "numpy<2" scipy matplotlib seaborn tqdm xarray netCDF4 \
    "bound_propagation @ git+https://github.com/Zinoex/bound_propagation.git@$BOUND_PROPAGATION_COMMIT"

"$WORKDIR/venv/bin/python" -I "$SCRIPT_DIR/generate_dubins_car_model.py" "$WORKDIR/neural-barrier-functions" "$OUTPUT"
