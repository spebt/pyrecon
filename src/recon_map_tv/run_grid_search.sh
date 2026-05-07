#!/usr/bin/env bash
# =============================================================================
# run_grid_search.sh — Sequential MAP-TV grid search (local, no SLURM)
#
# Usage:
#   bash run_grid_search.sh [base_config]
#
# What it does:
#   1. Generates the HDF5 file list (once, skipped if it already exists).
#   2. Generates noiseless projections (once, shared).
#   3. Generates one set of Poisson-noisy projections per SCALE_FACTOR (shared).
#   4. Runs MAP-TV reconstruction + view_npz (CNR + image) for every combination
#      of noise condition × beta × other params.
#   5. Copies all final PNGs into <data_dir>/experiments/grid_summary/ with
#      descriptive filenames.
#
# Edit the GRID PARAMETERS block below to change the sweep.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── VIRTUAL ENVIRONMENT ───────────────────────────────────────────────────────
# Set VENV_PATH to your venv directory, or pass it as the second argument.
# Examples:
#   VENV_PATH=~/envs/pyrecon bash run_grid_search.sh
#   bash run_grid_search.sh configs/base_config.yml ~/envs/pyrecon
VENV_PATH="${2:-${VENV_PATH:-}}"

if [ -f "../../../venv/bin/activate" ]; then
    source "../../../venv/bin/activate"
    echo "Activated venv: ../../venv"
else
    echo "[warn] No venv found — using system Python ($(which python3))"
fi

BASE_CONFIG="${1:-configs/base_config.yml}"

# ── GRID PARAMETERS ──────────────────────────────────────────────────────────

BETAS=(0.001 0.01 0.1 1.0 10.0)

N_OUTERS=(100)
N_INNERS=(60)
TAUS=(0.2)
SIGMAS=(0.2)

# One noisy run per entry. Remove entries or set to empty array to skip.
SCALE_FACTORS=(100 1000 10000)

# ── READ BASE PATHS FROM CONFIG ───────────────────────────────────────────────
_cfg() { python3 -c "import yaml; cfg=yaml.safe_load(open('${BASE_CONFIG}')); print($1)"; }

FLIST_PATH=$(_cfg "cfg['paths']['flist_path']")
PROJS_NOISELESS=$(_cfg "cfg['paths']['projs_path']")
PROJS_NOISY_TEMPLATE=$(_cfg "cfg['paths']['projs_noisy_path']")   # _sfXXX inserted before .npy
BASE_OUT=$(python3 -c "import yaml, os; cfg=yaml.safe_load(open('${BASE_CONFIG}')); print(os.path.dirname(cfg['paths']['recon_out_path']))")

SUMMARY_DIR="${BASE_OUT}/experiments/grid_summary"
mkdir -p "$SUMMARY_DIR" logs

# ── PATCHER ───────────────────────────────────────────────────────────────────
# Shared Python script: applies dot-notation overrides, sets per-experiment
# recon_out_path, returns the experiment directory path.
PATCHER=$(mktemp /tmp/maptv_patcher_XXXXXX.py)
trap 'rm -f "$PATCHER"' EXIT

cat > "$PATCHER" << 'PYEOF'
import sys, os, yaml

base_cfg, out_cfg, exp_tag = sys.argv[1], sys.argv[2], sys.argv[3]
overrides = sys.argv[4:]

with open(base_cfg) as f:
    cfg = yaml.safe_load(f)

for kv in overrides:
    key, val = kv.split("=", 1)
    try:    val = int(val)
    except ValueError:
        try: val = float(val)
        except ValueError:
            if   val.lower() in ("true",  "yes"): val = True
            elif val.lower() in ("false", "no"):  val = False
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = val

base_out = os.path.dirname(cfg["paths"]["recon_out_path"])
exp_dir  = os.path.join(base_out, "experiments", exp_tag)
os.makedirs(exp_dir, exist_ok=True)
cfg["paths"]["recon_out_path"] = os.path.join(exp_dir, "recon_map_tv.npz")

with open(out_cfg, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(exp_dir)
PYEOF

# ── HELPERS ───────────────────────────────────────────────────────────────────

# Derive per-scale-factor noisy projections path:
#   e.g. /path/to/projs-noisy.npy → /path/to/projs-noisy_sf1000.npy
noisy_path_for_sf() {
    local sf="$1"
    local base="${PROJS_NOISY_TEMPLATE%.npy}"
    echo "${base}_sf${sf}.npy"
}

# Print a section header
header() { echo ""; echo "══════════════════════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════════════════════"; }
subheader() { echo ""; echo "──────────────────────────────────────────────────────────"; echo "  $*"; echo "──────────────────────────────────────────────────────────"; }

TOTAL=0
FAILED=0

run_one() {
    local noise_label="$1"   # "noiseless" or "noisy_sf1000"
    local projs_file="$2"    # path to .npy projections
    local beta="$3"
    local n_outer="$4"
    local n_inner="$5"
    local tau="$6"
    local sigma="$7"

    local exp_tag="${noise_label}__b${beta}_out${n_outer}_in${n_inner}_t${tau}_s${sigma}"
    local tmp_cfg
    tmp_cfg=$(mktemp /tmp/maptv_exp_XXXXXX.yml)

    subheader "Experiment: ${exp_tag}"

    local exp_dir
    exp_dir=$(python3 "$PATCHER" "$BASE_CONFIG" "$tmp_cfg" "$exp_tag" \
        "paths.projs_path=${projs_file}" \
        "map_tv.beta=${beta}" \
        "map_tv.n_outer=${n_outer}" \
        "map_tv.n_inner=${n_inner}" \
        "map_tv.tau=${tau}" \
        "map_tv.sigma=${sigma}")

    echo "  Output : ${exp_dir}"

    local log="logs/${exp_tag}.log"

    if python3 map_tv_recon.py --config "$tmp_cfg" 2>&1 | tee "$log" \
    && python3 view_npz.py     --config "$tmp_cfg" 2>&1 | tee -a "$log"; then
        # Copy final image to summary directory
        local png
        png=$(ls "${exp_dir}/recon_map_tv_"*.png 2>/dev/null | head -1 || true)
        if [ -n "$png" ]; then
            cp "$png" "${SUMMARY_DIR}/${exp_tag}.png"
            echo "  Image  : ${SUMMARY_DIR}/${exp_tag}.png"
        else
            echo "  [warn] No PNG found in ${exp_dir}"
        fi
        TOTAL=$((TOTAL + 1))
    else
        echo "  [ERROR] Experiment failed — see ${log}"
        FAILED=$((FAILED + 1))
    fi

    rm -f "$tmp_cfg"
}

# ── STEP 1: File list ─────────────────────────────────────────────────────────
header "Setup"

if [ -f "$FLIST_PATH" ]; then
    echo "[1/3] File list exists — skipping."
else
    echo "[1/3] Generating HDF5 file list..."
    python3 generate_flist.py --config "$BASE_CONFIG"
fi

# ── STEP 2: Projections ───────────────────────────────────────────────────────

# 2a. Noiseless
if [ -f "$PROJS_NOISELESS" ]; then
    echo "[2/3] Noiseless projections exist — skipping."
else
    echo "[2/3] Generating noiseless projections..."
    python3 fake_projection.py --config "$BASE_CONFIG"
fi

# 2b. Noisy — one per scale factor
for SF in "${SCALE_FACTORS[@]}"; do
    NOISY_OUT=$(noisy_path_for_sf "$SF")
    if [ -f "$NOISY_OUT" ]; then
        echo "[2/3] Noisy projections (sf=${SF}) exist — skipping."
    else
        echo "[2/3] Generating noisy projections (scale_factor=${SF})..."
        TMP_NOISE=$(mktemp /tmp/maptv_noise_XXXXXX.yml)
        python3 "$PATCHER" "$BASE_CONFIG" "$TMP_NOISE" "_noise_gen_sf${SF}" \
            "noise.scale_factor=${SF}" \
            "paths.projs_noisy_path=${NOISY_OUT}" > /dev/null
        python3 fake_projection_noisy.py --config "$TMP_NOISE"
        rm -f "$TMP_NOISE"
    fi
done

# ── STEP 3: Grid search ───────────────────────────────────────────────────────
header "Grid search — $(( ${#BETAS[@]} * ${#N_OUTERS[@]} * ${#N_INNERS[@]} * ${#TAUS[@]} * ${#SIGMAS[@]} * (1 + ${#SCALE_FACTORS[@]}) )) total experiments"

for BETA in "${BETAS[@]}"; do
for N_OUTER in "${N_OUTERS[@]}"; do
for N_INNER in "${N_INNERS[@]}"; do
for TAU in "${TAUS[@]}"; do
for SIGMA in "${SIGMAS[@]}"; do

    # Noiseless
    run_one "noiseless" "$PROJS_NOISELESS" \
        "$BETA" "$N_OUTER" "$N_INNER" "$TAU" "$SIGMA"

    # Noisy — one experiment per scale factor
    for SF in "${SCALE_FACTORS[@]}"; do
        run_one "noisy_sf${SF}" "$(noisy_path_for_sf "$SF")" \
            "$BETA" "$N_OUTER" "$N_INNER" "$TAU" "$SIGMA"
    done

done; done; done; done; done

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
header "Done"
echo "  Completed : ${TOTAL}"
echo "  Failed    : ${FAILED}"
echo "  Images    : ${SUMMARY_DIR}/"
ls -1 "${SUMMARY_DIR}/"
