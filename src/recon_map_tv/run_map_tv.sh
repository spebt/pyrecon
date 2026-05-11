#!/bin/bash
#SBATCH --job-name=map_tv_recon
#SBATCH --output=logs/map_tv_%j.out
#SBATCH --error=logs/map_tv_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# =============================================================================
# MAP-TV Full Pipeline Script
#
# Usage:
#   sbatch run_map_tv.sh [config] [section.key=value ...]
#
# Examples:
#   sbatch run_map_tv.sh
#   sbatch run_map_tv.sh configs/base_config.yml map_tv.beta=0.1
#   sbatch run_map_tv.sh configs/base_config.yml map_tv.beta=0.1 map_tv.n_outer=40 map_tv.n_inner=20
#
# Overrides use dot-notation matching the YAML structure:
#   map_tv.beta  map_tv.n_outer  map_tv.n_inner  map_tv.tau  map_tv.sigma
#
# Output is written to:
#   <recon_out_path parent>/experiments/<exp_tag>/recon_map_tv.npz
# flist and projections are shared — re-generated only if missing.
# =============================================================================

set -e

mkdir -p logs

# --- Parse arguments (strip --noisy before positional args) ---
NOISY=false
RAW_ARGS=("$@")
FILTERED_ARGS=()
for arg in "${RAW_ARGS[@]}"; do
    if [ "$arg" = "--noisy" ]; then
        NOISY=true
    else
        FILTERED_ARGS+=("$arg")
    fi
done
CONFIG="${FILTERED_ARGS[0]:-configs/base_config.yml}"
OVERRIDES=("${FILTERED_ARGS[@]:1}")

# --- Build experiment tag ---
if [ ${#OVERRIDES[@]} -eq 0 ]; then
    EXP_TAG="default"
else
    EXP_TAG=""
    for kv in "${OVERRIDES[@]}"; do
        short=$(echo "$kv" | sed 's/^[^.]*\.//' | sed 's/=//')
        EXP_TAG="${EXP_TAG:+${EXP_TAG}_}${short}"
    done
fi

# --- Write patcher to a temp file (avoids /dev/stdin fragility on HPC) ---
PATCHER=$(mktemp /tmp/map_tv_patcher_XXXXXX.py)
TEMP_CONFIG=$(mktemp /tmp/map_tv_config_XXXXXX.yml)
trap "rm -f $PATCHER $TEMP_CONFIG" EXIT

cat > "$PATCHER" << 'PYEOF'
import sys, yaml, os

config_path = sys.argv[1]
out_config  = sys.argv[2]
exp_tag     = sys.argv[3]
overrides   = sys.argv[4:]

with open(config_path) as f:
    cfg = yaml.safe_load(f)

for kv in overrides:
    key, val = kv.split("=", 1)
    try:
        val = int(val)
    except ValueError:
        try:
            val = float(val)
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

with open(out_config, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(exp_dir)
PYEOF

EXP_DIR=$(python3 "$PATCHER" "$CONFIG" "$TEMP_CONFIG" "$EXP_TAG" "${OVERRIDES[@]}")

echo "=========================================="
echo "MAP-TV Pipeline"
echo "  Base config : $CONFIG"
echo "  Overrides   : ${OVERRIDES[*]:-none}"
echo "  Experiment  : $EXP_TAG"
echo "  Output dir  : $EXP_DIR"
echo "=========================================="

# --- Step 1: Generate file list (skip if already exists) ---
FLIST_PATH=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$TEMP_CONFIG')); print(cfg['paths']['flist_path'])")
echo ""
if [ -f "$FLIST_PATH" ]; then
    echo "[1/4] File list exists — skipping generation."
else
    echo "[1/4] Generating HDF5 file list..."
    python generate_flist.py --config "$TEMP_CONFIG"
fi

# --- Step 2: Forward project phantom ---
echo ""
if [ "$NOISY" = true ]; then
    PROJS_CHECK=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$TEMP_CONFIG')); print(cfg['paths']['projs_noisy_path'])")
    if [ -f "$PROJS_CHECK" ]; then
        echo "[2/4] Noisy projections exist — skipping generation."
    else
        echo "[2/4] Forward projecting phantom (Poisson noise)..."
        python fake_projection_noisy.py --config "$TEMP_CONFIG"
    fi
else
    PROJS_CHECK=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$TEMP_CONFIG')); print(cfg['paths']['projs_path'])")
    if [ -f "$PROJS_CHECK" ]; then
        echo "[2/4] Projections exist — skipping forward projection."
    else
        echo "[2/4] Forward projecting phantom (noiseless)..."
        python fake_projection.py --config "$TEMP_CONFIG"
    fi
fi

NOISY_FLAG=""
[ "$NOISY" = true ] && NOISY_FLAG="--noisy"

# --- Step 3: MAP-TV reconstruction ---
echo ""
echo "[3/4] Running MAP-TV reconstruction..."
python map_tv_recon.py --config "$TEMP_CONFIG" $NOISY_FLAG

# --- Step 4: Visualize result ---
echo ""
echo "[4/4] Generating reconstruction plot..."
python view_npz.py --config "$TEMP_CONFIG" $NOISY_FLAG

echo ""
echo "=========================================="
echo "Pipeline complete. Results in:"
echo "  $EXP_DIR"
echo "=========================================="
