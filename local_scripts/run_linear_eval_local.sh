#!/usr/bin/env bash
set -euo pipefail

# ── Activate conda env first ─────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate simsiam-mps

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_FILE="${CONFIG_FILE:-"$REPO_ROOT/configs/linear_eval.yaml"}"

_yaml_local() {
    python - "$CONFIG_FILE" "$1" << 'PYEOF'
import sys, os, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
val = cfg.get("local", {}).get(sys.argv[2], "")
print(os.path.expanduser(str(val)) if val else "")
PYEOF
}

EVAL_FROM="${EVAL_FROM:-$(_yaml_local eval_from)}"
DATA_DIR="${DATA_DIR:-$(_yaml_local data_dir)}"
SAVE_DIR="${SAVE_DIR:-$(_yaml_local save_dir)}"
DEVICE="${DEVICE:-$(_yaml_local device)}"
DEVICE="${DEVICE:-mps}"

DOWNLOAD="${DOWNLOAD:-false}"
DEBUG="${DEBUG:-false}"
HIDE_PROGRESS="${HIDE_PROGRESS:-false}"

# ── Activate conda env ────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate simsiam-mps

cd "$REPO_ROOT"

# ── Validate required fields ──────────────────────────────────────────────────
if [ -z "$EVAL_FROM" ]; then
    echo "ERROR: checkpoint path is not set."
    echo "  Set eval_from in configs/linear_eval.yaml (local.eval_from)"
    echo "  or pass:  EVAL_FROM=/path/to/checkpoint.pth bash $0"
    exit 1
fi

# ── ckpt_dir: required by build_args but unused by linear_eval; auto-generated ─
CKPT_BASENAME="$(basename "$EVAL_FROM" .pth)"
CKPT_DIR="/tmp/linear_eval_ckpt_${CKPT_BASENAME}"

# ── Build argument list ───────────────────────────────────────────────────────
ARGS=(
    --config-file "$CONFIG_FILE"
    --eval_from   "$EVAL_FROM"
    --data_dir    "$DATA_DIR"
    --log_dir     "$SAVE_DIR"
    --ckpt_dir    "$CKPT_DIR"
    --device      "$DEVICE"
)

[ "$DOWNLOAD"      = "true" ] && ARGS+=(--download)
[ "$DEBUG"         = "true" ] && ARGS+=(--debug)
[ "$HIDE_PROGRESS" = "true" ] && ARGS+=(--hide_progress)

# Forward any extra CLI args passed to this script.
ARGS+=("$@")

echo "Config     : $CONFIG_FILE"
echo "Checkpoint : $EVAL_FROM"
echo "Data dir   : $DATA_DIR"
echo "Save dir   : $SAVE_DIR"
echo "Device     : $DEVICE"
echo ""
python "$REPO_ROOT/linear_eval.py" "${ARGS[@]}"
