#!/usr/bin/env bash
#
# download_data.sh — fetch the WebShop and ALFWorld environment data.
#
# These datasets are public but large (the full WebShop catalog alone is
# ~5.2 GB), so they are NOT bundled with the repository.
#   - ALFWorld game files are fetched via `alfworld-download` into
#     $ALFWORLD_DATA (default ~/.cache/alfworld) — the same location the env
#     package and configs read them from (config_tw.yaml -> alfworld/info.py).
#     Export ALFWORLD_DATA to override; use the *same* value when you train.
#   - Three small WebShop variant files (items_shuffle_1000.json,
#     items_ins_v2_1000.json, items_human_ins.json) are already shipped and
#     back the `webshop.use_small: true` code path; the full WebShop catalog is
#     a manual fetch (see below), only needed for full-scale (non-use_small) runs.
#
# Usage:
#   bash download_data.sh [--webshop] [--alfworld]   # default: both
#
set -euo pipefail

DO_WEBSHOP=1
DO_ALFWORLD=1

if [[ $# -gt 0 ]]; then
  DO_WEBSHOP=0; DO_ALFWORLD=0
  for arg in "$@"; do
    case "$arg" in
      --webshop)  DO_WEBSHOP=1 ;;
      --alfworld) DO_ALFWORLD=1 ;;
      *) echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
  done
fi

# --------------------------------------------------------------------------- #
# WebShop
# --------------------------------------------------------------------------- #
if [[ "${DO_WEBSHOP}" -eq 1 ]]; then
  echo "[download_data] WebShop: all shipped configs use webshop.use_small=true,"
  echo "  backed by the three small data files already vendored under"
  echo "  third_party/verl-agent/.../webshop/webshop/data/ (items_shuffle_1000.json,"
  echo "  items_ins_v2_1000.json, items_human_ins.json) — no download is required"
  echo "  to reproduce the paper's WebShop results."
  echo "  For full-catalog (non-use_small) runs, fetch items_shuffle.json (~5.2GB)"
  echo "  and items_ins_v2.json (~178MB) from the Princeton NLP WebShop project"
  echo "  (github.com/princeton-nlp/WebShop) into that same data/ directory."
fi

# --------------------------------------------------------------------------- #
# ALFWorld
# --------------------------------------------------------------------------- #
if [[ "${DO_ALFWORLD}" -eq 1 ]]; then
  # Default to the alfworld package's own cache dir, which is what the env
  # (config_tw.yaml -> info.py) and the tools read by default — so a plain
  # `bash download_data.sh` lands the data exactly where training looks for it.
  export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
  echo "[download_data] ALFWorld: downloading via the alfworld-download CLI"
  echo "  into ALFWORLD_DATA=${ALFWORLD_DATA} ..."
  mkdir -p "${ALFWORLD_DATA}"
  if command -v alfworld-download >/dev/null 2>&1; then
    alfworld-download
    echo "  ALFWorld data -> ${ALFWORLD_DATA}"
  else
    echo "  ERROR: 'alfworld-download' not found. Activate the fedagent-alfworld" >&2
    echo "  conda env (it installs the alfworld package) first, then re-run." >&2
    exit 1
  fi
fi

echo "[download_data] done."
