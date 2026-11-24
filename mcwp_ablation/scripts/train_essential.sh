#!/usr/bin/env bash
# Sequential queue. Repeat with a larger TOTAL target to resume all four runs.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MAX_STEPS=${1:-20000}
cd "$ROOT"
for VARIANT in monolithic static_fusion horizon1 first_order; do
    RUN="${VARIANT}${RUN_SUFFIX:-}"
    bash scripts/train4.sh "$VARIANT" "$MAX_STEPS" "$RUN" auto
    if python - "$ROOT/runs/$RUN/training_status.json" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1])
sys.exit(0 if path.exists() and json.loads(path.read_text()).get('stopped_on_request') else 1)
PY
    then
        echo "Queue stopped after saving $RUN. Restart this queue to resume."
        exit 0
    fi
done
echo "ALL FOUR REQUIRED VARIANTS REACHED $MAX_STEPS COMPLETED UPDATES."
