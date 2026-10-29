#!/bin/sh
# M7 wrap-up: (1) extend the threshold grid on 256M (run_m7.py resumes from the seeded .partial.csv, so only
# the new configs run), (2) 500M replication on the same grid. Safe to stop and re-run.
# Usage (repo root):   caffeinate -i sh experiments/run_m7_part2.sh      Progress: tail -f logs/part2.log
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
mkdir -p logs
exec >> logs/part2.log 2>&1
step() { echo; echo "=== $(date '+%F %T')  $*"; "$@"; echo "--- exit $? at $(date '+%T')"; }

step $PY experiments/run_m7.py s256 --extras
step $PY experiments/analyze_m7.py
step $PY experiments/extract_many.py s500 HuggingFaceTB/SmolVLM-500M-Instruct mps float32
step $PY experiments/run_m7.py s500
step $PY experiments/analyze_m7.py --plots
echo; echo "=== $(date '+%F %T')  PART2 DONE"
