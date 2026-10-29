#!/bin/sh
# Milestone 7 overnight pipeline. Safe to stop (Ctrl-C / kill) and re-run: every step resumes.
#   extract_many.py skips traces already on disk; run_m7.py skips simulations already in its .partial.csv.
# Usage (from the repo root):   caffeinate -i sh experiments/run_overnight.sh
# Progress: tail -f logs/overnight.log     A failing step is logged and the pipeline continues.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
mkdir -p logs
exec >> logs/overnight.log 2>&1
step() { echo; echo "=== $(date '+%F %T')  $*"; "$@"; echo "--- exit $? at $(date '+%T')"; }

# 1. primary dataset: 256M simulations (traces already extracted), then the analysis
step $PY experiments/run_m7.py s256 --extras
step $PY experiments/analyze_m7.py --plots

# 2. mechanism ablation at the most frequently CV-selected thresholds
CFG=$($PY - <<'PYEOF' 2>/dev/null
import pandas as pd
r = pd.read_csv("results/m7_s256_primary_k0.0.csv")
p, d = r.cfg.value_counts().index[0].split("/"); print(p, d)
PYEOF
)
echo "mechanism thresholds: ${CFG:-1.0 0.1 (fallback)}"
step $PY experiments/run_m7_mechanism.py s256 ${CFG:-1.0 0.1}

# 3. larger models: 500M, then 2.2B (bf16). Extraction is the slow part for 2.2B.
step $PY experiments/extract_many.py s500 HuggingFaceTB/SmolVLM-500M-Instruct mps float32
step $PY experiments/run_m7.py s500
step $PY experiments/extract_many.py s2200 HuggingFaceTB/SmolVLM-Instruct mps bfloat16
step $PY experiments/run_m7.py s2200
step $PY experiments/analyze_m7.py --plots
echo; echo "=== $(date '+%F %T')  ALL DONE"
