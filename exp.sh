#!/usr/bin/env bash
# Usage:
#   ./exp.sh index    -- re-index all experiments
#   ./exp.sh serve    -- start browser at http://localhost:8000/experiment_browser.html
#   ./exp.sh browse   -- index then serve

PYTHON="/c/Users/User/anaconda3/envs/mini-icl/python.exe"
RESULTS_DIR="results"
PORT=8000

case "$1" in
  index)
    "$PYTHON" -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('experiment_index', 'src/icl/utils/legacy/experiment_index.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.index_all_experiments('$RESULTS_DIR')
"
    ;;
  serve)
    "$PYTHON" server.py $PORT
    ;;
  browse)
    "$PYTHON" -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('experiment_index', 'src/icl/utils/legacy/experiment_index.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.index_all_experiments('$RESULTS_DIR')
"
    "$PYTHON" server.py $PORT
    ;;
  *)
    echo "Usage: $0 {index|serve|browse}"
    exit 1
    ;;
esac
