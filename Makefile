PYTHON := C:\Users\User\anaconda3\envs\mini-icl\python.exe
RESULTS_DIR := results
PORT := 8000

.PHONY: index serve browse

## Re-index all experiments and export results/experiment_index.json
index:
	$(PYTHON) -c "import sys; sys.path.insert(0,'src'); from icl.utils.legacy.experiment_index import index_all_experiments; index_all_experiments('$(RESULTS_DIR)')"

## Start the experiment browser at http://localhost:$(PORT)/experiment_browser.html
serve:
	$(PYTHON) server.py $(PORT)

## Re-index then serve
browse: index serve
