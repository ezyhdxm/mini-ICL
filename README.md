# mini-ICL
A repo containing the essential functionality for investigating in-context learning and task vectors.

## Installation

### Using Conda (Recommended)

1. Create a new conda environment:
   ```bash
   conda create -n mini-icl python=3.10 -y
   conda activate mini-icl
   ```

2. Install `torch` first, for example:
    ```bash
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
    if you are using `conda`. 

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

Once installed, you can import the package:
```python
from icl.models import Transformer
from icl.latent_markov import LatentMarkov
from icl.utils import train_model, extract_task_vector_markov
```

Experiments can be reproduced using the notebooks provided under the `notebooks` folder.

## Experiment Browser

The project includes an experiment indexing and browsing system to help you search through experiment results.

### Indexing Experiments

To index all experiments in the `results` folder:

```bash
# Method 1: Using Python module
python -m icl.utils.experiment_index

# Method 2: Using the convenience script
python scripts/index_experiments.py

# Method 3: Specify custom results directory
python -m icl.utils.experiment_index results
```

This will:
1. Scan all experiment directories in `results/`
2. Extract configuration parameters from each `config.json`
3. Create a SQLite database at `results/experiment_index.db`
4. Export a JSON file at `results/experiment_index.json` for the web UI

### Using the Web UI

1. After indexing, open `results/experiment_browser.html` in your web browser
2. Use the search form to filter experiments by:
   - Task name (latent, linear, etc.)
   - Vocabulary size
   - Sequence length
   - Embedding dimension
   - Number of layers
   - Alpha parameter
   - Number of tasks
   - Learning rate
3. Click "Search" to filter results, or "Clear" to reset

### Programmatic Search

You can also search experiments programmatically:

```python
from icl.utils.experiment_index import ExperimentIndex

# Initialize index
index = ExperimentIndex()

# Search by parameters
results = index.search_experiments(
    task_name="latent",
    vocab_size=10,
    emb_dim=128
)

# Print results
for exp in results:
    print(f"{exp['exp_name']}: {exp['exp_path']}")
```

### Updating the Index

Run the indexing command again whenever you add new experiments. The database will be updated with new experiments and existing entries will be refreshed. 