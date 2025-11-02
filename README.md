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