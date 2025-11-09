<div align="right">
  <strong>Language / 语言:</strong>
  <a href="#english-content">English</a> | 
  <a href="#chinese-content">中文</a>
</div>

---

<a name="english-content"></a>
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
The intermediate results and model checkpoints are very large. Feel free to ask me about a physical copy if you want access.  

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

---

<details>
<summary><strong>中文版本 / Chinese Version (点击展开 / Click to expand)</strong></summary>

<a name="chinese-content"></a>
# mini-ICL

一个用于研究上下文学习和任务向量的核心功能库。

## 安装

### 使用 Conda（推荐）

1. 创建新的 conda 环境：
   ```bash
   conda create -n mini-icl python=3.10 -y
   conda activate mini-icl
   ```

2. 首先安装 `torch`，例如：
    ```bash
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
    如果你使用 `conda`。

3. 以可编辑模式安装包：
   ```bash
   pip install -e .
   ```

## 使用方法

安装后，可以导入包：
```python
from icl.models import Transformer
from icl.latent_markov import LatentMarkov
from icl.utils import train_model, extract_task_vector_markov
```

可以通过 `notebooks` 文件夹下提供的 notebook 来重现实验。

## 实验浏览器

项目包含一个实验索引和浏览系统，帮助您搜索实验结果。

### 索引实验

要索引 `results` 文件夹中的所有实验：

```bash
# 方法 1：使用 Python 模块
python -m icl.utils.experiment_index

# 方法 2：使用便捷脚本
python scripts/index_experiments.py

# 方法 3：指定自定义结果目录
python -m icl.utils.experiment_index results
```

这将：
1. 扫描 `results/` 中的所有实验目录
2. 从每个 `config.json` 中提取配置参数
3. 在 `results/experiment_index.db` 创建 SQLite 数据库
4. 导出 JSON 文件到 `results/experiment_index.json` 供 Web UI 使用

### 使用 Web UI

1. 索引后，在 Web 浏览器中打开 `results/experiment_browser.html`
2. 使用搜索表单按以下条件筛选实验：
   - 任务名称（latent、linear 等）
   - 词汇表大小
   - 序列长度
   - 嵌入维度
   - 层数
   - Alpha 参数
   - 任务数量
   - 学习率
3. 点击 "Search" 筛选结果，或点击 "Clear" 重置

### 编程方式搜索

你也可以通过编程方式搜索实验：

```python
from icl.utils.experiment_index import ExperimentIndex

# 初始化索引
index = ExperimentIndex()

# 按参数搜索
results = index.search_experiments(
    task_name="latent",
    vocab_size=10,
    emb_dim=128
)

# 打印结果
for exp in results:
    print(f"{exp['exp_name']}: {exp['exp_path']}")
```

### 更新索引

添加新实验后，记得重新运行索引命令。数据库将更新新实验，现有条目将刷新。

</details>

<div align="right">
  <strong>Language / 语言:</strong>
  <a href="#english-content">English</a> | 
  <a href="#chinese-content">中文</a>
</div>