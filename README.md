<div align="right">
  <strong>Language / 语言:</strong>
  <a href="#english-content">English</a> | 
  <a href="#chinese-content">中文</a>
</div>

---

<a name="english-content"></a>
# mini-ICL

A research framework for investigating **in-context learning (ICL)** in Transformers through controlled synthetic tasks. The project trains small Transformers on tasks with known Bayesian-optimal solutions, then uses mechanistic interpretability tools — probes, causal interventions, task-vector analysis, and posterior tracking — to understand *how* the models learn in context.

## Tasks

The framework includes four synthetic task families, each with its own sampler, Bayesian baselines, and analysis suite:

| Task | Module | Description |
|------|--------|-------------|
| **Latent Markov** | `icl.latent_markov` | High-order Markov chains with latent transition matrices drawn from a Dirichlet prior |
| **Coin** | `icl.coin` | Categorical (biased coin) sequences with known and novel coin pools |
| **Noisy Linear Regression** | `icl.linear` | In-context linear regression with Gaussian noise |
| **Dyck Paths** | `icl.dyck` | Dyck (balanced parenthesis) path sequences with trie-based Bayesian predictor |

Each task supports **in-distribution (ID)** and **out-of-distribution (OOD)** evaluation, with configurable major/minor task pools.

## Installation

### Using Conda (Recommended)

1. Create a new conda environment:
   ```bash
   conda create -n mini-icl python=3.10 -y
   conda activate mini-icl
   ```

2. Install PyTorch first (not included in `pyproject.toml` to allow platform-specific builds):
   ```bash
   conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Project Structure

```
mini-ICL/
├── src/icl/
│   ├── models/              # Transformer architecture (RoPE, flash attention, KV cache)
│   ├── latent_markov/       # Latent Markov task, config, and analysis
│   │   ├── analysis/        #   Probes, interventions, directions, posteriors, variance
│   │   └── legacy/          #   Deprecated utilities
│   ├── coin/                # Coin task, config, and analysis
│   │   ├── analysis/        #   Probes, interventions, directions, posteriors, variance
│   │   └── legacy/          #   Deprecated utilities
│   ├── linear/              # Noisy linear regression task, config, and analysis
│   │   ├── analysis/        #   Probes, interventions, directions, posteriors
│   │   └── legacy/          #   Deprecated utilities
│   ├── dyck/                # Dyck path task, config, and analysis
│   │   ├── analysis/        #   Probes, variance, loss/posterior plots
│   │   └── legacy/          #   Deprecated utilities
│   ├── utils/               # Training loop, notebook helpers, plotting, unified interface
│   └── figures/             # Attention and task-vector visualizations
├── notebooks/               # Experiment notebooks (one per task)
├── legacy/                  # Standalone legacy scripts
├── tests/                   # Unit tests
├── docs/                    # Documentation (Markdown, LaTeX)
├── results/                 # Experiment outputs and browser (gitignored)
└── pyproject.toml           # Dependencies and project metadata
```

## Usage

### Quick Start

```python
from icl.models import Transformer
from icl.latent_markov import LatentMarkov, get_config_base
from icl.utils import train_model, load_everything
```

### Training a Model

```python
from icl.latent_markov import get_config_base
from icl.utils import train_model

config = get_config_base()
train_model(config)
```

Each task has its own config function with sensible defaults:

```python
from icl.latent_markov import get_config_base    # Latent Markov
from icl.coin import get_config_coin             # Coin
from icl.linear import get_config                # Linear regression
from icl.dyck import get_config_dyck             # Dyck paths
```

### Loading a Trained Model

```python
from icl.utils import load_everything

model, sampler, config = load_everything("results/your_experiment")
```

### Notebooks

Experiments can be reproduced using the Jupyter notebooks in the `notebooks/` folder:

| Notebook | Task |
|----------|------|
| `coins.ipynb` | Coin (categorical) task |
| `Latent.ipynb` | Latent Markov task |
| `Linear.ipynb` | Noisy linear regression task |
| `Dyck.ipynb` | Dyck path task |

> **Note:** Intermediate results and model checkpoints are large and not included in the repo. Contact the authors if you need access.

## Analysis Toolkit

Each task includes a structured `analysis/` subpackage with the following capabilities:

### Probes
Train linear probes on hidden representations to predict task identity, posterior beliefs, or sufficient statistics.

### Interventions
Causal interventions on hidden states to test whether identified directions are functionally relevant:
- **Remove task subspace** — zero out the task-vector direction and measure loss increase
- **Inject task vector** — swap in a different task's direction and check if predictions follow
- **Optimal orthogonal direction** — find and remove the single direction most predictive of loss
- **Task-specific** — bigram removal (latent), unigram removal (coin), sufficient-stats removal (linear)

### Directions & Decomposition
Analyze what information the hidden states encode via R² regression against task subspaces, posterior projections, and OOD delta-h directions.

### Posteriors
Compare model predictions against Bayesian baselines:
- **KL divergence** — model vs. known-pool Bayes and vs. Dirichlet Bayes
- **Task posterior tracking** — posterior beliefs over task identity across sequence positions

### Variance Analysis
Token-conditioned residual variance (P1 variance) and task-level variance decomposition across layers and positions.

### Trajectory Plots
Visualize how hidden-state projections evolve over sequence positions and training time.

## Model Architecture

The default Transformer (`icl.models.Transformer`) uses:
- Rotary positional embeddings (RoPE)
- Flash attention with causal masking
- SiLU-activated MLP blocks
- Layer normalization (pre-norm)
- Optional KV cache for incremental decoding

The linear regression task uses a separate GPT-2 style architecture (`icl.linear.lr_transformer`).

## Experiment Browser

The project includes an experiment indexing and browsing system.

### Indexing Experiments

```bash
python -m icl.utils.experiment_index          # Index all experiments in results/
python scripts/index_experiments.py           # Alternative convenience script
```

This scans all experiment directories, extracts configuration parameters, and creates a SQLite database and JSON index.

### Using the Web UI

Open `results/experiment_browser.html` in a browser to search experiments by task name, vocabulary size, sequence length, embedding dimension, number of layers, and other parameters.

### Programmatic Search

```python
from icl.utils.experiment_index import ExperimentIndex

index = ExperimentIndex()
results = index.search_experiments(task_name="latent", vocab_size=10, emb_dim=128)
for exp in results:
    print(f"{exp['exp_name']}: {exp['exp_path']}")
```

---

<details>
<summary><strong>中文版本 / Chinese Version (点击展开 / Click to expand)</strong></summary>

<a name="chinese-content"></a>
# mini-ICL

一个用于研究 Transformer **上下文学习 (ICL)** 机制的研究框架。本项目在具有已知贝叶斯最优解的可控合成任务上训练小型 Transformer，然后利用机械可解释性工具——探针、因果干预、任务向量分析和后验追踪——来理解模型*如何*在上下文中学习。

## 任务

框架包含四个合成任务族，每个任务都有独立的采样器、贝叶斯基线和分析工具：

| 任务 | 模块 | 描述 |
|------|--------|------|
| **隐马尔可夫** | `icl.latent_markov` | 从 Dirichlet 先验中采样转移矩阵的高阶马尔可夫链 |
| **硬币** | `icl.coin` | 具有已知和未知硬币池的分类（偏硬币）序列 |
| **噪声线性回归** | `icl.linear` | 带高斯噪声的上下文线性回归 |
| **Dyck 路径** | `icl.dyck` | 基于 Trie 的贝叶斯预测器的 Dyck（平衡括号）路径序列 |

每个任务均支持**分布内 (ID)** 和**分布外 (OOD)** 评估，并可配置主/次任务池。

## 安装

### 使用 Conda（推荐）

1. 创建新的 conda 环境：
   ```bash
   conda create -n mini-icl python=3.10 -y
   conda activate mini-icl
   ```

2. 首先安装 PyTorch（未包含在 `pyproject.toml` 中以支持平台特定的构建）：
   ```bash
   conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. 以可编辑模式安装包：
   ```bash
   pip install -e .
   ```

## 项目结构

```
mini-ICL/
├── src/icl/
│   ├── models/              # Transformer 架构（RoPE、flash attention、KV cache）
│   ├── latent_markov/       # 隐马尔可夫任务、配置与分析
│   │   ├── analysis/        #   探针、干预、方向分析、后验、方差
│   │   └── legacy/          #   已弃用的工具
│   ├── coin/                # 硬币任务、配置与分析
│   │   ├── analysis/        #   探针、干预、方向分析、后验、方差
│   │   └── legacy/          #   已弃用的工具
│   ├── linear/              # 噪声线性回归任务、配置与分析
│   │   ├── analysis/        #   探针、干预、方向分析、后验
│   │   └── legacy/          #   已弃用的工具
│   ├── dyck/                # Dyck 路径任务、配置与分析
│   │   ├── analysis/        #   探针、方差、损失/后验图
│   │   └── legacy/          #   已弃用的工具
│   ├── utils/               # 训练循环、notebook 工具、绘图、统一接口
│   └── figures/             # 注意力和任务向量可视化
├── notebooks/               # 实验 notebook（每个任务一个）
├── legacy/                  # 独立的遗留脚本
├── tests/                   # 单元测试
├── docs/                    # 文档（Markdown、LaTeX）
├── results/                 # 实验输出和浏览器（已 gitignore）
└── pyproject.toml           # 依赖和项目元数据
```

## 使用方法

### 快速开始

```python
from icl.models import Transformer
from icl.latent_markov import LatentMarkov, get_config_base
from icl.utils import train_model, load_everything
```

### 训练模型

```python
from icl.latent_markov import get_config_base
from icl.utils import train_model

config = get_config_base()
train_model(config)
```

每个任务都有自己的配置函数和合理的默认值：

```python
from icl.latent_markov import get_config_base    # 隐马尔可夫
from icl.coin import get_config_coin             # 硬币
from icl.linear import get_config                # 线性回归
from icl.dyck import get_config_dyck             # Dyck 路径
```

### 加载已训练的模型

```python
from icl.utils import load_everything

model, sampler, config = load_everything("results/your_experiment")
```

### Notebooks

可以通过 `notebooks/` 文件夹中的 Jupyter notebook 来重现实验：

| Notebook | 任务 |
|----------|------|
| `coins.ipynb` | 硬币（分类）任务 |
| `Latent.ipynb` | 隐马尔可夫任务 |
| `Linear.ipynb` | 噪声线性回归任务 |
| `Dyck.ipynb` | Dyck 路径任务 |

> **注意：** 中间结果和模型检查点文件较大，未包含在仓库中。如需访问请联系作者。

## 分析工具

每个任务都包含结构化的 `analysis/` 子包，提供以下功能：

### 探针
在隐层表示上训练线性探针，用于预测任务身份、后验信念或充分统计量。

### 干预
对隐层状态进行因果干预，测试所识别的方向是否具有功能相关性：
- **移除任务子空间** — 将任务向量方向置零并测量损失增加
- **注入任务向量** — 替换为不同任务的方向并检查预测是否跟随变化
- **最优正交方向** — 找到并移除对损失最具预测性的单一方向
- **任务特定** — bigram 移除（隐马尔可夫）、unigram 移除（硬币）、充分统计量移除（线性）

### 方向与分解
通过 R² 回归分析隐层状态编码的信息，包括任务子空间、后验投影和 OOD delta-h 方向。

### 后验
将模型预测与贝叶斯基线进行比较：
- **KL 散度** — 模型 vs. 已知池贝叶斯 和 vs. Dirichlet 贝叶斯
- **任务后验追踪** — 在序列位置上追踪任务身份的后验信念

### 方差分析
跨层和位置的 token 条件残差方差（P1 方差）和任务级方差分解。

### 轨迹图
可视化隐层状态投影在序列位置和训练时间上的演变。

## 模型架构

默认 Transformer（`icl.models.Transformer`）使用：
- 旋转位置编码（RoPE）
- 带因果掩码的 Flash Attention
- SiLU 激活的 MLP 块
- 层归一化（pre-norm）
- 可选的 KV cache 增量解码

线性回归任务使用独立的 GPT-2 风格架构（`icl.linear.lr_transformer`）。

## 实验浏览器

项目包含实验索引和浏览系统。

### 索引实验

```bash
python -m icl.utils.experiment_index          # 索引 results/ 中的所有实验
python scripts/index_experiments.py           # 便捷脚本
```

扫描所有实验目录，提取配置参数，创建 SQLite 数据库和 JSON 索引。

### 使用 Web UI

在浏览器中打开 `results/experiment_browser.html`，可按任务名称、词汇表大小、序列长度、嵌入维度、层数等参数搜索实验。

### 编程方式搜索

```python
from icl.utils.experiment_index import ExperimentIndex

index = ExperimentIndex()
results = index.search_experiments(task_name="latent", vocab_size=10, emb_dim=128)
for exp in results:
    print(f"{exp['exp_name']}: {exp['exp_path']}")
```

</details>

<div align="right">
  <strong>Language / 语言:</strong>
  <a href="#english-content">English</a> | 
  <a href="#chinese-content">中文</a>
</div>
