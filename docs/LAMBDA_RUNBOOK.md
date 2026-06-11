# Running experiments on Lambda Cloud (mini-ICL)

This is an operational runbook for training/analyzing the architecture grid
(transformer / rnn / lstm / mamba on latent-Markov and coin tasks) on a Lambda
Cloud GPU. It is written so an agent (or person) can launch a fresh instance and
reproduce a run end to end. **The heavy compute is on the GPU; the laptop only
drives it over SSH and views/pulls results.**

---

## 0. Mental model (read first)

- **The instance is ephemeral; the persistent filesystem is not.** Lambda wipes
  the instance's boot disk on terminate. A **Persistent Filesystem** (created in
  Lambda → Storage) survives. We keep the repo, the `uv` venv, checkpoints, and
  results on it, so relaunching never rebuilds anything.
- Filesystem mount: **`/lambda/nfs/<name>`** (ours is `/lambda/nfs/ICL`, also
  reachable via the `~/ICL` symlink). Repo lives at `~/ICL/mini-ICL`.
- Login user is **`ubuntu`**. Stop billing by **terminating** (there is no
  stop/pause); checkpoints persist on the filesystem.

---

## 1. Launch an instance (Lambda dashboard)

1. **Instances → Launch instance.**
2. **GPU:** `1x A10` is plenty for these models (~1.2M params). Use a bigger GPU
   only for the real-LLM notebook or much larger models.
3. **Region:** must match the region of the persistent filesystem.
4. **Image:** **Lambda Stack 22.04** (ships NVIDIA driver + CUDA; we install our
   own CUDA torch into a `uv` venv on top — no conflict).
5. **Attach:** the persistent filesystem (`ICL`) **and** your SSH key.
6. Launch; wait for **Running** + a public IP.

## 2. SSH config

Add to `~/.ssh/config` on the laptop (update `HostName` to the new IP each
launch — the IP changes every time):

```
Host lambda
    HostName <NEW_IP>
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
    ServerAliveCountMax 6
    TCPKeepAlive yes
```

Test: `ssh lambda 'nvidia-smi'`. (First connection may refuse for ~30–60s while
sshd comes up — just retry. "Network is unreachable"/timeout that persists across
two tries usually means the instance was terminated; ping the IP to confirm.)

## 3. One-time environment setup (on the persistent filesystem)

Only needed the first time a filesystem is used (or after deleting the venv):

```bash
ssh lambda
cd ~/ICL
# clone onto the persistent FS (skip if ~/ICL/mini-ICL already exists)
git clone https://github.com/ezyhdxm/mini-ICL.git mini-ICL
cd mini-ICL && git checkout multi-arch-icl

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync                                                    # pyproject deps
uv pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU torch

# persist env for interactive shells
cat >> ~/.bashrc <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/lambda/nfs/ICL/.hf
export WANDB_DIR=/lambda/nfs/ICL/wandb
export WANDB_MODE=offline
EOF
```

Verify: `uv run python -c "import torch; print(torch.cuda.is_available())"` → `True`.

> Note: `uv sync` installs everything **except** torch (torch is intentionally
> not pinned in `pyproject.toml`); always install the cu121 wheel explicitly.
> Non-interactive `ssh lambda '<cmd>'` does **not** source `~/.bashrc`, so for
> scripted commands export the env inline (see below).

## 4. Run an experiment

Always run long jobs under `tmux` so they survive disconnects, and `tee` to a log
on the persistent FS. Standard env preamble for non-interactive runs:

```bash
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline WANDB_DIR=/lambda/nfs/ICL/wandb HF_HOME=/lambda/nfs/ICL/.hf
cd ~/ICL/mini-ICL
```

### 4a. Latent-Markov architecture grid (the main experiment)

```bash
tmux new -s run
# train {transformer,lstm,rnn,mamba} x {4,1024} latents (uniform-over-K prior)
uv run python scripts/train_arch_grid.py --arch transformer lstm rnn mamba --n-tasks 4 1024
# analyze: conditional variance, multi-layer alpha-beta (Fig 3), task-vector R2,
# KL to the two CORRECT Bayes solutions; K=1024 cells use task subsampling.
uv run python scripts/analyze_arch_grid.py \
    --max-tasks 32 --positions $(seq 0 6 186) --batch-size 32 --beta-layers 0 1 2 3 4 5
uv run python scripts/check_ood_loss.py --num-samples 256 --ks 4 1024     # OOD loss vs Bayes-optimal
uv run python scripts/plot_training_loss.py                                # train + ID/OOD loss
uv run python scripts/combine_arch_grid.py                                 # cross-arch overlays
# detach: Ctrl-b d   |   reattach: tmux attach -t run
```

Outputs land in `results/arch_grid_analysis/` (per-cell + `combined/` + `ood_loss/`
+ `training/`) and the manifest `results/arch_grid_manifest.json`.

### 4b. Coin (real-coin) task — vocab=2

```bash
uv run python scripts/train_arch_grid.py --task coin --vocab-size 2 \
    --arch transformer rnn lstm mamba --n-tasks 3 \
    --manifest results/coin_grid_manifest.json
uv run python scripts/coin_task_vectors.py --manifest results/coin_grid_manifest.json
# -> results/coin_analysis/<arch>_coin/projection.png + combined_coin_projection.png
```

### 4c. Adding one architecture to an existing grid (no retraining of the rest)

`train_arch_grid` **merges** into the manifest, and `analyze_arch_grid --archs`
restricts processing. Example — add mamba to a done latent grid:

```bash
uv run python scripts/train_arch_grid.py --arch mamba --n-tasks 4 1024         # adds 2 cells to manifest
uv run python scripts/analyze_arch_grid.py --archs mamba \
    --max-tasks 32 --positions $(seq 0 6 186) --batch-size 32 --beta-layers 0 1 2 3 4 5
uv run python scripts/check_ood_loss.py --num-samples 256 --ks 4 1024
uv run python scripts/plot_training_loss.py
uv run python scripts/combine_arch_grid.py                                      # re-overlay all archs
```

## 5. Monitor

```bash
ssh lambda 'tail -n 30 ~/ICL/run.log'          # progress (training prints per cell; analysis per position)
ssh lambda 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
ssh lambda 'cat ~/ICL/mini-ICL/results/arch_grid_manifest.json | python3 -c "import json,sys; [print(r[\"arch\"],r[\"n_tasks\"],r.get(\"trained\")) for r in json.load(sys.stdin)[\"runs\"]]"'
# checkpoints advance every 1000 steps -> proof a cell is progressing:
ssh lambda 'ls -t ~/ICL/mini-ICL/results/latent/<exp>/checkpoints | head -3'
```

## 6. Pull results to the laptop

Results are small (the analysis npz hold metrics, not raw hiddens):

```bash
rsync -az lambda:'~/ICL/mini-ICL/results/arch_grid_analysis' ./results/
rsync -az lambda:'~/ICL/mini-ICL/results/coin_analysis' ./results/
```

Checkpoints (~1.5 GB each, ~9 GB for 6 cells) stay on the FS; pull only if you
want an offline copy: `rsync -az lambda:'~/ICL/mini-ICL/results/latent' ./results/`.

## 7. Terminate (stop billing)

Dashboard → Instances → **Terminate**. The GPU bill stops; the persistent
filesystem (repo, venv, checkpoints, results) **survives**. Next time: launch a
new instance, attach the `ICL` filesystem, update `~/.ssh/config` HostName, and
continue. The filesystem itself keeps a small storage fee (~$0.20/GB/mo).

---

## 8. Gotchas / things that bit us

- **CUDA vs CPU torch.** The repo assumes CUDA. The local laptop venv may have a
  CPU torch (for testing); the box must use the **cu121** wheel. Two CPU-only
  guards are already in the code (`torch.cuda.synchronize` behind `is_available`;
  non-mutating exp-name hash that preserves `config.device`), so CPU smoke runs
  work, but real training needs the GPU.
- **Recurrent + Mamba are slow.** RNN/LSTM/Mamba use sequential scans; the
  **K=1024** cells are the long poles (LSTM K=1024 ≈ 75 min; Mamba slower). Budget
  hours, not minutes, for a full 4-arch grid. Mamba's pure-PyTorch scan is the
  slowest; if needed, an associative/parallel scan would speed it ~24x.
- **K=1024 analysis must subsample.** The variance/R2 extractors iterate every
  task; always pass `--max-tasks 32` (and subsampled `--positions`) for large K,
  else a single cell can take many hours. The interpolation/alpha-beta analysis is
  auto-skipped above `--max-lambda-k` (default 64).
- **exp_name = hash of the full config** (`train_<md5>`). It includes `arch`,
  `vocab_size`, `n_tasks`, epochs, etc., so different settings never collide AND
  analysis finds the right checkpoint via `get_exp_name(...)` with matching args.
  If you change a training hyperparam, the analysis must use the same one.
- **Manifest is the source of truth** linking (arch, n_tasks) -> exp_name; the
  launcher writes/merges it, and all analysis/combine scripts read it.
- **Bayesian solutions are DGP-specific.** The two-modes KL uses the correct
  uniform-over-K solutions (`two_modes_kl.py`: known-pool 1/K + Dirichlet-new),
  not the paper's 3-known+minor baselines. If the DGP changes, update those.
- **Network blips.** Transient SSH drops happen; `tmux` keeps the job alive. A
  drop that persists across two retries (ping the IP) means the instance is gone.
- **wandb** runs offline here (`WANDB_MODE=offline`); training also writes a
  `log.json` per checkpoint dir (train/loss, eval/IDLoss, eval/OODLoss) which
  `plot_training_loss.py` reads — no wandb account needed.

## 9. One-line summary for an agent

> Launch A10 + Lambda Stack 22.04 in the filesystem's region, attach the `ICL`
> filesystem + SSH key; set `~/.ssh/config` Host `lambda` to the new IP; `ssh
> lambda`, `cd ~/ICL/mini-ICL`, `git pull`; if no venv, do §3; then run §4 under
> `tmux`; pull with §6; terminate with §7.
