"""ANCOVA: additive separability for continuous covariates (linear regression)."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch


@dataclass
class ANCOVAResult:
    """ANCOVA slope-homogeneity test for a single (layer, position)."""

    r2_additive: float
    r2_full: float
    separability_gap: float

    ss_total: float
    ss_res_additive: float
    ss_res_full: float

    n_tasks: int
    n_covariate_dims: int
    n_samples: int

    layer_num: Optional[int] = None
    position: Optional[int] = None

    r2_additive_val: Optional[float] = None
    r2_full_val: Optional[float] = None
    separability_gap_val: Optional[float] = None


def ancova_separability(
    hiddens: torch.Tensor,
    covariates: torch.Tensor,
    task_labels: torch.Tensor,
    eps: float = 1e-10,
) -> ANCOVAResult:
    """ANCOVA slope-homogeneity test for a single (layer, position).

    Compares an additive model (common slopes across tasks) against a
    full interaction model (task-specific slopes) to quantify whether
    the effect of the continuous covariate x_t on h is task-independent.

    The additive model is::

        h = one_hot(k) @ W_task + x_t @ W_x + b

    The full model uses per-task slopes with no redundant columns::

        h = one_hot(k) @ W_task + [one_hot(k) ⊗ x_t] @ W_int + b

    (The shared x_t term is dropped from the full model because it
    is in the column span of the interaction terms.)

    All computations use float64 for numerical stability.

    Parameters
    ----------
    hiddens : torch.Tensor
        Shape ``(N, D)`` — hidden states.
    covariates : torch.Tensor
        Shape ``(N, d)`` — continuous covariates (e.g. x_t).
    task_labels : torch.Tensor
        Shape ``(N,)`` — integer task labels in ``{0, ..., K-1}``.
    eps : float

    Returns
    -------
    ANCOVAResult
    """
    h = hiddens.double()
    x = covariates.double()
    labels = task_labels.long()

    N, D = h.shape
    d = x.shape[1]
    K = int(labels.max().item()) + 1

    one_hot_k = torch.zeros(N, K, dtype=torch.float64, device=h.device)
    one_hot_k.scatter_(1, labels.unsqueeze(1), 1.0)

    def _ols_r2(X, Y):
        ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
        X_aug = torch.cat([X, ones], dim=1)
        W = torch.linalg.lstsq(X_aug, Y).solution
        pred = X_aug @ W
        ss_res = ((Y - pred) ** 2).sum().item()
        ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + eps)
        return r2, ss_res, ss_tot

    # Additive: h = one_hot(k) @ W_task + x_t @ W_x + b
    X_add = torch.cat([one_hot_k, x], dim=1)  # (N, K + d)
    r2_add, ss_res_add, ss_tot = _ols_r2(X_add, h)

    # Full: h = one_hot(k) @ W_task + [one_hot(k) ⊗ x_t] @ W_int + b
    interaction = one_hot_k.unsqueeze(2) * x.unsqueeze(1)  # (N, K, d)
    interaction = interaction.reshape(N, K * d)
    X_full = torch.cat([one_hot_k, interaction], dim=1)  # (N, K + K*d)
    r2_full, ss_res_full, _ = _ols_r2(X_full, h)

    gap = r2_full - r2_add

    return ANCOVAResult(
        r2_additive=r2_add,
        r2_full=r2_full,
        separability_gap=gap,
        ss_total=ss_tot,
        ss_res_additive=ss_res_add,
        ss_res_full=ss_res_full,
        n_tasks=K,
        n_covariate_dims=d,
        n_samples=N,
    )


def mlp_ancova_separability(
    hiddens: torch.Tensor,
    covariates: torch.Tensor,
    task_labels: torch.Tensor,
    hidden_dim: int = 256,
    n_hidden_layers: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 800,
    val_fraction: float = 0.2,
    patience: int = 80,
    batch_size: int = 512,
    verbose: bool = False,
    eps: float = 1e-10,
) -> ANCOVAResult:
    """MLP-based separability test for a single (layer, position).

    Replaces the linear covariate model in ``ancova_separability`` with
    a small MLP so that nonlinear token effects are properly captured.

    Additive model::

        h = onehot(k) @ W_task + MLP_shared(x_t) + b

    Full (interaction) model::

        h = MLP(onehot(k), x_t)

    Both models are trained with Adam + early stopping on a held-out
    validation set. The gap  R²_full − R²_additive  measures genuine
    task–token interaction that persists even after accounting for
    nonlinear token effects.

    Parameters
    ----------
    hiddens : (N, D) tensor
    covariates : (N, d) tensor
    task_labels : (N,) long tensor  — values in {0, ..., K-1}
    hidden_dim : int — MLP hidden width
    n_hidden_layers : int — number of hidden layers (≥1)
    lr, weight_decay : float — Adam parameters
    n_epochs : int — max training epochs
    val_fraction : float — held-out fraction for early stopping
    patience : int — early-stopping patience (epochs without improvement)
    batch_size : int — minibatch size (0 = full batch)
    verbose : bool
    eps : float

    Returns
    -------
    ANCOVAResult  (same dataclass as the linear version)
    """
    import torch.nn as nn
    import torch.optim as optim

    h = hiddens.float()
    x = covariates.float()
    labels = task_labels.long()

    N, D = h.shape
    d = x.shape[1]
    K = int(labels.max().item()) + 1
    device = h.device

    one_hot_k = torch.zeros(N, K, dtype=torch.float32, device=device)
    one_hot_k.scatter_(1, labels.unsqueeze(1), 1.0)

    # ---- train/val split (deterministic) ----
    n_val = max(1, int(N * val_fraction))
    n_train = N - n_val
    perm = torch.randperm(N, device=device)
    idx_train, idx_val = perm[:n_train], perm[n_train:]

    def _make_mlp(in_dim, out_dim):
        layers = []
        cur = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.GELU())
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        return nn.Sequential(*layers).to(device)

    def _count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _train_model(model, X_train, Y_train, X_val, Y_val, model_name="model"):
        n_params = _count_params(model)
        if verbose:
            print(f"  [{model_name}] params={n_params:,d}, "
                  f"training up to {n_epochs} epochs ...")

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        final_epoch = 0
        bs = batch_size if batch_size > 0 else n_train

        for epoch in range(n_epochs):
            model.train()
            shuf = torch.randperm(X_train.shape[0], device=device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, X_train.shape[0], bs):
                idx = shuf[start:start + bs]
                pred = model(X_train[idx])
                loss = ((pred - Y_train[idx]) ** 2).sum() / idx.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = ((val_pred - Y_val) ** 2).mean().item()

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
                print(f"    epoch {epoch:4d}: train_mse={epoch_loss / n_batches:.4f}  "
                      f"val_mse={val_loss:.4f}")

            final_epoch = epoch
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  [{model_name}] early stop at epoch {epoch} "
                          f"(best_val_mse={best_val_loss:.4f})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        if verbose and epochs_no_improve < patience:
            print(f"  [{model_name}] finished {final_epoch + 1} epochs "
                  f"(best_val_mse={best_val_loss:.4f})")
        return model

    def _compute_r2(model, X, Y):
        model.eval()
        with torch.no_grad():
            pred = model(X)
        ss_res = ((Y - pred) ** 2).sum().item()
        ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum().item()
        return 1.0 - ss_res / (ss_tot + eps), ss_res, ss_tot

    # ---- OLS warm-start ----
    X_ols = torch.cat([one_hot_k, x, torch.ones(N, 1, device=device)], dim=1)
    W_ols = torch.linalg.lstsq(X_ols, h).solution  # (K+d+1, D)
    W_task_ols = W_ols[:K]      # (K, D)
    W_x_ols = W_ols[K:K + d]   # (d, D)
    b_ols = W_ols[K + d]       # (D,)

    def _zero_init_last_linear(seq):
        for module in reversed(list(seq.modules())):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                break

    class AdditiveModel(nn.Module):
        def __init__(self_):
            super().__init__()
            self_.task_linear = nn.Linear(K, D, bias=False)
            self_.token_linear = nn.Linear(d, D, bias=False)
            self_.token_mlp = _make_mlp(d, D)
            self_.bias = nn.Parameter(torch.zeros(D))

            with torch.no_grad():
                self_.task_linear.weight.copy_(W_task_ols.T)
                self_.token_linear.weight.copy_(W_x_ols.T)
                self_.bias.copy_(b_ols)
            _zero_init_last_linear(self_.token_mlp)

        def forward(self_, inputs):
            oh, xt = inputs[:, :K], inputs[:, K:]
            return (self_.task_linear(oh) + self_.token_linear(xt)
                    + self_.token_mlp(xt) + self_.bias)

    class FullModel(nn.Module):
        def __init__(self_):
            super().__init__()
            self_.linear_skip = nn.Linear(K + d, D)
            self_.mlp = _make_mlp(K + d, D)

            with torch.no_grad():
                W_skip = torch.cat([W_task_ols, W_x_ols], dim=0)  # (K+d, D)
                self_.linear_skip.weight.copy_(W_skip.T)
                self_.linear_skip.bias.copy_(b_ols)
            _zero_init_last_linear(self_.mlp)

        def forward(self_, inputs):
            return self_.linear_skip(inputs) + self_.mlp(inputs)

    X_combined = torch.cat([one_hot_k, x], dim=1)  # (N, K+d)
    X_tr, X_va = X_combined[idx_train], X_combined[idx_val]
    Y_tr, Y_va = h[idx_train], h[idx_val]

    with torch.no_grad():
        h_ols = X_ols @ W_ols
        ss_res_ols = ((h - h_ols) ** 2).sum().item()
        ss_tot_all = ((h - h.mean(dim=0)) ** 2).sum().item()
        r2_ols = 1.0 - ss_res_ols / (ss_tot_all + eps)

    if verbose:
        print(f"[mlp_ancova] N={N}, K={K}, d_cov={d}, D_hidden={D}, "
              f"train={n_train}, val={n_val}, "
              f"arch={n_hidden_layers}x{hidden_dim}")
        print(f"[mlp_ancova] OLS baseline R² = {r2_ols:.6f} (warm-start)")

    add_model = AdditiveModel()
    add_model = _train_model(
        add_model, X_tr, Y_tr, X_va, Y_va,
        model_name="additive: W_task @ onehot(k) + MLP(x_t) + b",
    )
    r2_add, ss_res_add, ss_tot = _compute_r2(add_model, X_combined, h)
    r2_add_val, _, _ = _compute_r2(add_model, X_va, Y_va)

    if verbose:
        print(f"  [additive] R²_full={r2_add:.6f}  R²_val={r2_add_val:.6f}")

    full_model = FullModel()
    full_model = _train_model(
        full_model, X_tr, Y_tr, X_va, Y_va,
        model_name="full: MLP(onehot(k), x_t)",
    )
    r2_full, ss_res_full, _ = _compute_r2(full_model, X_combined, h)
    r2_full_val, _, _ = _compute_r2(full_model, X_va, Y_va)

    if verbose:
        print(f"  [full]     R²_full={r2_full:.6f}  R²_val={r2_full_val:.6f}")
        print(f"  gap(full_data)={r2_full - r2_add:.6f}  "
              f"gap(val)={r2_full_val - r2_add_val:.6f}")

    gap = r2_full - r2_add
    gap_val = r2_full_val - r2_add_val

    del add_model, full_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ANCOVAResult(
        r2_additive=r2_add,
        r2_full=r2_full,
        separability_gap=gap,
        ss_total=ss_tot,
        ss_res_additive=ss_res_add,
        ss_res_full=ss_res_full,
        n_tasks=K,
        n_covariate_dims=d,
        n_samples=N,
        r2_additive_val=r2_add_val,
        r2_full_val=r2_full_val,
        separability_gap_val=gap_val,
    )


def mlp_ancova_separability_from_hiddens(
    all_hiddens: torch.Tensor,
    demo_data: torch.Tensor,
    layers: "Optional[Sequence[int]]" = None,
    positions: "Optional[Sequence[int]]" = None,
    **mlp_kwargs,
) -> Dict[int, Dict[int, ANCOVAResult]]:
    """Run MLP-based ANCOVA across layers and positions for linear regression.

    Same interface as ``ancova_separability_from_hiddens`` but uses
    ``mlp_ancova_separability`` internally.

    Parameters
    ----------
    all_hiddens : (L, n_tasks, n_points, batch_size, D)
    demo_data : (batch_size, n_points, n_dims)
    layers, positions : optional index lists
    **mlp_kwargs : forwarded to ``mlp_ancova_separability``

    Returns
    -------
    dict  {layer_num: {position: ANCOVAResult}}
    """
    L, n_tasks, n_points, B, D = all_hiddens.shape
    n_dims = demo_data.shape[-1]

    if layers is None:
        layers = list(range(L))
    if positions is None:
        positions = list(range(n_points))

    task_labels = (
        torch.arange(n_tasks)
        .unsqueeze(1)
        .expand(n_tasks, B)
        .reshape(-1)
    )

    is_verbose = mlp_kwargs.get("verbose", False)
    total_fits = len(layers) * len(positions)
    fit_count = 0

    results: Dict[int, Dict[int, ANCOVAResult]] = {}

    for l_idx, l_num in enumerate(layers):
        if l_idx >= L:
            continue
        layer_results: Dict[int, ANCOVAResult] = {}

        for pos in positions:
            if pos >= n_points:
                continue

            fit_count += 1
            if is_verbose:
                print(f"\n{'='*60}")
                print(f"[mlp_ancova] Layer {l_num}, Position {pos}  "
                      f"({fit_count}/{total_fits})")
                print(f"{'='*60}")

            h = all_hiddens[l_idx, :, pos, :, :]  # (n_tasks, B, D)
            h_flat = h.reshape(n_tasks * B, D)

            x = demo_data[:, pos, :]  # (B, n_dims)
            x_flat = x.unsqueeze(0).expand(n_tasks, B, n_dims).reshape(
                n_tasks * B, n_dims,
            )

            res = mlp_ancova_separability(
                h_flat, x_flat, task_labels, **mlp_kwargs,
            )
            res.layer_num = l_num
            res.position = pos
            layer_results[pos] = res

        results[l_num] = layer_results

    return results


def mlp_ancova_separability_joint(
    all_hiddens_layer: torch.Tensor,
    demo_data: torch.Tensor,
    positions: "Optional[Sequence[int]]" = None,
    layer_num: int = 0,
    fit_position: bool = False,
    hidden_dim: int = 256,
    n_hidden_layers: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 800,
    val_fraction: float = 0.2,
    patience: int = 80,
    batch_size: int = 512,
    verbose: bool = False,
    eps: float = 1e-10,
) -> Dict[int, ANCOVAResult]:
    """Joint MLP-based separability test across positions for one layer.

    Instead of fitting a separate model at every position, this function
    pools data across all positions and fits *one* additive model and
    *one* full model.  ``W_task`` is shared.

    When ``fit_position=True``, normalised position ``t/T`` is appended
    to the covariate vector so the MLP can learn position-dependent token
    effects.  When ``False`` (default), only ``x_t`` is used.

    Both models are warm-started from the pooled OLS solution.
    After joint fitting, R² is evaluated **per position** separately.

    Parameters
    ----------
    all_hiddens_layer : (n_tasks, n_points, B, D)  — one layer
    demo_data : (B, n_points, n_dims)
    positions : list of position indices to use
    layer_num : int — for labelling results
    fit_position : bool — include normalised position as an extra input
    hidden_dim, n_hidden_layers : MLP architecture
    lr, weight_decay : Adam parameters
    n_epochs, patience : training budget
    val_fraction : held-out fraction
    batch_size : minibatch size
    verbose : bool
    eps : float

    Returns
    -------
    dict  {position: ANCOVAResult}
    """
    import torch.nn as nn
    import torch.optim as optim

    n_tasks, n_points, B, D = all_hiddens_layer.shape
    d = demo_data.shape[-1]
    K = n_tasks
    device = all_hiddens_layer.device

    if positions is None:
        positions = list(range(n_points))
    n_pos = len(positions)
    pos_max = max(max(positions), 1)

    # ---- Flatten data across (position, task, batch) ----
    h_parts, x_parts, k_parts = [], [], []
    p_parts = [] if fit_position else None
    pos_slices: Dict[int, tuple] = {}
    offset = 0
    for pos in positions:
        for ki in range(K):
            h_parts.append(all_hiddens_layer[ki, pos, :, :])     # (B, D)
            x_parts.append(demo_data[:, pos, :])                  # (B, d)
            k_parts.append(torch.full((B,), ki, dtype=torch.long, device=device))
            if fit_position:
                p_parts.append(torch.full((B, 1), pos / pos_max,
                                          dtype=torch.float32, device=device))
        pos_slices[pos] = (offset, offset + K * B)
        offset += K * B

    h_all = torch.cat(h_parts, dim=0).float()       # (N, D)
    x_all = torch.cat(x_parts, dim=0).float()       # (N, d)
    k_all = torch.cat(k_parts, dim=0)               # (N,)
    N = h_all.shape[0]

    if fit_position:
        p_all = torch.cat(p_parts, dim=0)            # (N, 1)
        cov_all = torch.cat([x_all, p_all], dim=1)   # (N, d+1)
        d_cov = d + 1
    else:
        cov_all = x_all                               # (N, d)
        d_cov = d

    one_hot_k = torch.zeros(N, K, dtype=torch.float32, device=device)
    one_hot_k.scatter_(1, k_all.unsqueeze(1).to(device), 1.0)

    # ---- train/val split ----
    n_val = max(1, int(N * val_fraction))
    n_train = N - n_val
    perm = torch.randperm(N, device=device)
    idx_train, idx_val = perm[:n_train], perm[n_train:]

    # ---- Helpers (same as per-position version) ----
    def _make_mlp(in_dim, out_dim):
        layers_list = []
        cur = in_dim
        for _ in range(n_hidden_layers):
            layers_list.append(nn.Linear(cur, hidden_dim))
            layers_list.append(nn.GELU())
            cur = hidden_dim
        layers_list.append(nn.Linear(cur, out_dim))
        return nn.Sequential(*layers_list).to(device)

    def _zero_init_last_linear(seq):
        for module in reversed(list(seq.modules())):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                break

    def _count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _train_model(model, X_train, Y_train, X_val, Y_val, model_name="model"):
        n_params = _count_params(model)
        if verbose:
            print(f"  [{model_name}] params={n_params:,d}, "
                  f"training up to {n_epochs} epochs ...")
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        final_epoch = 0
        bs = batch_size if batch_size > 0 else n_train
        for epoch in range(n_epochs):
            model.train()
            shuf = torch.randperm(X_train.shape[0], device=device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, X_train.shape[0], bs):
                idx = shuf[start:start + bs]
                pred = model(X_train[idx])
                loss = ((pred - Y_train[idx]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = ((val_pred - Y_val) ** 2).mean().item()
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
                print(f"    epoch {epoch:4d}: train_mse={epoch_loss / n_batches:.4f}  "
                      f"val_mse={val_loss:.4f}")
            final_epoch = epoch
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  [{model_name}] early stop at epoch {epoch} "
                          f"(best_val_mse={best_val_loss:.4f})")
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        if verbose and epochs_no_improve < patience:
            print(f"  [{model_name}] finished {final_epoch + 1} epochs "
                  f"(best_val_mse={best_val_loss:.4f})")
        return model

    def _compute_r2(model, X, Y):
        model.eval()
        with torch.no_grad():
            pred = model(X)
        ss_res = ((Y - pred) ** 2).sum().item()
        ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum().item()
        return 1.0 - ss_res / (ss_tot + eps), ss_res, ss_tot

    # ---- OLS warm-start (pooled across positions) ----
    X_ols = torch.cat([one_hot_k, cov_all,
                       torch.ones(N, 1, device=device)], dim=1)
    W_ols = torch.linalg.lstsq(X_ols, h_all).solution
    W_task_ols = W_ols[:K]                  # (K, D)
    W_cov_ols = W_ols[K:K + d_cov]         # (d_cov, D)
    b_ols = W_ols[K + d_cov]               # (D,)

    with torch.no_grad():
        h_ols_pred = X_ols @ W_ols
        r2_ols = 1.0 - (((h_all - h_ols_pred) ** 2).sum().item()
                        / (((h_all - h_all.mean(0)) ** 2).sum().item() + eps))

    pos_tag = "+pos" if fit_position else ""
    if verbose:
        print(f"[mlp_ancova_joint] Layer {layer_num}: "
              f"N={N} ({n_pos} pos x {K} tasks x {B} batch), "
              f"K={K}, d_cov={d}{pos_tag}, D_hidden={D}, "
              f"train={n_train}, val={n_val}, "
              f"arch={n_hidden_layers}x{hidden_dim}")
        print(f"[mlp_ancova_joint] OLS baseline R² = {r2_ols:.6f} (warm-start)")

    class AdditiveModel(nn.Module):
        def __init__(self_):
            super().__init__()
            self_.task_linear = nn.Linear(K, D, bias=False)
            self_.token_linear = nn.Linear(d_cov, D, bias=False)
            self_.token_mlp = _make_mlp(d_cov, D)
            self_.bias = nn.Parameter(torch.zeros(D, device=device))
            with torch.no_grad():
                self_.task_linear.weight.copy_(W_task_ols.T)
                self_.token_linear.weight.copy_(W_cov_ols.T)
                self_.bias.copy_(b_ols)
            _zero_init_last_linear(self_.token_mlp)

        def forward(self_, inputs):
            oh = inputs[:, :K]
            cov = inputs[:, K:K + d_cov]
            return (self_.task_linear(oh) + self_.token_linear(cov)
                    + self_.token_mlp(cov) + self_.bias)

    class FullModel(nn.Module):
        def __init__(self_):
            super().__init__()
            in_dim = K + d_cov
            self_.linear_skip = nn.Linear(in_dim, D)
            self_.mlp = _make_mlp(in_dim, D)
            with torch.no_grad():
                W_skip = torch.cat([W_task_ols, W_cov_ols], dim=0)
                self_.linear_skip.weight.copy_(W_skip.T)
                self_.linear_skip.bias.copy_(b_ols)
            _zero_init_last_linear(self_.mlp)

        def forward(self_, inputs):
            return self_.linear_skip(inputs) + self_.mlp(inputs)

    X_combined = torch.cat([one_hot_k, cov_all], dim=1)
    X_tr = X_combined[idx_train]
    X_va = X_combined[idx_val]
    Y_tr = h_all[idx_train]
    Y_va = h_all[idx_val]

    cov_desc = "x,pos" if fit_position else "x"
    add_model = AdditiveModel()
    add_model = _train_model(
        add_model, X_tr, Y_tr, X_va, Y_va,
        model_name=f"additive(joint): W_task@oh + MLP({cov_desc}) + b",
    )

    full_model = FullModel()
    full_model = _train_model(
        full_model, X_tr, Y_tr, X_va, Y_va,
        model_name=f"full(joint): MLP(oh, {cov_desc})",
    )

    # ---- Per-position R² evaluation ----
    results: Dict[int, ANCOVAResult] = {}
    for pos in positions:
        s, e = pos_slices[pos]
        X_pos = X_combined[s:e]
        Y_pos = h_all[s:e]

        r2_add, ss_res_add, ss_tot_pos = _compute_r2(add_model, X_pos, Y_pos)
        r2_full, ss_res_full, _ = _compute_r2(full_model, X_pos, Y_pos)

        val_mask = (idx_val >= s) & (idx_val < e)
        if val_mask.any():
            val_idx_local = idx_val[val_mask] - s
            X_pos_val = X_pos[val_idx_local]
            Y_pos_val = Y_pos[val_idx_local]
            r2_add_val, _, _ = _compute_r2(add_model, X_pos_val, Y_pos_val)
            r2_full_val, _, _ = _compute_r2(full_model, X_pos_val, Y_pos_val)
        else:
            r2_add_val = r2_add
            r2_full_val = r2_full

        gap = r2_full - r2_add
        gap_val = r2_full_val - r2_add_val

        results[pos] = ANCOVAResult(
            r2_additive=r2_add,
            r2_full=r2_full,
            separability_gap=gap,
            ss_total=ss_tot_pos,
            ss_res_additive=ss_res_add,
            ss_res_full=ss_res_full,
            n_tasks=K,
            n_covariate_dims=d,
            n_samples=e - s,
            layer_num=layer_num,
            position=pos,
            r2_additive_val=r2_add_val,
            r2_full_val=r2_full_val,
            separability_gap_val=gap_val,
        )

    if verbose:
        for pos in positions:
            r = results[pos]
            print(f"  pos={pos:3d}: R²_add={r.r2_additive:.4f} "
                  f"R²_full={r.r2_full:.4f}  gap={r.separability_gap:.4f}  "
                  f"R²_add_v={r.r2_additive_val:.4f} "
                  f"R²_full_v={r.r2_full_val:.4f}  gap_v={r.separability_gap_val:.4f}")

    del add_model, full_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def mlp_ancova_separability_joint_from_hiddens(
    all_hiddens: torch.Tensor,
    demo_data: torch.Tensor,
    layers: "Optional[Sequence[int]]" = None,
    positions: "Optional[Sequence[int]]" = None,
    **mlp_kwargs,
) -> Dict[int, Dict[int, ANCOVAResult]]:
    """Run joint MLP ANCOVA across layers (one joint fit per layer).

    Parameters
    ----------
    all_hiddens : (L, n_tasks, n_points, batch_size, D)
    demo_data : (batch_size, n_points, n_dims)
    layers, positions : optional index lists
    **mlp_kwargs : forwarded to ``mlp_ancova_separability_joint``

    Returns
    -------
    dict  {layer_num: {position: ANCOVAResult}}
    """
    L = all_hiddens.shape[0]
    if layers is None:
        layers = list(range(L))

    results: Dict[int, Dict[int, ANCOVAResult]] = {}
    for l_idx, l_num in enumerate(layers):
        if l_idx >= L:
            continue
        results[l_num] = mlp_ancova_separability_joint(
            all_hiddens[l_idx],
            demo_data,
            positions=positions,
            layer_num=l_num,
            **mlp_kwargs,
        )
    return results


def ancova_separability_from_hiddens(
    all_hiddens: torch.Tensor,
    demo_data: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
    positions: Optional[Sequence[int]] = None,
    eps: float = 1e-10,
) -> Dict[int, Dict[int, ANCOVAResult]]:
    """Run ANCOVA across layers and positions for linear regression.

    Parameters
    ----------
    all_hiddens : torch.Tensor
        Shape ``(L, n_tasks, n_points, batch_size, D)``.
    demo_data : torch.Tensor
        Shape ``(batch_size, n_points, n_dims)``.
        Input vectors x_t (shared across tasks).
    layers : sequence of int, optional
        Layer numbers.  ``None`` → ``range(L)``.
    positions : sequence of int, optional
        Point indices to analyse.  ``None`` → all.
    eps : float

    Returns
    -------
    dict
        ``{layer_num: {position: ANCOVAResult, ...}, ...}``
    """
    L, n_tasks, n_points, B, D = all_hiddens.shape
    n_dims = demo_data.shape[-1]

    if layers is None:
        layers = list(range(L))
    if positions is None:
        positions = list(range(n_points))

    task_labels = (
        torch.arange(n_tasks)
        .unsqueeze(1)
        .expand(n_tasks, B)
        .reshape(-1)
    )

    results: Dict[int, Dict[int, ANCOVAResult]] = {}

    for l_idx, l_num in enumerate(layers):
        if l_idx >= L:
            continue
        layer_results: Dict[int, ANCOVAResult] = {}

        for pos in positions:
            if pos >= n_points:
                continue

            h = all_hiddens[l_idx, :, pos, :, :]  # (n_tasks, B, D)
            h_flat = h.reshape(n_tasks * B, D)

            x = demo_data[:, pos, :]  # (B, n_dims)
            x_flat = x.unsqueeze(0).expand(n_tasks, B, n_dims).reshape(n_tasks * B, n_dims)

            res = ancova_separability(h_flat, x_flat, task_labels, eps=eps)
            res.layer_num = l_num
            res.position = pos
            layer_results[pos] = res

        results[l_num] = layer_results

    return results


def plot_ancova_separability(
    results: Dict[int, Dict[int, ANCOVAResult]],
    figsize: tuple = (5, 3.2),
    log_x: bool = True,
    show: bool = True,
    show_ylabel: bool = True,
):
    """Plot ANCOVA interaction proportion and separability R².

    To match the interpretation of :func:`plot_anova_separability`, we define

        η²_interaction = (R²_full - R²_additive) / R²_full
        separability R² = R²_additive / R²_full = 1 - η²_interaction

    so that both ANOVA and ANCOVA plots share the same semantics:
    small η²_interaction (or large separability R²) means additivity holds.

    Returns
    -------
    (fig_interaction, fig_sep) : tuple of Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np  # noqa: F401

    layers = sorted(results.keys())
    if not layers:
        return None, None

    _COLORS = [
        "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
        "#56B4E9", "#F0E442", "#000000",
    ]
    _LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    def _style(i):
        return dict(
            color=_COLORS[i % len(_COLORS)],
            linestyle=_LINESTYLES[i % len(_LINESTYLES)],
            linewidth=2.2,
        )

    def _eta2_interaction(res: ANCOVAResult) -> float:
        if res.r2_full <= 0:
            return 0.0
        return res.separability_gap / res.r2_full

    # ---- Figure 1: η²_interaction ----
    fig1, ax1 = plt.subplots(figsize=figsize)
    pos_list = []
    for i, l_num in enumerate(layers):
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        if not pos_list:
            continue
        eta2_int = [_eta2_interaction(pos_results[p]) for p in pos_list]
        ax1.plot(pos_list, eta2_int, label=f"Layer {l_num}", **_style(i))

    ax1.set_xlabel("Position", fontsize=13)
    if show_ylabel:
        ax1.set_ylabel("$\\eta^2_{\\mathrm{interaction}}$", fontsize=13)
    if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
        ax1.set_xscale("symlog", linthresh=1)
    ax1.set_ylim(-0.02, None)
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=10, framealpha=0.9, loc="best",
               borderaxespad=0.3, handlelength=1.8)
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # ---- Figure 2: separability R² ----
    fig2, ax2 = plt.subplots(figsize=figsize)
    for i, l_num in enumerate(layers):
        pos_results = results[l_num]
        pos_list = sorted(pos_results.keys())
        if not pos_list:
            continue
        sep_r2 = [1.0 - _eta2_interaction(pos_results[p]) for p in pos_list]
        ax2.plot(pos_list, sep_r2, label=f"Layer {l_num}", **_style(i))

    ax2.set_xlabel("Position", fontsize=13)
    if show_ylabel:
        ax2.set_ylabel("Separability $R^2$", fontsize=13)
    if log_x and len(pos_list) > 1 and min(pos_list) >= 0:
        ax2.set_xscale("symlog", linthresh=1)
    ax2.set_ylim(None, 1.00)
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=10, framealpha=0.9, loc="best",
               borderaxespad=0.3, handlelength=1.8)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig2)

    return fig1, fig2


def print_ancova_summary(
    results: Dict[int, Dict[int, ANCOVAResult]],
    positions: Optional[Sequence[int]] = None,
):
    """Print a formatted ANCOVA summary table."""
    layers = sorted(results.keys())
    if not layers:
        return

    sample_layer = results[layers[0]]
    all_pos = sorted(sample_layer.keys())
    if positions is not None:
        all_pos = [p for p in all_pos if p in positions]

    sample_res = sample_layer[all_pos[0]] if all_pos else None
    has_val = sample_res is not None and sample_res.r2_additive_val is not None

    if has_val:
        header = (
            f"{'Layer':>6} {'Pos':>5} "
            f"{'R²_add':>8} {'R²_full':>8} {'gap':>8} "
            f"{'R²_add_v':>9} {'R²_ful_v':>9} {'gap_v':>8} "
            f"{'N':>7} {'K':>3} {'d':>3}"
        )
    else:
        header = (
            f"{'Layer':>6} {'Pos':>5} {'R²_add':>8} {'R²_full':>8} "
            f"{'gap':>8} {'N':>7} {'K':>3} {'d':>3}"
        )

    print("=" * len(header))
    print("  ANCOVA: slope homogeneity (additive separability)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for l_num in layers:
        pos_results = results[l_num]
        for pos in all_pos:
            if pos not in pos_results:
                continue
            r = pos_results[pos]
            if has_val:
                print(
                    f"{l_num:>6} {pos:>5} "
                    f"{r.r2_additive:>8.4f} {r.r2_full:>8.4f} "
                    f"{r.separability_gap:>8.4f} "
                    f"{r.r2_additive_val:>9.4f} {r.r2_full_val:>9.4f} "
                    f"{r.separability_gap_val:>8.4f} "
                    f"{r.n_samples:>7} "
                    f"{r.n_tasks:>3} {r.n_covariate_dims:>3}"
                )
            else:
                print(
                    f"{l_num:>6} {pos:>5} {r.r2_additive:>8.4f} {r.r2_full:>8.4f} "
                    f"{r.separability_gap:>8.4f} {r.n_samples:>7} "
                    f"{r.n_tasks:>3} {r.n_covariate_dims:>3}"
                )

    print("=" * len(header))
