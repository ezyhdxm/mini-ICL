import numpy as np
import torch
from torch import nn

########################
# Linear Algebra Utils #
#########################

def effective_rank(A):
    """
    Compute the effective rank (also called the Shannon entropy rank) of a matrix.
    
    The effective rank measures the dimensionality of a matrix based on the entropy
    of its singular values. It ranges from 1 to the actual rank of the matrix.
    This is useful for understanding the dimensionality of data representations.
    
    Reference: https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf
    
    Args:
        A: Input matrix of shape (m, n)
    
    Returns:
        Effective rank as a float, exp(entropy) where entropy is the Shannon entropy
        of the normalized singular values
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p + 1e-12))  # Add small epsilon to avoid log(0)
    return np.exp(entropy)

def stable_rank(A):
    """
    Compute the stable rank of a matrix.
    
    The stable rank is the ratio of the squared Frobenius norm to the squared spectral norm.
    It provides a robust measure of the dimensionality of a matrix and is always bounded
    by the actual rank. Unlike the rank, it's stable under small perturbations.
    
    Args:
        A: Input matrix of shape (m, n)
    
    Returns:
        Stable rank as a float, (||A||_F^2) / (||A||_2^2)
    """
    frob_norm = np.linalg.norm(A, ord='fro')
    op_norm = np.linalg.norm(A, ord=2)  # spectral norm
    return (frob_norm ** 2) / (op_norm ** 2)


def get_stationary(P: torch.Tensor)->torch.Tensor:
    """
    Compute the stationary distribution of Markov chain transition matrices.
    
    Uses SVD to find the stationary distribution by finding the right singular vector
    (eigenvector of the transition matrix corresponding to eigenvalue 1).
    The method works by computing the nullspace of (P - I).
    
    Args:
        P: Transition matrix/matrices. Can be:
           - 2D tensor of shape (num_states, num_states) for a single transition matrix
           - 3D tensor of shape (num_samples, num_states, num_states_order) for batch
    
    Returns:
        mu: Stationary distribution(s). Shape depends on input:
           - If input was 2D: (num_states,)
           - If input was 3D: (num_samples, num_states)
        The distribution is normalized to sum to 1.
    """
    if P.ndim == 2:
        P = P.unsqueeze(0)
    assert P.ndim == 3, "P should be a 3D tensor"
    P = P.transpose(1, 2)  # Transpose each matrix, Shape: (num_samples, num_states, num_states_order)
    num_states = P.shape[1]
    svd_input = P - torch.eye(num_states, device=P.device).unsqueeze(0)
    _, _, v = torch.linalg.svd(svd_input)
    mu = torch.abs(v[:, -1, :])  # Last singular vector for each matrix, Shape: (num_samples, num_states)
    mu = mu / mu.sum(dim=-1, keepdim=True)  # Normalize
    if mu.size(0) == 1:
        mu = mu.squeeze(0)
    return mu

def project_points_to_plane(points, anchors):
    """
    Project points onto a 2D plane defined by three anchor points.
    
    Constructs an orthonormal basis from the three anchor points using Gram-Schmidt
    orthogonalization, then projects all points onto this 2D plane and returns
    their 2D coordinates.
    
    Args:
        points: Points to project. Can be:
                - Single point: array of shape (d,) where d is the dimension
                - Multiple points: array of shape (N, d)
        anchors: Three anchor points defining the plane. Can be:
                 - array of shape (3, d) or (d, 3)
    
    Returns:
        coords: 2D coordinates of projected points. Shape (N, 2) where N is the number of points
    """
    P = np.asarray(points)
    A = np.asarray(anchors)
    if A.shape[0] == 3:
        pass
    elif A.shape[1] == 3:
        A = A.T
    if P.ndim == 1:
        P = P[None, :]
    p1, p2, p3 = A[0], A[1], A[2]
    u = p2 - p1
    v = p3 - p1
    u_norm = np.linalg.norm(u)
    e1 = u / u_norm
    v_ortho = v - e1 * np.dot(v, e1)
    v_ortho_norm = np.linalg.norm(v_ortho)
    e2 = v_ortho / v_ortho_norm

    E = np.vstack([e1, e2])       # (2, d)
    Proj = E.T @ E                # (d, d)

    Q_shift = P - p1              # (N, d)
    x = Q_shift @ e1              # (N,)
    y = Q_shift @ e2              # (N,)
    coords = np.stack([x, y], axis=1)  # (N, 2)
    return coords


def project_onto_cellmeans(
    cell_means: torch.Tensor,
    hiddens: torch.Tensor,
    token_ids: torch.Tensor,
    project_simplex: bool = True,
) -> np.ndarray:
    """Decompose hidden states onto task-token cell means conditioned on token.

    For each sample (b, t) with observed token a = token_ids[b, t],
    solve for lambda such that  h_{b,t} ≈ sum_k lambda_k theta_{k,a}
    under the affine constraint sum_k lambda_k = 1.

    Concretely, for token a define the (K-1) x D difference matrix
        D_a[k, :] = theta_{k,a} - theta_{K,a},   k = 1, ..., K-1
    and solve
        lambda_{1:K-1} = (D_a D_a^T)^{-1} D_a (h - theta_{K,a})
        lambda_K       = 1 - sum_{k<K} lambda_k

    Parameters
    ----------
    cell_means : (n_tokens, K, D) tensor
        Estimated task-token vectors theta_{a,k}.
    hiddens : (B, T, D) tensor
        Hidden states to decompose.
    token_ids : (B, T) long tensor
        Observed token index at each (batch, position).
    project_simplex : bool
        If True, project the raw lambda onto the probability simplex.

    Returns
    -------
    lambdas : (B, T, K) numpy array
    """
    n_tokens, K, D = cell_means.shape
    B, T, _ = hiddens.shape
    device = cell_means.device
    dtype = cell_means.dtype

    hiddens = hiddens.to(device=device, dtype=dtype)
    token_ids = token_ids.to(device=device)

    lambdas = torch.zeros((B, T, K), dtype=dtype, device=device)

    # Pre-compute per-token difference matrices and their pseudo-inverses
    diff_matrices = {}  # token_val -> (D_a, pinv)  where D_a is (K-1, D)
    for a in range(n_tokens):
        theta_a = cell_means[a]  # (K, D)
        anchor = theta_a[-1]     # theta_{K,a}
        D_a = theta_a[:-1] - anchor  # (K-1, D)
        pinv = torch.linalg.pinv(D_a)  # (D, K-1)
        diff_matrices[a] = (anchor, D_a, pinv)

    with torch.no_grad():
        for a, (anchor, D_a, pinv) in diff_matrices.items():
            mask = (token_ids == a)  # (B, T)
            if not mask.any():
                continue
            indices = mask.nonzero(as_tuple=False)  # (N, 2)
            h_sel = hiddens[indices[:, 0], indices[:, 1]]  # (N, D)
            residual = h_sel - anchor.unsqueeze(0)  # (N, D)
            lam_short = residual @ pinv  # (N, K-1)
            lam_last = 1.0 - lam_short.sum(dim=1, keepdim=True)  # (N, 1)
            lam_full = torch.cat([lam_short, lam_last], dim=1)   # (N, K)
            lambdas[indices[:, 0], indices[:, 1]] = lam_full

    lam_np = lambdas.cpu().numpy()

    if project_simplex:
        flat = lam_np.reshape(-1, K)
        flat = _project_onto_simplex_np(flat)
        lam_np = flat.reshape(B, T, K)

    return lam_np


def _project_onto_simplex_np(v: np.ndarray) -> np.ndarray:
    """Project each row of v onto the probability simplex."""
    if v.ndim == 1:
        v = v[np.newaxis, :]
    n = v.shape[1]
    u = np.sort(v, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    rho = np.zeros(v.shape[0], dtype=int)
    for i in range(v.shape[0]):
        conds = u[i] - cssv[i] / np.arange(1, n + 1) > 0
        rho[i] = np.where(conds)[0][-1]
    theta = cssv[np.arange(v.shape[0]), rho] / (rho + 1.0)
    return np.maximum(v - theta[:, np.newaxis], 0.0)


def estimate_lambda_with_r2(
    task_vecs: torch.Tensor,
    task_vecs_over_all_time: torch.Tensor,
    is_zero_mean: bool = True,
    chunk_size: int = 32,
):
    """Express each target vector as a mixture of reference task vectors via OLS.

    Setup
    -----
    Reference vectors  w_1, ..., w_K in R^d  (rows of *task_vecs*) form the
    columns of the design matrix  X = [w_1 | ... | w_K] in R^{d x K}.

    For each evaluation index j and position t, the target vector is
    v_{j,t} in R^d  (``task_vecs_over_all_time[j, t, :]``).

    Unconstrained mode  (is_zero_mean=False)
    -----------------------------------------
    Solve the d-dimensional OLS problem treating the d coordinates as
    "observations" and the K reference vectors as "regressors"::

        lambda_hat = argmin_lam  ||v - X lam||^2  =  (X^T X)^+ X^T v

    Sum-to-one mode  (is_zero_mean=True)
    -------------------------------------
    When the reference vectors are centred, the last column is redundant.
    We drop it, solve OLS for lam_1 ... lam_{K-1}, then redistribute
    the residual mass so that sum(lam) = 1.

    Parameters
    ----------
    task_vecs : (K, d) tensor -- reference task vectors.
    task_vecs_over_all_time : (k, seq_len, d) tensor -- target vectors.
    is_zero_mean : bool -- if True, enforce sum(lam) = 1.
    chunk_size : int -- number of target vectors solved per batch.

    Returns
    -------
    lambdas     : (k, seq_len, K)  numpy
    r2_scores   : (k, seq_len)     numpy
    task_norms  : (k, seq_len)     numpy
    ortho_norms : (k, seq_len)     numpy
    """
    eps = 1e-12
    device = task_vecs.device
    dtype = task_vecs.dtype

    task_vecs_over_all_time = task_vecs_over_all_time.to(device=device, dtype=dtype)

    k, seq_len, d = task_vecs_over_all_time.shape
    num_tasks = task_vecs.shape[0]

    lambdas = torch.zeros((k, seq_len, num_tasks), dtype=dtype)
    r2_scores = torch.zeros((k, seq_len), dtype=dtype)
    task_norms = torch.zeros((k, seq_len), dtype=dtype)
    ortho_norms = torch.zeros((k, seq_len), dtype=dtype)

    X = task_vecs.T.contiguous()  # (d, K)
    X_design = X[:, :-1] if is_zero_mean else X  # (d, K-1) or (d, K)

    with torch.no_grad():
        for start in range(0, k, chunk_size):
            end = min(start + chunk_size, k)
            k_chunk = end - start

            # Fold seq_len into the RHS columns so one lstsq covers all positions.
            # (k_chunk, seq_len, d) → (d, k_chunk, seq_len) → (d, k_chunk*seq_len)
            Y_flat = (
                task_vecs_over_all_time[start:end]
                .permute(2, 0, 1)
                .reshape(d, k_chunk * seq_len)
                .contiguous()
            )

            W_flat = torch.linalg.lstsq(X_design, Y_flat).solution  # (K-1, k_chunk*seq_len)

            if is_zero_mean:
                lambda_full = torch.zeros((num_tasks, k_chunk * seq_len), device=device, dtype=dtype)
                lambda_full[:-1, :] = W_flat
                lambda_full += (1.0 - W_flat.sum(dim=0, keepdim=True)) / num_tasks
            else:
                lambda_full = W_flat  # (K, k_chunk*seq_len)

            # (num_tasks, k_chunk*seq_len) → (num_tasks, k_chunk, seq_len) → (k_chunk, seq_len, num_tasks)
            lambdas[start:end, :, :] = (
                lambda_full.reshape(num_tasks, k_chunk, seq_len).permute(1, 2, 0).cpu()
            )

            y_pred_flat = X_design @ W_flat   # (d, k_chunk*seq_len)
            resid_flat  = Y_flat - y_pred_flat

            task_norms[start:end, :]  = y_pred_flat.norm(dim=0).reshape(k_chunk, seq_len).cpu()
            ortho_norms[start:end, :] = resid_flat.norm(dim=0).reshape(k_chunk, seq_len).cpu()

            ss_res = (resid_flat * resid_flat).sum(dim=0)        # (k_chunk*seq_len,)
            y_mean = Y_flat.mean(dim=0, keepdim=True)            # (1, k_chunk*seq_len)
            ss_tot = ((Y_flat - y_mean) ** 2).sum(dim=0)         # (k_chunk*seq_len,)
            r2 = 1.0 - ss_res / (ss_tot + eps)
            r2.clamp_(min=0.0, max=1.0)
            r2_scores[start:end, :] = r2.reshape(k_chunk, seq_len).cpu()

    return lambdas.numpy(), r2_scores.numpy(), task_norms.numpy(), ortho_norms.numpy()


def estimate_lambda_with_r2_batched_B(
    task_vecs: torch.Tensor,
    task_vecs_over_all_time: torch.Tensor,
    is_zero_mean: bool = True,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Same OLS / R² as ``estimate_lambda_with_r2``, but batched over leading dim ``B``.

    Parameters
    ----------
    task_vecs : (B, K, d)
        Reference task vectors per batch (same layout as ``estimate_lambda_with_r2``).
    task_vecs_over_all_time : (B, k, seq_len, d)
        Target vectors per batch.

    Returns
    -------
    torch.Tensor
        ``r2_scores`` of shape ``(B, k, seq_len)`` (same device/dtype as inputs).
    """
    eps = 1e-12
    device = task_vecs.device
    dtype = task_vecs.dtype
    tvot = task_vecs_over_all_time.to(device=device, dtype=dtype)

    B, K, d = task_vecs.shape
    B2, k_targets, seq_len, d2 = tvot.shape
    if B != B2 or d != d2:
        raise ValueError(
            f"Shape mismatch: task_vecs {task_vecs.shape}, "
            f"task_vecs_over_all_time {tvot.shape}"
        )

    r2_scores = torch.zeros((B, k_targets, seq_len), dtype=dtype, device=device)

    X_full = task_vecs.transpose(1, 2).contiguous()  # (B, d, K)
    if is_zero_mean:
        X_design = X_full[:, :, :-1]  # (B, d, K-1)
    else:
        X_design = X_full

    with torch.no_grad():
        for start in range(0, k_targets, chunk_size):
            end = min(start + chunk_size, k_targets)
            k_chunk = end - start

            # Fold seq_len into the RHS columns so one lstsq covers all positions.
            # (B, k_chunk, seq_len, d) → (B, d, k_chunk, seq_len) → (B, d, k_chunk*seq_len)
            Y_flat = (
                tvot[:, start:end, :, :]
                .permute(0, 3, 1, 2)
                .reshape(B, d, k_chunk * seq_len)
                .contiguous()
            )

            W_flat = torch.linalg.lstsq(X_design, Y_flat).solution  # (B, K-1, k_chunk*seq_len)

            y_pred_flat = torch.bmm(X_design, W_flat)   # (B, d, k_chunk*seq_len)
            resid_flat  = Y_flat - y_pred_flat

            ss_res = (resid_flat * resid_flat).sum(dim=1)          # (B, k_chunk*seq_len)
            y_mean = Y_flat.mean(dim=1, keepdim=True)              # (B, 1, k_chunk*seq_len)
            ss_tot = ((Y_flat - y_mean) ** 2).sum(dim=1)           # (B, k_chunk*seq_len)
            r2 = 1.0 - ss_res / (ss_tot + eps)
            r2.clamp_(min=0.0, max=1.0)
            r2_scores[:, start:end, :] = r2.reshape(B, k_chunk, seq_len)

    return r2_scores