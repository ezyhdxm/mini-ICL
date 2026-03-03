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

    X = task_vecs.T.contiguous()  # (d, K)
    if is_zero_mean:
        X = X[:, :-1]

    task_norms = torch.zeros((k, seq_len), dtype=dtype)
    ortho_norms = torch.zeros((k, seq_len), dtype=dtype)

    with torch.no_grad():
        for t in range(seq_len):
            Y_full = task_vecs_over_all_time[:, t, :].T  # (d, k)

            for start in range(0, k, chunk_size):
                end = min(start + chunk_size, k)
                Y = Y_full[:, start:end]  # (d, k_chunk)

                W = torch.linalg.lstsq(X, Y).solution

                if is_zero_mean:
                    lambda_full = torch.zeros((num_tasks, W.shape[1]), device=device, dtype=dtype)
                    lambda_full[:-1, :] = W
                    shift = (1.0 - W.sum(dim=0, keepdim=True)) / num_tasks
                    lambda_full += shift
                else:
                    lambda_full = W

                lambdas[start:end, t, :] = lambda_full.T.cpu()

                y_pred = X @ W
                resid = Y - y_pred
                ss_res = (resid * resid).sum(dim=0)

                task_norms[start:end, t] = torch.norm(y_pred, dim=0).cpu()
                ortho_norms[start:end, t] = torch.norm(resid, dim=0).cpu()

                y_mean = Y.mean(dim=0, keepdim=True)
                ss_tot = ((Y - y_mean) ** 2).sum(dim=0)
                r2 = 1.0 - ss_res / (ss_tot + eps)
                r2.clamp_(min=0.0, max=1.0)
                r2_scores[start:end, t] = r2.cpu()

    return lambdas.numpy(), r2_scores.numpy(), task_norms.numpy(), ortho_norms.numpy()