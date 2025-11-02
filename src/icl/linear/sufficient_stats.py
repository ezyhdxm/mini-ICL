import torch
from icl.linear.linear_utils import extract_hidden


def estimate_lambda(
    task_vecs: torch.Tensor,
    task_vecs_over_all_time: torch.Tensor,
    chunk_size: int = 32,
):
    """
    Memory-friendly version using column-chunking over k.

    Args:
        task_vecs: (num_tasks, d)
        task_vecs_over_all_time: (k, seq_len, d)
        is_zero_mean: exclude last task vec from X and enforce sum-to-1 (same as your code)
        chunk_size: number of columns (k) to solve per batch

    Returns:
        lambdas: (k, seq_len, num_tasks)  [numpy]
    """
    eps = 1e-12
    device = task_vecs.device
    dtype = task_vecs.dtype

    task_vecs_over_all_time = task_vecs_over_all_time.to(device=device, dtype=dtype)

    k, seq_len, d = task_vecs_over_all_time.shape
    num_tasks = task_vecs.shape[0]

    # outputs on device; convert to numpy at the end
    lambdas = torch.zeros((k, seq_len, num_tasks), dtype=dtype)

    # Design matrix X: (d, num_tasks or num_tasks-1)
    X = task_vecs.T.contiguous()  # (d, num_tasks)
    X = X[:, :-1]  # (d, num_tasks-1)


    with torch.no_grad():
        for t in range(seq_len):
            # Y_full: (d, k)  -- do not clone to avoid extra memory; slice columns per chunk
            Y_full = task_vecs_over_all_time[:, t, :].T  # (d, k)

            # Process k-dimension in chunks
            for start in range(0, k, chunk_size):
                end = min(start + chunk_size, k)

                Y = Y_full[:, start:end]                    # (d, k_chunk)
                # Solve X W = Y for W, least squares (stable)
                # W: (num_tasks-1 or num_tasks, k_chunk)
                W = torch.linalg.lstsq(X, Y).solution
                residual = 1.0 - W.sum(dim=0, keepdim=True)           # (1, k_chunk)
                last_lambda = residual / num_tasks                    # (1, k_chunk)
                zero_pad = torch.zeros((1, W.shape[1]), device=device, dtype=dtype)
                lambda_full = torch.cat([W, zero_pad], dim=0) + last_lambda  # (num_tasks, k_chunk)
                k_chunk = lambda_full.shape[1]
                assert torch.allclose(lambda_full.sum(dim=0), torch.ones(k_chunk, device=device, dtype=dtype), atol=1e-6)

                # Store lambdas
                lambdas[start:end, t, :] = lambda_full.T.cpu()  # (k_chunk, num_tasks)

    return lambdas # (k, seq_len, num_tasks)


def get_sufficient_statistics_proj_fit(
    config,
    model,
    train_task,
    task_layer_index: int = 3,
    max_t: int = 20,
    l2_reg: float = 1e-4,  # ridge regularization; set to 0.0 for OLS
    A = None,  # precomputed A matrix for projection (if any
):
    """
    For each t = 1..T, fit the linear map
        theta_k^{(t)} ≈ cat[ X^T Y^{(k,t)}, vec(X^T X^{(t)}) ] @ Theta_t + b_t
    using EACH batch instance as its own training sample (no batch averaging).

    Returns per-t parameters and goodness-of-fit metrics.
    """
    STEP = 1008600  # arbitrary fixed step for sampling train_task
    train_task.batch_size = 1024
    batch_size = train_task.batch_size
    device = config.device
    dtype = torch.float32
    n_tasks = train_task.task_pool.shape[0]
    d = int(getattr(config.task, "n_dims", train_task.task_pool.shape[-1]))
    n_points = config.task.n_points
    n_embd = config.model.n_embd

    demo_data = train_task.sample_data(step=STEP)  # shape (batch_size, n_points, n_dims)

    task_pos = 3 * torch.arange(n_points, device=config.device) + 1
    
    # Repeat the demo data for the tasks in this chunk
    demo_data_repeated = demo_data.unsqueeze(0).expand(n_tasks, batch_size, n_points, -1)
    demo_data_repeated = demo_data_repeated.reshape(-1, n_points, demo_data.size(-1))  # shape (3*batch_size, n_points, n_dims)
    
    # Get the task-specific targets for this chunk
    demo_target = train_task.evaluate(demo_data, 
                                      train_task.task_pool.squeeze(-1).T, 
                                      step=STEP)  # (batch_size, n_points, 3)
    demo_target = demo_target.permute(2, 0, 1).reshape(-1, n_points)  # shape (3*batch_size, n_points)
    
    all_hiddens = extract_hidden(
        model=model,
        demo_data=demo_data_repeated,
        demo_target=demo_target,
        l=task_layer_index,
        task_pos=task_pos
    )  # shape (3*batch_size, max_t, n_embd)

    all_hiddens = all_hiddens.reshape(n_tasks, batch_size, -1, n_embd) # (3, batch_size, n_points, n_embd)
    task_vectors = all_hiddens - all_hiddens.mean(dim=0, keepdim=True) # (3, batch_size, n_points, n_embd)
    final_task_vector = task_vectors[:, :, -1, :].mean(dim=1)  # (3, n_embd)
    task_vectors = task_vectors.reshape(-1, n_points, n_embd)  # (3*batch_size, n_points, n_embd)
    task_vectors = task_vectors[:, :max_t, :]  # (3*batch_size, max_t, n_embd)

    del all_hiddens  # free memory
    torch.cuda.empty_cache()  # clear CUDA memory cache

    lambdas = estimate_lambda(final_task_vector, task_vectors, chunk_size=16)  # (3*batch_size, max_t, 3)

    del final_task_vector, task_vectors  # free memory
    torch.cuda.empty_cache()  # clear CUDA memory cache
    
    X = demo_data_repeated[:, :max_t, :].to(device=device, dtype=dtype)  # (3*batch_size, max_t, n_dims)
    Y_full = demo_target[:, :max_t].to(device=device, dtype=dtype)  # (3*batch_size, max_t)

    XtYt = torch.empty((3*batch_size, max_t, d), device=device, dtype=dtype)
    XtXt = torch.empty((3*batch_size, max_t, d, d), device=device, dtype=dtype)
    for t in range(max_t):
        Yt = Y_full[:, t]  # (3*batch_size, t)
        if t == 0:
            XtYt[:, t] = torch.einsum('bd,b->bd', X[:, t, :], Yt)  # (3*batch_size, d)
            XtXt[:, t, :, :] = torch.einsum('bi,bj->bij', X[:, t, :], X[:, t, :])
        else:
            XtYt[:, t] = (t*XtYt[:, t-1] + torch.einsum('bd,b->bd', X[:, t, :], Yt)) / (t+1) # (3*batch_size, d)
            XtXt[:, t, :, :] = (t*XtXt[:, t-1, :, :] + torch.einsum('bi,bj->bij', X[:, t, :], X[:, t, :])) / (t+1)
    
    triu_idx = torch.triu_indices(d, d, offset=0, device=device)
    XtXt_upper = XtXt[..., triu_idx[0], triu_idx[1]]  # shape: (3*batch_size, max_t, num_upper)
    XtXt_upper = XtXt_upper.view(3*batch_size, max_t, -1)
    del XtXt  # free memory
    torch.cuda.empty_cache()  # clear CUDA memory cache
    
    XtYt = XtYt[:, 1:] # (3*batch_size, (max_t-1), d)
    XtXt_upper = XtXt_upper[:, 1:]  # (3*batch_size, (max_t-1), num_upper)
    features = torch.cat([XtYt, XtXt_upper], dim=-1)  # (3*batch_size, (max_t-1), d + num_upper)

    del XtYt  # free memory
    del XtXt_upper  # free memory
    torch.cuda.empty_cache()  # clear CUDA memory cache

    features_aug = torch.cat([features, torch.ones_like(features[:, :, :1])], dim=-1)  # (3*batch_size, (max_t-1), d + num_upper + 1)
    lambdas = lambdas[:, 1:]
    #features_aug = features_aug.reshape(3*batch_size*(max_t-1), -1)
    #lambdas = lambdas[:, 1:].reshape(3*batch_size*(max_t-1), -1)  # (3*batch_size*(max_t-1), 3)


    # Compute the linear map
    if A is None:
        dA = features_aug.shape[-1]
        A = torch.empty((dA, max_t - 1, 3), device=device, dtype=dtype)
        for t in range(max_t - 1):
            if l2_reg > 0.0:
                reg_matrix = l2_reg * torch.eye(dA, device=device, dtype=dtype)
                A[:, t, :] = torch.linalg.solve(
                    features_aug[:, t, :].T @ features_aug[:, t, :] + reg_matrix,
                    features_aug[:, t, :].T @ lambdas[:, t, :].to(device=device)
                )
            else:
                A[:, t, :] = torch.linalg.lstsq(features_aug[:, t, :], lambdas[:, t, :].to(device=device), rcond=None).solution
        
        '''
            # Compute the linear map
            dA = features_aug.shape[-1]
            if l2_reg > 0.0:
                reg_matrix = l2_reg * torch.eye(dA, device=device, dtype=dtype)
                A = torch.linalg.solve(
                    features_aug.T @ features_aug + reg_matrix,
                    features_aug.T @ lambdas.to(device=device)
                ) # (d + num_upper + 1, 2)
            else:
                A = torch.linalg.lstsq(features_aug, lambdas.to(device=device), rcond=None).solution
        '''
    else:
        A = A.to(device=device, dtype=dtype)

    # Compute r2

    #features_aug = features_aug.reshape(3*batch_size, max_t-1, -1)  # (3*batch_size, max_t-1, d + num_upper + 1)
    #lambdas = lambdas.reshape(3*batch_size, max_t-1, -1)  # (3*batch_size, max_t-1, 2)

    lambdas_pred = torch.einsum('btn,ntd->btd', features_aug, A)  # (3*batch_size, max_t-1, d) 
    lambdas = lambdas.to(device=device)
    res = lambdas - lambdas_pred                            # (3*batch_size, max_t-1, 3) 
    ssr = (res**2).sum(dim=0)                         # (max_t-1, 2)
    sst = ((lambdas - lambdas.mean(dim=0, keepdim=True))**2).sum(dim=0).clamp_min(1e-12)  # (max_t-1, 3)
    r2_per_dim = 1.0 - ssr / sst                      # (max_t-1, 3)
    r2 = r2_per_dim.mean(dim=-1)                      # (max_t-1)
    
    del features_aug, lambdas, lambdas_pred  # free memory
    torch.cuda.empty_cache()

    results = {
        "r2": r2.cpu(),
        "A": A.cpu(),
        }
    
    return results




def get_sufficient_statistics_fit(
    config,
    model,
    train_task,
    task_layer_index: int = 3,
    max_t: int = 100,
    l2_reg: float = 1e-4,  # ridge regularization; set to 0.0 for OLS
):
    STEP = 1008600  # arbitrary fixed step for sampling train_task
    train_task.batch_size = 512
    batch_size = train_task.batch_size
    device = config.device
    dtype = torch.float32
    n_tasks = train_task.task_pool.shape[0]
    d = int(getattr(config.task, "n_dims", train_task.task_pool.shape[-1]))
    n_points = config.task.n_points
    n_embd = config.model.n_embd

    demo_data = train_task.sample_data(step=STEP)  # shape (batch_size, n_points, n_dims)

    # output_shape = (n_tasks, max_t, batch_size, n_embd)
    task_pos = 3 * torch.arange(max_t, device=config.device) + 1
    
    # Repeat the demo data for the tasks in this chunk
    demo_data_repeated = demo_data.unsqueeze(0).expand(n_tasks, batch_size, n_points, -1)
    demo_data_repeated = demo_data_repeated.reshape(-1, n_points, demo_data.size(-1))  # shape (3*batch_size, n_points, n_dims)
    
    # Get the task-specific targets for this chunk
    demo_target = train_task.evaluate(demo_data, 
                                      train_task.task_pool.squeeze(-1).T, 
                                      step=STEP)  # (batch_size, n_points, 3)
    demo_target = demo_target.permute(2, 0, 1).reshape(-1, n_points)  # shape (3*batch_size, n_points)
    
    all_hiddens = extract_hidden(
        model=model,
        demo_data=demo_data_repeated,
        demo_target=demo_target,
        l=task_layer_index,
        task_pos=task_pos
    )  # shape (3*batch_size, max_t, n_embd)

    all_hiddens = all_hiddens.reshape(n_tasks, batch_size, max_t, n_embd) # (3, batch_size, max_t, n_embd)
    task_vectors = all_hiddens - all_hiddens.mean(dim=0, keepdim=True) # (3, batch_size, max_t, n_embd)
    task_vectors = task_vectors.reshape(-1, max_t, n_embd)  # (3*batch_size, max_t, n_embd)


    del all_hiddens  # free memory
    torch.cuda.empty_cache()  # clear CUDA memory cache
    
    X = demo_data_repeated[:, :max_t, :].to(device=device, dtype=dtype)  # (3*batch_size, max_t, n_dims)
    Y_full = demo_target[:, :max_t].to(device=device, dtype=dtype)  # (3*batch_size, max_t)

    XtYt = torch.empty((3*batch_size, max_t, d), device=device, dtype=dtype)
    for t in range(max_t):
        Yt = Y_full[:, t]  # (3*batch_size, t)
        if t == 0:
            XtYt[:, t] = torch.einsum('bd,b->bd', X[:, t, :], Yt)  # (3*batch_size, d)
        else:
            XtYt[:, t] = (t*XtYt[:, t-1] + torch.einsum('bd,b->bd', X[:, t, :], Yt)) / (t+1) # (3*batch_size, d)
    
    XtYt = XtYt[:, 1:] # (3*batch_size, (max_t-1), d)
    task_vectors = task_vectors[:, 1:]  # (3*batch_size, (max_t-1), n_embd)
    tv_aug = torch.cat([task_vectors, torch.ones_like(task_vectors[:, :, :1])], dim=-1)  # (3*batch_size, (max_t-1), n_embd + 1)

    # Compute the linear map
    A = torch.empty((n_embd + 1, max_t - 1, d), device=device, dtype=dtype)
    for t in range(max_t - 1):
        if l2_reg > 0.0:
            reg_matrix = l2_reg * torch.eye(n_embd + 1, device=device, dtype=dtype)
            A[:, t, :] = torch.linalg.solve(
                tv_aug[:, t, :].T @ tv_aug[:, t, :] + reg_matrix,
                tv_aug[:, t, :].T @ XtYt[:, t, :]
            )
        else:
            A[:, t, :] = torch.linalg.lstsq(tv_aug[:, t, :], XtYt[:, t, :], rcond=None).solution

    # Compute r2

    XtYt_pred = torch.einsum('btn,ntd->btd', tv_aug, A)  # (3*batch_size, max_t-1, d) 
    res = XtYt - XtYt_pred                            # (3*batch_size, max_t-1, d) 
    ssr = (res**2).sum(dim=0)                         # (max_t-1, d)
    sst = ((XtYt - XtYt.mean(dim=0, keepdim=True))**2).sum(dim=0).clamp_min(1e-12)  # (max_t-1, d)
    r2_per_dim = 1.0 - ssr / sst                      # (max_t-1, d)
    r2 = r2_per_dim.mean(dim=-1)                      # (max_t-1)

    del XtYt, XtYt_pred, task_vectors  # free memory
    torch.cuda.empty_cache()

    val_data = train_task.sample_data(step=STEP + 1)
    val_target = train_task.evaluate(val_data,
                                     train_task.task_pool.squeeze(-1).T,
                                     step=STEP + 1)
    val_target = val_target.permute(2, 0, 1).reshape(-1, n_points)
    
    val_dat_repeated = val_data.unsqueeze(0).expand(3, batch_size, n_points, -1)
    val_dat_repeated = val_dat_repeated.reshape(-1, n_points, val_data.size(-1))

    val_hiddens = extract_hidden(
        model=model,
        demo_data=val_dat_repeated,
        demo_target=val_target,
        l=task_layer_index,
        task_pos=task_pos
    )

    val_hiddens = val_hiddens.reshape(3, batch_size, max_t, n_embd)  # (3, batch_size, max_t, n_embd)
    val_task_vectors = val_hiddens - val_hiddens.mean(dim=0, keepdim=True)  # (3, batch_size, max_t, n_embd)
    val_task_vectors = val_task_vectors.reshape(-1, max_t, n_embd) # (3*batch_size, max_t, n_embd)

    del val_hiddens
    torch.cuda.empty_cache()  # clear CUDA memory cache

    val_X = val_dat_repeated[:, :max_t, :].to(device=device, dtype=dtype)  # (3*batch_size, max_t, n_dims)
    val_Y_full = val_target[:, :max_t].to(device=device, dtype=dtype)  # (3*batch_size, max_t)

    val_XtYt = torch.empty((3*batch_size, max_t, d), device=device, dtype=dtype)
    for t in range(max_t):
        val_Yt = val_Y_full[:, t]
        if t == 0:
            val_XtYt[:, t] = torch.einsum('bd,b->bd', val_X[:, t, :], val_Yt)
        else:
            val_XtYt[:, t] = (t*val_XtYt[:, t-1] + torch.einsum('bd,b->bd', val_X[:, t, :], val_Yt)) / (t+1)
    val_XtYt = val_XtYt[:, 1:]  # (3*batch_size, max_t-1, d)
    val_task_vectors = val_task_vectors[:, 1:]  # (3*batch_size, max_t-1, n_embd)
    val_tv_aug = torch.cat([val_task_vectors, torch.ones_like(val_task_vectors[:, :, :1])], dim=-1)  # (3*batch_size, max_t-1, n_embd + 1)

    val_XtYt_pred = torch.einsum('btn,ntd->btd', val_tv_aug, A)  # (3*batch_size, max_t-1, d) 
    val_res = val_XtYt - val_XtYt_pred                           # (3*batch_size, max_t-1, d) 
    val_ssr = (val_res**2).sum(dim=0)                            # (max_t-1, d)
    val_sst = ((val_XtYt - val_XtYt.mean(dim=0, keepdim=True))**2).sum(dim=0).clamp_min(1e-12)  # (max_t-1, d)
    val_r2_per_dim = 1.0 - val_ssr / val_sst                     # (max_t-1, d)
    val_r2 = val_r2_per_dim.mean(dim=-1)                         # (max_t-1)

    val_mse = torch.mean(torch.sum((val_XtYt - val_XtYt_pred)**2, dim=-1), dim=0)    # (max_t-1,)

    results = {
        "r2": r2.cpu(),
        "val_mse": val_mse.cpu(),
        "val_r2": val_r2.cpu(),
        "A": A.cpu(),
        }
    
    return results



def make_nonconvergent_targets_from_X(
    X,                       # (B, T, D)
    device=None,
    block_first_len: int = 2,
    growth: float = 1.0,
    seed: int | None = None,
):
    """
    随机块化构造：每块抽随机单位向量 u_k 和随机符号 s_k，块内固定 w_t = s_k * u_k，
    令 Y_t = X_t @ w_t。指数增长的块长使 (X^T Y)/t 在块末端随机跳动，从而不收敛。

    返回：
      Y: (B, T) —— 作为 demo_target
    """
    B, T, D = X.shape
    device = X.device if device is None else device

    # 生成块边界（所有样本共用同样的时间分块）
    blocks = []
    L = max(1, int(block_first_len))
    start = 0
    while start < T:
        end = min(T, start + L)
        blocks.append((start, end))
        start = end
        L = int((L * growth) + 0.9999)  # 天花板取整

    # 随机源
    # 说明：Generator 只能用于 CPU 随机；若你需要严格可复现，可先在 CPU 采样再 .to(device)
    gen = torch.Generator(device='cpu')
    if seed is not None:
        gen.manual_seed(seed)

    # 为每个块、每个样本生成随机单位向量 u_k 和随机符号 s_k
    K = len(blocks)
    u = torch.randn(B, K, D, generator=gen, device='cpu')
    u = u / (u.norm(dim=-1, keepdim=True).clamp_min(1e-12))  # 单位向量
    s = torch.randint(0, 2, (B, K, 1), generator=gen, device='cpu').float() * 2 - 1  # ∈{-1, +1}

    u = u.to(device)
    s = s.to(device)

    # 组装 W: (B, T, D)，块内固定 w_t = s_k * u_k
    W = torch.zeros(B, T, D, device=device, dtype=X.dtype)
    for k, (a, b) in enumerate(blocks):
        W[:, a:b, :] = (s[:, k, :] * u[:, k, :]).unsqueeze(1)  # (B,1,D) 广播到 (B, b-a, D)

    # 生成 Y: (B, T), 每个时间步 Y_t = <X_t, w_t>
    Y = (X * W).sum(dim=-1)
    return Y


def get_betahat_fit(
    config,
    model,
    train_task,
    task_layer_index: int = 3,
    max_t: int = 40,
    l2_reg: float = 1e-4,  # ridge regularization; set to 0.0 for OLS
):
    # print("This is the new implementation of betahat fitting.")
    STEP = 1008600  # arbitrary fixed step for sampling train_task
    train_task.batch_size = 2
    batch_size = train_task.batch_size
    device = config.device
    dtype = torch.float32
    n_tasks = train_task.task_pool.shape[0]
    d = int(getattr(config.task, "n_dims", train_task.task_pool.shape[-1]))
    n_points = config.task.n_points
    n_embd = config.model.n_embd

    demo_data = train_task.sample_data(step=STEP)  # shape (batch_size, n_points, n_dims)

    # output_shape = (n_tasks, max_t, batch_size, n_embd)
    task_pos = 3 * torch.arange(max_t, device=config.device) + 1
    
    # Repeat the demo data for the tasks in this chunk
    demo_data_repeated = demo_data.unsqueeze(0).expand(n_tasks, batch_size, n_points, -1)
    demo_data_repeated = demo_data_repeated.reshape(-1, n_points, demo_data.size(-1))  # shape (n_tasks*batch_size, n_points, n_dims)
    
    # Get the task-specific targets for this chunk
    # demo_target = # train_task.evaluate(demo_data, 
                                      # train_task.task_pool.squeeze(-1).T, 
                                      # step=STEP)  # (batch_size, n_points, n_tasks)

    demo_target = make_nonconvergent_targets_from_X(
        demo_data_repeated.to(device=device, dtype=dtype),
        device=device,
        block_first_len=2,   # 可调：首块长度
        growth=2.0,          # 可调：块长增长倍率（>1，推荐 2）
        seed=0,              # 可选：设定为 None 则每次不同
    ) # (n_tasks*batch_size, n_points)

    # demo_target = demo_target.permute(2, 0, 1).reshape(-1, n_points)  # shape (n_tasks*batch_size, n_points)
    
    all_hiddens = extract_hidden(
        model=model,
        demo_data=demo_data_repeated,
        demo_target=demo_target,
        l=task_layer_index,
        task_pos=task_pos
    )  # shape (n_tasks*batch_size, max_t, n_embd)

    all_hiddens = all_hiddens.reshape(n_tasks, batch_size, max_t, n_embd) # (n_tasks, batch_size, max_t, n_embd)
    task_vectors = all_hiddens - all_hiddens.mean(dim=0, keepdim=True) # (n_tasks, batch_size, max_t, n_embd)
    task_vectors = task_vectors.reshape(-1, max_t, n_embd)  # (n_tasks*batch_size, max_t, n_embd)


    del all_hiddens  # free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # clear CUDA memory cache
    
    X = demo_data_repeated[:, :max_t, :].to(device=device, dtype=dtype)  # (n_tasks*batch_size, max_t, n_dims)
    Y_full = demo_target[:, :max_t].to(device=device, dtype=dtype)  # (n_tasks*batch_size, max_t)

    XtYt = torch.empty((n_tasks*batch_size, max_t, d), device=device, dtype=dtype)
    # XtXt = torch.empty((n_tasks*batch_size, max_t, d, d), device=device, dtype=dtype)
    betat = torch.empty((n_tasks*batch_size, max_t, d), device=device, dtype=dtype)
    for t in range(max_t):
        Yt = Y_full[:, t]  # (n_tasks*batch_size,)
        if t == 0:
            XtYt[:, t] = torch.einsum('bd,b->bd', X[:, t, :], Yt)  # (n_tasks*batch_size, d)
            # XtXt[:, t, :, :] = torch.einsum('bi,bj->bij', X[:, t, :], X[:, t, :])
        else:
            XtYt[:, t] = (t*XtYt[:, t-1] + torch.einsum('bd,b->bd', X[:, t, :], Yt)) / (t+1) # (n_tasks*batch_size, d)
            # XtXt[:, t, :, :] = (t*XtXt[:, t-1, :, :] + torch.einsum('bi,bj->bij', X[:, t, :], X[:, t, :])) / (t+1)
        betat[:, t, :] = XtYt[:, t, :] # torch.linalg.lstsq(XtXt[:, t, :, :], XtYt[:, t, :], rcond=None).solution
    
    betat = betat[:, 1:] # (n_tasks*batch_size, (max_t-1), d)
    # print(betat[:, 10])
    task_vectors = task_vectors[:, 1:]  # (n_tasks*batch_size, (max_t-1), n_embd)
    tv_aug = torch.cat([task_vectors, torch.ones_like(task_vectors[:, :, :1])], dim=-1)  # (n_tasks*batch_size, (max_t-1), n_embd + 1)

    # Compute the linear map
    A = torch.empty((n_embd + 1, max_t - 1, d), device=device, dtype=dtype)
    for t in range(max_t - 1):
        if l2_reg > 0.0:
            reg_matrix = l2_reg * torch.eye(n_embd + 1, device=device, dtype=dtype)
            A[:, t, :] = torch.linalg.solve(
                tv_aug[:, t, :].T @ tv_aug[:, t, :] + reg_matrix,
                tv_aug[:, t, :].T @ betat[:, t, :]
            )
        else:
            A[:, t, :] = torch.linalg.lstsq(tv_aug[:, t, :], betat[:, t, :], rcond=None).solution

    # Compute r2

    betat_pred = torch.einsum('btn,ntd->btd', tv_aug, A)  # (n_tasks*batch_size, max_t-1, d) 
    res = betat - betat_pred                            # (n_tasks*batch_size, max_t-1, d) 
    ssr = (res**2).sum(dim=0)                         # (max_t-1, d)
    sst = ((betat - betat.mean(dim=0, keepdim=True))**2).sum(dim=0).clamp_min(1e-12)  # (max_t-1, d)
    r2_per_dim = 1.0 - ssr / sst                      # (max_t-1, d)
    r2 = r2_per_dim.mean(dim=-1)                      # (max_t-1)


    results = {
        "r2": r2.cpu(),
        "A": A.cpu(),
        }
    
    return results





