import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Any, Optional, Tuple
from math import log, exp, sqrt

from icl.linear.lr_config import *
from icl.linear.lr_transformer import GPT2Model, GPT2Config
from icl.linear.lr_utils import to_seq, seq_to_targets


########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################


def get_model_name(model):
    if isinstance(model, Ridge):
        return "Ridge"
    elif isinstance(model, DiscreteMMSE):
        return "dMMSE"
    elif isinstance(model, TransformerLin):
        return "Transformer"
    elif isinstance(model, Mixture):
        return "Mixture"
    elif isinstance(model, MixedRidge):
        return "MixedRidge"
    elif isinstance(model, UnbalancedMMSE):
        return "UnbalancedMMSE"
    else:
        raise ValueError(f"model type={type(model)} not supported")


########################################################################################################################
# Transformer                                                                                                          #
########################################################################################################################



class TransformerLin(nn.Module):
    def __init__(self, n_dims: int, n_points: int, n_layer: int, n_embd: int, n_head: int, seed: int, dtype: Any, pad: str = "bos", activation: Optional[str] = None):
        super().__init__()
        self.n_points = n_points
        self.dtype = dtype
        self.pad = pad
        self.input_dim = n_dims+1

        # GPT-style config (assuming your custom GPT2Model/GPT2Config implementation)
        config = GPT2Config(
            block_size=2 * n_points,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dtype=dtype,
            device= "cuda" if torch.cuda.is_available() else "cpu",
            activation=activation if activation else "gelu",
        )

        torch.manual_seed(seed)

        self.input_proj = nn.Linear(self.input_dim, config.n_embd, bias=False).to(device=config.device)
        self.transformer = GPT2Model(config)
        self.output_proj = nn.Linear(config.n_embd, 1, bias=False).to(device=config.device)
        self.device = config.device

        self.pad_token = nn.Parameter(torch.zeros(1, 1, self.input_dim, dtype=dtype, device=config.device))
        self.to(self.device)


    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert input into sequential format
        seq = to_seq(data, targets).to(device=self.device)  # shape: (batch, seq_len=2*n_points, input_dim) if bos
        # Prepend BOS token: shape (B, 1, D+1)
        B, T = seq.shape[0], seq.shape[1]
        if self.pad == "bos":
            bos = self.pad_token.expand(B, 1, self.input_dim)  # shape (B, 1, D+1)
            input_seq = torch.cat([bos, seq], dim=1)  # shape (B, seq_len+1, D+1)
        elif self.pad == "mapsto":
            input_seq = torch.zeros((B, T + T//2, self.input_dim), dtype=self.dtype, device=self.device)  # shape (B, 2seq_len, D+1)
            bos = self.pad_token.expand(B, T//2, self.input_dim)  # shape (B, 2seq_len, D+1)
            input_seq[:, 1::3, :] = bos  
            input_seq[:, 0::3, :] = seq[:, 0::2, :] 
            input_seq[:, 2::3, :] = seq[:, 1::2, :]
        else:
            raise ValueError(f"pad={self.pad} not supported. Use 'bos' or 'mapsto'.")

        # Project to embedding space
        embds = self.input_proj(input_seq)  

        # Pass through transformer
        outputs = self.transformer(embds)  

        # Project to output
        preds = self.output_proj(outputs)  

        # Remove BOS before passing to seq_to_targets
        if self.pad == "bos":
            preds = preds[:, 1:, :] 
            preds = seq_to_targets(preds)  # shape: (batch, n_points)
        elif self.pad == "mapsto":
            preds = preds[:, 1::3, 0]  
        
        return preds


########################################################################################################################
# Mixture                                                                                                                #
########################################################################################################################

class Mixture(nn.Module):
    def __init__(self, tau: float, task_pool: torch.Tensor, p0: float, noise_scale: float, dtype=torch.float32):
        """
        Args:
            scale: noise std
            task_pool: Tensor of shape (n_tasks, n_dims, 1)
        """
        super().__init__()
        self.tau = tau
        self.noise_scale = noise_scale
        self.noise_var = noise_scale ** 2
        self.minor_scale = noise_scale / tau
        self.minor_var = self.minor_scale ** 2
        self.dtype = dtype
        self.p0 = p0

        assert task_pool.ndim == 3 and task_pool.shape[2] == 1
        self.task_pool = task_pool.to(dtype)

        # Preprocess: squeeze and transpose for use in prediction
        self.W = task_pool.squeeze(-1).T  # shape: (n_dims, n_tasks)
        if self.dtype == torch.float16: self.W = self.W.to("cuda")
        
    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: (batch_size, n_points, n_dims)
            targets: (batch_size, n_points)
        Returns:
            preds: (batch_size, n_points)
        """
        if data.ndim == 4: data = data.squeeze(0)
        if targets.ndim == 3: targets = targets.squeeze(0)
        device = data.device
        if data.dtype == torch.float16:
            data = data.to("cuda")
            targets = targets.to("cuda")
        batch_size, n_points, _ = data.shape
        targets = targets.unsqueeze(-1)  # (batch_size, n_points, 1)

        preds = []  # list of (batch_size,)
        preds_i = self.predict(torch.zeros_like(data[:, :1]), torch.zeros_like(targets[:, :1]), data[:, 0:1]).to(device)
        preds.append(preds_i)  # initial prediction with zero input

        # Iterative MMSE predictions
        for i in range(1, n_points):
            pred_i = self.predict(data[:, :i], targets[:, :i], data[:, i:i+1]).to(device)
            preds.append(pred_i)

        return torch.stack(preds, dim=1)  # (batch_size, n_points)

    def predict(self, X: torch.Tensor, Y: torch.Tensor, test_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch_size, t, n_dims)
            Y: (batch_size, t, 1)
            test_x: (batch_size, 1, n_dims)
        Returns:
            (batch_size,)
        """
        assert X.ndim == 3 and Y.ndim == 3 and test_x.ndim == 3, "X, Y, and test_x must be 3D tensors"
        batch_size, t, n_dims = X.shape
        lam = self.tau ** 2

        XT = X.transpose(1, 2)                          # (batch_size, n_dims, t)
        XT_Y = torch.bmm(XT, Y)                         # (batch_size, n_dims, 1)

        # Ridge matrix: X^T X + λI
        eye = torch.eye(n_dims, dtype=self.dtype, device=X.device).unsqueeze(0).expand(batch_size, -1, -1)
        ridge_matrix = torch.bmm(XT, X) + lam * eye     # (batch_size, n_dims, n_dims)
        A = torch.linalg.inv(ridge_matrix)  # (batch_size, n_dims, n_dims)
        
        bs = XT_Y + lam * self.W.unsqueeze(0)  # (batch_size, n_dims, 1) + (1, n_dims, n_tasks) -> (batch_size, n_dims, n_tasks)

        log_delta_weights = self._get_delta_weights(Y, X)  # (batch_size, n_tasks)
        log_gauss_weights = self._get_gaussian_weights(A, Y, bs) # (batch_size, n_tasks)
        log_weights = torch.cat([log_delta_weights, log_gauss_weights], dim=1)  # (batch_size, 2*n_tasks)
        weights = torch.softmax(log_weights, dim=-1)  # normalize, (batch_size, 2*n_tasks)
        
        locs_gauss_weight = torch.bmm(A, bs) # (batch_size, n_dims, n_tasks)

        locs = torch.cat([self.W.unsqueeze(0).expand(batch_size, -1, -1), locs_gauss_weight], dim=2)  # (batch_size, n_dims, 2*n_tasks)
        ws = torch.einsum('bik,bk->bi', locs, weights)  # (batch_size, n_dims)
        pred = torch.bmm(test_x, ws.unsqueeze(-1)) # (batch_size, 1, 1)

        return pred[:,0,0] # (batch_size,)
    
    def _get_gaussian_weights(self, A: torch.Tensor, Y: torch.Tensor, bs:torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y: (batch_size, t, 1)
            A: (batch_size, n_dims, n_dims)
            bs: (batch_size, n_dims, n_tasks)
        Returns:
            (batch_size,)
        """
        d = A.shape[-1]
        log_det_A = torch.logdet(A)  # (batch_size,)
        log_det_A = log_det_A.unsqueeze(-1) # (batch_size, 1)
        y_norm_sq = torch.sum(Y * Y, dim=1)  # (batch_size, 1)
        quad_form = torch.einsum('bik,bij,bjk->bk', bs, A, bs) # bs[b,:,t]^T A[b] bs[b,:,t], aligns on batch, A: (batch_size, n_dims, n_dims), bs: (batch_size, n_dims, n_tasks)
        w_norm_sq = torch.sum(self.W * self.W, dim=0, keepdim=True)  # (1, n_tasks)
        log_gauss_weights = -0.5 * (y_norm_sq / self.noise_var + w_norm_sq / self.minor_var - quad_form / self.noise_var)  # (batch_size, n_tasks)
        log_gauss_weights += log(self.p0) + d * log(self.tau) + 0.5 * log_det_A # (batch_size, n_tasks)
        return log_gauss_weights
    
    def _get_delta_weights(self, Y: torch.Tensor, X:torch.Tensor) -> torch.Tensor:
        # X: (batch_size, n_points, n_dims), Y: (batch_size, n_points, 1)
        batch_size, n_points, n_dims = X.shape
        log_delta_weights = -0.5 * torch.sum((Y - torch.bmm(X, self.W.expand(batch_size, -1, -1)))**2, dim=1) / self.noise_var  # (batch_size, n_tasks)
        log_delta_weights += log(1 - self.p0) # (batch_size, n_tasks)
        return log_delta_weights


class MixedRidge(nn.Module):
    def __init__(self, tau: float, task_pool: torch.Tensor, p0: float, noise_scale: float, dtype=torch.float32):
        """
        Args:
            scale: noise std
            task_pool: Tensor of shape (n_tasks, n_dims, 1)
        """
        super().__init__()
        self.tau = tau
        self.noise_scale = noise_scale
        self.noise_var = noise_scale ** 2
        self.dtype = dtype
        self.p0 = p0

        assert task_pool.ndim == 3 and task_pool.shape[2] == 1
        self.task_pool = task_pool.to(dtype)

        # Preprocess: squeeze and transpose for use in prediction
        self.W = task_pool.squeeze(-1).T  # shape: (n_dims, n_tasks)
        if self.dtype == torch.float16: self.W = self.W.to("cuda")
        
    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: (batch_size, n_points, n_dims)
            targets: (batch_size, n_points)
        Returns:
            preds: (batch_size, n_points)
        """
        if data.ndim == 4: data = data.squeeze(0)
        if targets.ndim == 3: targets = targets.squeeze(0)
        device = data.device
        batch_size, n_points, _ = data.shape
        targets = targets.unsqueeze(-1)  # (batch_size, n_points, 1)

        preds = []  # list of (batch_size,)
        preds_i = self.predict(torch.zeros_like(data[:, :1]), torch.zeros_like(targets[:, :1]), data[:, 0:1]).to(device)
        preds.append(preds_i)  # initial prediction with zero input

        # Iterative MMSE predictions
        for i in range(1, n_points):
            pred_i = self.predict(data[:, :i], targets[:, :i], data[:, i:i+1]).to(device)
            preds.append(pred_i)

        return torch.stack(preds, dim=1)  # (batch_size, n_points)

    def predict(self, X: torch.Tensor, Y: torch.Tensor, test_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch_size, t, n_dims)
            Y: (batch_size, t, 1)
            test_x: (batch_size, 1, n_dims)
        Returns:
            (batch_size,)
        """
        assert X.ndim == 3 and Y.ndim == 3 and test_x.ndim == 3, "X, Y, and test_x must be 3D tensors"
        batch_size, t, n_dims = X.shape
        lam = self.tau ** 2

        XT = X.transpose(1, 2)                          # (batch_size, n_dims, t)
        XT_Y = torch.bmm(XT, Y)                         # (batch_size, n_dims, 1)

        # Ridge matrix: X^T X + λI
        eye = torch.eye(n_dims, dtype=self.dtype, device=X.device).unsqueeze(0).expand(batch_size, -1, -1)
        ridge_matrix = torch.bmm(XT, X) + lam * eye     # (batch_size, n_dims, n_dims)

        log_delta_weights = self._get_delta_weights(X, Y)  # (batch_size, n_tasks)
        log_gauss_weights = self._get_gaussian_weights(X, Y) # (batch_size, 1)
        log_weights = torch.cat([log_delta_weights, log_gauss_weights], dim=1)  # (batch_size, n_tasks+1)
        weights = torch.softmax(log_weights, dim=-1)  # normalize, (batch_size, n_tasks+1)
        
        locs_gauss_weight = torch.linalg.solve(ridge_matrix, XT_Y)  # (batch_size, n_dims, 1)

        locs = torch.cat([self.W.unsqueeze(0).expand(batch_size, -1, -1), locs_gauss_weight], dim=2)  # (batch_size, n_dims, n_tasks+1)
        ws = torch.einsum('bik,bk->bi', locs, weights)  # (batch_size, n_dims)
        pred = torch.bmm(test_x, ws.unsqueeze(-1)) # (batch_size, 1, 1)

        return pred[:,0,0] # (batch_size,)

    def _get_gaussian_weights(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y: (batch_size, t, 1)
            A: (batch_size, n_dims, n_dims)
            X: (batch_size, t, n_dims)
        Returns:
            (batch_size,)
        """
        d = X.shape[-1]
        A = torch.bmm(X.transpose(1, 2), X) + (self.tau ** 2) * torch.eye(d, device=X.device, dtype=X.dtype).unsqueeze(0)  # (batch_size, n_dims, n_dims)
        z = torch.linalg.solve(A, torch.bmm(X.transpose(1,2), Y))  # (batch_size, n_dims, 1)
        log_det_A = torch.logdet(A)  # (batch_size,)
        log_det_A = log_det_A.unsqueeze(-1) # (batch_size, 1)
        quad_form = torch.einsum('bik,bij,bjk->bk', z, A, z)  # bs[b,:,t]^T A[b] bs[b,:,t], aligns on batch, A: (batch_size, n_dims, n_dims), bs: (batch_size, n_dims, 1)

        log_gauss_weights = 0.5 * quad_form / self.noise_var - torch.sum(Y*Y, dim=1)  # (batch_size, 1)
        log_gauss_weights += log(self.p0) + d * log(self.tau) - 0.5 * log_det_A # (batch_size, 1)
        return log_gauss_weights
    
    def _get_delta_weights(self, X: torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
        # X: (batch_size, n_points, n_dims), Y: (batch_size, n_points, 1)
        batch_size, n_points, n_dims = X.shape
        log_delta_weights = -0.5 * torch.sum((Y - torch.bmm(X, self.W.expand(batch_size, -1, -1)))**2, dim=1) / self.noise_var  # (batch_size, n_tasks)
        log_delta_weights += log((1 - self.p0) / 3)# (batch_size, n_tasks)
        return log_delta_weights




########################################################################################################################
# Ridge                                                                                                                #
########################################################################################################################

class Ridge(nn.Module):
    def __init__(self, lam: float, dtype=torch.float32):
        super().__init__()
        self.lam = lam
        self.dtype = dtype
    
    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        device = data.device

        if data.ndim == 4: data = data.squeeze(0)
        if data.dtype == torch.float16:
            data = data.to("cuda")
            targets = targets.to("cuda")
        batch_size, n_points, _ = data.shape
        if targets.ndim == 3: targets = targets.squeeze(0)
        targets = targets.unsqueeze(-1)
        preds = [torch.zeros(batch_size, dtype=self.dtype, device=data.device)]
        for i in range(1, n_points):
            pred_i = self.predict(
                data[:, :i],          # X: (batch, i, dim)
                targets[:, :i],       # Y: (batch, i, 1)
                data[:, i:i+1],       # test_x: (batch, 1, dim)
                self.lam
            )
            preds.append(pred_i)
        preds = torch.stack(preds, dim=1)  # (batch_size, n_points)
        return preds.to(device)
    
    def predict(self, X: torch.Tensor, Y: torch.Tensor, test_x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Args:
            X: (batch_size, i, n_dims)
            Y: (batch_size, i, 1)
            test_x: (batch_size, 1, n_dims)
        Returns:
            (batch_size,)
        """
        batch_size, i, n_dims = X.shape

        XT = X.transpose(1, 2)                          # (batch_size, n_dims, i)
        XT_Y = torch.bmm(XT, Y)                         # (batch_size, n_dims, 1)

        # Ridge matrix: X^T X + λI
        eye = torch.eye(n_dims, dtype=self.dtype, device=X.device).unsqueeze(0).expand(batch_size, -1, -1)
        ridge_matrix = torch.bmm(XT, X) + lam * eye     # (batch_size, n_dims, n_dims)

        # Solve (XT X + λI) w = XT Y
        if ridge_matrix.dtype == torch.float16:
            ridge_matrix_fp32 = ridge_matrix.float()
            XT_Y_fp32 = XT_Y.float()

            ws_fp32 = torch.linalg.solve(ridge_matrix_fp32, XT_Y_fp32)  # (batch_size, n_dims, 1)
            ws = ws_fp32.to(dtype=torch.float16)
        else:
            ws = torch.linalg.solve(ridge_matrix, XT_Y) 

        # Predict: test_x @ w
        pred = torch.bmm(test_x, ws)                    # (batch_size, 1, 1)
        return pred[:, 0, 0]  

    def evolve(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if data.ndim == 4:
            data = data.squeeze(0)
        if targets.ndim == 3:
            targets = targets.squeeze(0)
        if data.dtype == torch.float16:
            data = data.to("cuda")
            targets = targets.to("cuda")
        batch_size, n_points, _ = data.shape
        targets = targets.unsqueeze(-1)  # (batch_size, n_points, 1)

        ws = torch.zeros((batch_size, data.shape[-1], n_points-1), 
                         dtype=self.dtype, device=data.device)  # (batch_size, n_dims, 1)

        # Iterative MMSE predictions
        for i in range(1, n_points):
            X, Y = data[:, :i], targets[:, :i]
            batch_size, i, n_dims = X.shape

            XT = X.transpose(1, 2)                          # (batch_size, n_dims, i)
            XT_Y = torch.bmm(XT, Y)                         # (batch_size, n_dims, 1)

            # Ridge matrix: X^T X + λI
            eye = torch.eye(n_dims, dtype=self.dtype, device=X.device).unsqueeze(0).expand(batch_size, -1, -1)
            ridge_matrix = torch.bmm(XT, X) + self.lam * eye     # (batch_size, n_dims, n_dims)

            # Solve (XT X + λI) w = XT Y
            if ridge_matrix.dtype == torch.float16:
                ridge_matrix_fp32 = ridge_matrix.float()
                XT_Y_fp32 = XT_Y.float()

                ws_fp32 = torch.linalg.solve(ridge_matrix_fp32, XT_Y_fp32)  # (batch_size, n_dims, 1)
                weights = ws_fp32.to(dtype=torch.float16)
            else:
                weights = torch.linalg.solve(ridge_matrix, XT_Y) 
            
            ws[:, :, i-1] = weights.squeeze(2)  # (batch_size, n_dims, 1)

        return ws        

########################################################################################################################
# MMSE                                                                                                                #
########################################################################################################################


class DiscreteMMSE(nn.Module):
    def __init__(self, scale: float, task_pool: torch.Tensor, dtype=torch.float32):
        """
        Args:
            scale: noise std
            task_pool: Tensor of shape (n_tasks, n_dims, 1)
        """
        super().__init__()
        self.scale = scale
        self.dtype = dtype

        assert task_pool.ndim == 3 and task_pool.shape[2] == 1
        self.task_pool = task_pool.to(dtype)

        # Preprocess: squeeze and transpose for use in prediction
        self.W = task_pool.squeeze(-1).T  # shape: (n_dims, n_tasks)
        if self.dtype == torch.float16:
            self.W = self.W.to("cuda")

    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: (batch_size, n_points, n_dims)
            targets: (batch_size, n_points)
        Returns:
            preds: (batch_size, n_points)
        """
        if data.ndim == 4:
            data = data.squeeze(0)
        if targets.ndim == 3:
            targets = targets.squeeze(0)
        device = data.device
        if data.dtype == torch.float16:
            data = data.to("cuda")
            targets = targets.to("cuda")
        batch_size, n_points, _ = data.shape
        targets = targets.unsqueeze(-1)  # (batch_size, n_points, 1)

        # Initial prediction with mean of W
        w_mean = self.W.mean(dim=1)  # (n_dims,)
        preds = [torch.matmul(data[:, 0], w_mean).to(device)]  # list of (batch_size,)

        # Iterative MMSE predictions
        for i in range(1, n_points):
            pred_i = self.predict(data[:, :i], targets[:, :i], data[:, i:i+1]).to(device)
            preds.append(pred_i)

        return torch.stack(preds, dim=1)  # (batch_size, n_points)

    def evolve(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if data.ndim == 4:
            data = data.squeeze(0)
        if targets.ndim == 3:
            targets = targets.squeeze(0)
        if data.dtype == torch.float16:
            data = data.to("cuda")
            targets = targets.to("cuda")
        batch_size, n_points, _ = data.shape
        targets = targets.unsqueeze(-1)  # (batch_size, n_points, 1)

        ws = torch.zeros((batch_size, self.W.shape[0], n_points-1), 
                         dtype=self.dtype, device=data.device)  # (batch_size, n_dims, 1)

        # Iterative MMSE predictions
        for i in range(1, n_points):
            X, Y = data[:, :i], targets[:, :i] # (batch_size, i, n_dims), (batch_size, i, 1), (batch_size, 1, n_dims)
            XW = torch.matmul(X, self.W)  # broadcasting matmul

            # shape: (batch_size, i, n_tasks)
            diff = Y - XW

            # Compute log-likelihood under Gaussian noise model
            log_scale_sq = 2 * torch.log(
                torch.tensor(self.scale, dtype=self.dtype, device=data.device)
            )
            log_probs = -0.5 * ((diff / self.scale) ** 2 + log_scale_sq)
            # log_probs = -0.5 * ((diff / self.scale) ** 2 + 2 * torch.log(torch.tensor(self.scale)))  # log N(·; 0, scale^2)
            alpha = log_probs.sum(dim=1)  # (batch_size, n_tasks)

            # Softmax weights over task pool
            weights = F.softmax(alpha, dim=1)  # (batch_size, n_tasks)

            # Compute MMSE weights: (batch_size, n_dims)
            w_mmse = torch.bmm(weights.unsqueeze(1), self.W.T.unsqueeze(0).expand(X.size(0), -1, -1))  # (batch, 1, n_dims)
            w_mmse = w_mmse.transpose(1, 2)  # (batch_size, n_dims, 1)
            ws[:, :, i-1] = w_mmse.squeeze(2)  # (batch_size, n_dims, 1)


        return ws


    def predict(self, X: torch.Tensor, Y: torch.Tensor, test_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch_size, i, n_dims)
            Y: (batch_size, i, 1)
            test_x: (batch_size, 1, n_dims)
        Returns:
            pred: (batch_size,)
        """
        # shape: (batch_size, i, n_tasks)
        XW = torch.matmul(X, self.W)  # broadcasting matmul

        # shape: (batch_size, i, n_tasks)
        diff = Y - XW

        log_scale_sq = 2 * torch.log(
            torch.tensor(self.scale, dtype=self.dtype, device=X.device)
        )
        log_probs = -0.5 * ((diff / self.scale) ** 2 + log_scale_sq)  # (batch_size, i, n_tasks)

        alpha = log_probs.sum(dim=1)  # (batch_size, n_tasks)

        # Softmax weights over task pool
        weights = F.softmax(alpha, dim=1)  # (batch_size, n_tasks)

        # Compute MMSE weights: (batch_size, n_dims)
        w_mmse = torch.bmm(weights.unsqueeze(1), self.W.T.unsqueeze(0).expand(X.size(0), -1, -1))  # (batch, 1, n_dims)
        w_mmse = w_mmse.transpose(1, 2)  # (batch_size, n_dims, 1)

        # Predict: test_x @ w_mmse → (batch, 1, 1)
        pred = torch.bmm(test_x, w_mmse)
        return pred[:, 0, 0]  # (batch_size,)
    

class UnbalancedMMSE(nn.Module):
    def __init__(self, scale: float, task_pool: torch.Tensor, p0: float,
                 minor_task_pool: torch.Tensor, dtype=torch.float32):
        """
        Args:
            scale: noise std
            task_pool: Tensor of shape (n_tasks, n_dims, 1)
            minor_task_pool: Tensor of shape (n_minor_tasks, n_dims, 1)
            p0: prior probability of major tasks
        """
        super().__init__()
        self.scale = scale
        self.dtype = dtype
        self.p0 = p0
        self.n_minor_tasks = minor_task_pool.shape[0]

        assert task_pool.ndim == 3 and task_pool.shape[2] == 1
        self.task_pool = task_pool.to(dtype)

        # Preprocess: squeeze and transpose for use in prediction
        self.W = task_pool.squeeze(-1).T  # shape: (n_dims, n_tasks)
        self.W_minor = minor_task_pool.squeeze(-1).T  # shape: (n_dims, n_tasks)

    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: (batch_size, n_points, n_dims)
            targets: (batch_size, n_points)
        Returns:
            preds: (batch_size, n_points)
        """
        if data.ndim == 4:
            data = data.squeeze(0)
        if targets.ndim == 3:
            targets = targets.squeeze(0)
        device = data.device
        batch_size, n_points, _ = data.shape
        targets = targets.unsqueeze(-1)  # (batch_size, n_points, 1)

        # Initial prediction with mean of W
        preds = []
        preds_i = self.predict(torch.zeros_like(data[:, :1]), torch.zeros_like(targets[:, :1]), data[:, 0:1]).to(device)
        preds.append(preds_i)  # initial prediction with zero input

        # Iterative MMSE predictions
        for i in range(1, n_points):
            pred_i = self.predict(data[:, :i], targets[:, :i], data[:, i:i+1]).to(device)
            preds.append(pred_i)

        return torch.stack(preds, dim=1)  # (batch_size, n_points)


    def predict(self, X: torch.Tensor, Y: torch.Tensor, test_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch_size, i, n_dims)
            Y: (batch_size, i, 1)
            test_x: (batch_size, 1, n_dims)
        Returns:
            pred: (batch_size,)
        """
        # shape: (batch_size, i, n_tasks)
        XW = torch.matmul(X, self.W)  # broadcasting matmul
        XW_minor = torch.matmul(X, self.W_minor)  # broadcasting matmul for minor tasks

        # shape: (batch_size, i, n_tasks)
        diff = Y - XW
        diff_norm_sq = (diff ** 2).sum(dim=1)  # (batch_size, n_tasks)
        diff_minor = Y - XW_minor
        diff_minor_norm_sq = (diff_minor ** 2).sum(dim=1)  # (batch_size, n_minor_tasks)

        log_probs = -0.5 * (diff_norm_sq / self.scale**2) + log((1-self.p0) / 3) # (batch_size, n_tasks)
        log_probs_minor = -0.5 * (diff_minor_norm_sq / self.scale**2) + log(self.p0 / self.n_minor_tasks) # (batch_size, n_minor_tasks)
        log_probs = torch.cat([log_probs, log_probs_minor], dim=1) # (batch_size, n_tasks + n_minor_tasks)
        

        # Softmax weights over task pool
        weights = F.softmax(log_probs, dim=1)  # (batch_size, n_tasks + n_minor_tasks)

        # Compute MMSE weights: (batch_size, n_dims)
        regression_weights = torch.cat([self.W, self.W_minor], dim=1).T # (n_tasks + n_minor_tasks, n_dims)
        w_mmse = torch.bmm(weights.unsqueeze(1), regression_weights.unsqueeze(0).expand(X.size(0), -1, -1))  # (batch, 1, n_dims)
        w_mmse = w_mmse.transpose(1, 2)  # (batch_size, n_dims, 1)

        # Predict: test_x @ w_mmse → (batch, 1, 1)
        pred = torch.bmm(test_x, w_mmse)
        return pred[:, 0, 0]  # (batch_size,)

Model = Ridge | DiscreteMMSE | MixedRidge | UnbalancedMMSE

def get_model(name: str, **kwargs) -> Model:
    models = {"ridge": Ridge, "discrete_mmse": DiscreteMMSE, "transformer": TransformerLin, 
              "mixture": Mixture, "mixed_ridge": MixedRidge, "unbalanced_mmse": UnbalancedMMSE}
    return models[name](**kwargs)