import torch
import torch.nn as nn
import dataclasses
from typing import Optional, Tuple, Any, List, Callable

from icl.linear.lr_models import get_model

# Adapted from https://github.com/mansheej/icl-task-diversity/blob/main/icl/tasks.py

Sampler = Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

@dataclasses.dataclass
class NoisyLinearRegression:
    n_tasks: int
    n_minor_tasks: int
    n_dims: int
    n_points: int
    p_minor: float
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float
    task_scale: float
    minor_scale: float
    noise_scale: float
    is_mixture: bool = False
    dtype: Any = torch.float32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.data_gen = torch.Generator(device=self.device).manual_seed(self.data_seed) # set independent generator for data sampling
        self.task_gen = torch.Generator(device=self.device).manual_seed(self.task_seed)
        self.minor_gen = torch.Generator(device=self.device).manual_seed(self.task_seed + 1)  # separate generator for OOD tasks
        self.noise_gen = torch.Generator(device=self.device).manual_seed(self.noise_seed)
        self.task_pool: Optional[torch.Tensor] = self.generate_task_pool() if self.n_tasks > 0 else None
        self.minor_pool: Optional[torch.Tensor] = self.generate_minor_pool() if (self.n_minor_tasks > 0 and self.n_tasks > 0) else None

    @property
    def name(self) -> str:
        return f"NoisyLinReg({self.n_tasks})"
    
    @classmethod
    def from_task_pool(cls, task_pool: torch.Tensor, **kwargs) -> "NoisyLinearRegression":
        assert kwargs["n_tasks"] == task_pool.shape[0]
        task = cls(**kwargs)
        task.task_pool = task_pool
        return task
    
    def generate_task_pool(self) -> torch.Tensor:
        # generate a pool of tasks w1, w2, ..., wN, where N = n_tasks
        # w_i ~ N(0, task_scale^2 * I), where I is the identity matrix of size D = n_dims
        shape = (self.n_tasks, self.n_dims, 1)
        return torch.randn(shape, generator=self.task_gen, dtype=self.dtype, device=self.device) * self.task_scale

    def generate_minor_pool(self) -> torch.Tensor:
        # generate a pool of tasks w1, w2, ..., wN, where N = n_minor_tasks
        # w_i ~ N(0, minor_scale^2 * I) + task[k], k ~ Unif([n_tasks]), where I is the identity matrix of size D = n_dims
        assert self.n_tasks > 0, "Minor tasks can only be generated if n_tasks > 0"
        assert self.task_pool is not None, "Task pool must be initialized"
        
        # Sample random centroids from the task pool
        if self.is_mixture:
            indices = torch.randint(
                low=0, 
                high=self.n_tasks, 
                size=(self.n_minor_tasks,), 
                generator=self.minor_gen,
                device=self.device
            )
            minor_centroid = self.task_pool[indices]

        # Add Gaussian noise
        noise_shape = (self.n_minor_tasks, self.n_dims, 1)
        noise = torch.randn(noise_shape, generator=self.minor_gen, dtype=self.dtype, device=self.device)

        if self.is_mixture:
            return minor_centroid + self.minor_scale * noise

        else:
            return noise

    def sample_data(self, step: int) -> torch.Tensor:
        # generate a batch of data points x1, x2, ..., xN, where N = n_points
        # x_i ~ N(0, data_scale^2 * I), where I is the identity matrix of size D = n_dims
        # step is used to generate different data points for each batch
        self.data_gen.manual_seed(self.data_seed + step)
        shape = (self.batch_size, self.n_points, self.n_dims)
        return torch.randn(shape, generator=self.data_gen, dtype=self.dtype, device=self.device) * self.data_scale
    
    def sample_tasks(self, step: int, is_eval: bool=False) -> torch.Tensor:
        # sample a batch of tasks w1, w2, ..., wB from the task pool, where B = batch_size
        self.task_gen.manual_seed(self.task_seed + step)
        if self.n_tasks > 0:
            assert self.task_pool is not None, "Task pool must be initialized"
            idxs = torch.randint(low=0, high=self.n_tasks, size=(self.batch_size,), 
                                 generator=self.task_gen, device=self.device) # (batch_size,)
            tasks = self.task_pool[idxs]
            if not is_eval and self.p_minor > 0:
                minor_mask = torch.rand(self.batch_size, generator=self.task_gen, device=self.device) < self.p_minor
                n_minor = int(minor_mask.sum().item())
                if n_minor > 0:
                    if self.minor_pool is not None:
                        minor_tasks = self.minor_pool[torch.randint(low=0, high=self.n_minor_tasks, size=(n_minor,), generator=self.task_gen, device=self.device)]
                    else:
                        minor_shape = (n_minor, self.n_dims, 1)
                        minor_tasks = torch.randn(minor_shape, generator=self.task_gen, dtype=self.dtype, device=self.device) * self.task_scale
                    tasks[minor_mask] = minor_tasks

        else:
            # infinite task pool
            shape = (self.batch_size, self.n_dims, 1)
            tasks = torch.randn(shape, generator=self.task_gen, dtype=self.dtype, device=self.device) * self.task_scale
        return tasks
    
    def evaluate(self, data: torch.Tensor, tasks: torch.Tensor, step: int) -> torch.Tensor:
        # data: (batch, n_points, n_dims)
        # tasks: (batch, n_dims, k)
        if data.device != self.device:
            data = data.to(self.device)
        if tasks.device != self.device:
            tasks = tasks.to(self.device)
        
        targets = (data @ tasks)  # (batch, n_points, k)
        self.noise_gen.manual_seed(self.noise_seed + step)
        if targets.shape[2] == 1:
            targets = targets.squeeze(-1)
        noise = torch.randn(targets.shape, dtype=targets.dtype, device=targets.device, generator=self.noise_gen) * self.noise_scale
        return targets + noise

    def sample_batch(self, step: int, is_eval: bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.sample_data(step)
        tasks = self.sample_tasks(step, is_eval=is_eval) # (batch_size, n_dims, 1)
        targets = self.evaluate(data, tasks, step)
        return data, tasks, targets
    
    def sample_from_task(self, task: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of data and noisy targets for a specific given task.

        Args:
            task: Tensor of shape [n_dims, 1] or [1, n_dims, 1].
            step: Integer step for deterministic sampling.

        Returns:
            data: [batch_size, n_points, n_dims]
            targets: [batch_size, n_points]
        """
        if task.device != self.device:
            task = task.to(self.device)
        data = self.sample_data(step)  # [B, T, D]
        if task.ndim == 2:
            task = task.unsqueeze(0)  # [1, D, 1]
        assert task.shape == (1, self.n_dims, 1), f"Task shape should be [1, {self.n_dims}, 1], got {task.shape}"

        tasks = task.expand(self.batch_size, -1, -1)  # [B, D, 1]
        targets = self.evaluate(data, tasks, step)    # [B, T]
        return data, targets

    @staticmethod
    def evaluate_oracle(data: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        return (data @ tasks).squeeze(-1)

    def get_default_eval_tasks(
        self, batch_size: int, task_seed: int, data_seed: int, noise_seed: int, **kwargs
    ) -> List["NoisyLinearRegression"]:
        assert task_seed != self.task_seed
        assert data_seed != self.data_seed
        assert noise_seed != self.noise_seed
        config = dataclasses.asdict(self)
        config.update(dict(
            batch_size=batch_size,
            task_seed=task_seed,
            data_seed=data_seed,
            noise_seed=noise_seed,
            n_tasks=0,
        ))
        eval_tasks = [NoisyLinearRegression(**config)]
        if self.n_tasks > 0:
            config["n_tasks"] = self.n_tasks
            eval_tasks.append(NoisyLinearRegression.from_task_pool(task_pool=self.task_pool.clone(), **config))
        return eval_tasks


    def get_default_eval_models(self) -> List[Any]:
        models = [get_model(name="ridge", lam=self.noise_scale**2 / self.task_scale**2, dtype=self.dtype)]
        if self.n_tasks > 0:
            assert self.task_scale == 1.0  # TODO
            models.append(
                get_model(
                    name="discrete_mmse", scale=self.noise_scale, task_pool=self.task_pool.clone(), dtype=self.dtype
                )
            )
            #models.append(
            #    get_model(
            #        name="mixture", tau=self.noise_scale/self.minor_scale, task_pool=self.task_pool.clone(), p0=self.p_minor, 
            #        noise_scale=self.noise_scale, dtype=self.dtype
            #    )
            #)
            models.append(
                get_model(
                    name="mixed_ridge", tau=self.noise_scale/self.minor_scale, task_pool=self.task_pool.clone(), p0=self.p_minor, 
                    noise_scale=self.noise_scale, dtype=self.dtype
                )
            )
            models.append(
                get_model(
                    name="unbalanced_mmse", minor_task_pool=self.minor_pool.clone(), task_pool=self.task_pool.clone(), p0=self.p_minor, 
                    scale=self.noise_scale, dtype=self.dtype
                )
            )

        return models
    
    def to(self, device: str) -> "NoisyLinearRegression":
        """Move the entire task to a different device"""
        if device == self.device:
            return self  # No-op if already on correct device
        
        # Update device
        self.device = device
        
        # Recreate generators on new device
        self.data_gen = torch.Generator(device=device).manual_seed(self.data_seed)
        self.task_gen = torch.Generator(device=device).manual_seed(self.task_seed)
        self.minor_gen = torch.Generator(device=device).manual_seed(self.task_seed + 1)
        self.noise_gen = torch.Generator(device=device).manual_seed(self.noise_seed)
        
        # Move task pools to new device
        if self.task_pool is not None:
            self.task_pool = self.task_pool.to(device)
        if self.minor_pool is not None:
            self.minor_pool = self.minor_pool.to(device)
        
        return self
    

########################################################################################################################
# Get Task                                                                                                             #
########################################################################################################################

Task = NoisyLinearRegression


def get_task(name: str, **kwargs) -> Task:
    tasks = {"noisy_linear_regression": NoisyLinearRegression}
    return tasks[name](**kwargs)

def get_task_name(task: "Task") -> str:
    return "Latent" if task.name.endswith("(0)") else "Pretrain"
