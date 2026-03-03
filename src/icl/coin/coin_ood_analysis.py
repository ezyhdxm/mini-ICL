import copy
import torch

import icl.utils.notebook_utils as nu


def get_new_sampler(exp_name, n_minor=64, n_ood=30):
    _, sampler, _ = nu.load_everything("coin", exp_name)
    sampler_clone = copy.deepcopy(sampler)

    orig = sampler_clone.minor_p
    if orig is None:
        raise ValueError("sampler.minor_p is None")

    # number of original minor tasks to keep
    k_minor = min(int(n_minor), int(sampler.n_minor_tasks))
    n_ood = int(n_ood)
    n_tasks = n_ood + k_minor

    sampler_clone.n_minor_tasks = n_tasks

    trailing_shape = orig.shape[1:]

    def make_random_rows(n: int):
        n = int(n)
        x = torch.rand((n, *trailing_shape), device=orig.device, dtype=orig.dtype)
        if orig.ndim >= 2 and n > 0:
            x = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return x

    new_minor = orig.new_empty((n_tasks, *trailing_shape)) if n_tasks > 0 else orig.new_empty((0, *trailing_shape))

    # OOD first
    if n_ood > 0:
        new_minor[:n_ood].copy_(make_random_rows(n_ood))

    # original minors last (only if k_minor > 0; avoids the -0 slice bug)
    if k_minor > 0:
        new_minor[-k_minor:].copy_(orig[:k_minor])

    sampler_clone.minor_p = new_minor
    return sampler_clone, k_minor