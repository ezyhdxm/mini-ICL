import torch
from typing import Optional

import icl.utils.notebook_utils as nu
from icl.dyck.dyck_utils import sample_binary_mask


def plot_dyck_task_posterior_padded(
    exp_name: str,
    n_plots: int = 3,
    dyck_mask: Optional[torch.Tensor] = None,
    uniform_prior: bool = True,
    max_positions: Optional[int] = None,
    figsize: tuple = (10, 3.5),
    title: Optional[str] = None,
) -> dict:
    """
    Generate random Dyck samples and plot the task posterior over Dyck-token positions.

    Parameters
    ----------
    exp_name : str
        Experiment name (folder under results/dyck/).
    n_plots : int, default=3
        Number of independent samples to plot (one subplot each).
    dyck_mask : torch.Tensor, optional
        1D binary mask. If None, one is sampled via sample_binary_mask.
    uniform_prior : bool, default=True
        If True, sets p_minor for uniform task prior.
    max_positions : int, optional
        Maximum number of Dyck positions to show per subplot.
    figsize : tuple, default=(12, 4)
        Figure size per subplot row (total height = figsize[1] * n_plots).
    title : str, optional
        Custom suptitle. If None, a default is generated.

    Returns
    -------
    info : dict
        - 'posteriors': list of (n_dyck, T) tensors, one per sample
        - 'dyck_mask': the mask used
        - 'fig': matplotlib Figure
        - 'axes': list of matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from icl.dyck.dyck import dyck_task_posterior_over_time

    _, sampler, config = nu.load_everything("dyck", exp_name)
    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    device = config.device

    original_p_minor = getattr(sampler, 'p_minor', 0.0)
    if uniform_prior and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)

    if dyck_mask is None:
        dyck_mask = sample_binary_mask(config).to(device)
    else:
        dyck_mask = dyck_mask.to(device)

    samples_raw, masks_raw = sampler.generate(
        mode="train", task=None, num_samples=n_plots, epochs=1,
        dyck_mask=dyck_mask.clone(),
    )
    if samples_raw.dim() == 3:
        samples_raw = samples_raw.squeeze(0)
        masks_raw = masks_raw.squeeze(0)
    samples = samples_raw.to(device)
    masks = masks_raw.to(device)

    posterior_all = dyck_task_posterior_over_time(sampler, samples, masks)

    dyck_positions = torch.nonzero(dyck_mask == 1, as_tuple=True)[0].cpu()

    n_major = sampler.n_major_tasks
    n_minor_tasks = sampler.n_minor_tasks
    major_cmap = plt.cm.Blues
    minor_cmap = plt.cm.Reds
    major_colors = [major_cmap(0.3 + 0.6 * i / max(n_major - 1, 1)) for i in range(n_major)]
    minor_colors = [minor_cmap(0.3 + 0.6 * i / max(n_minor_tasks - 1, 1)) for i in range(n_minor_tasks)]
    T = n_tasks

    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots), squeeze=False)
    axes = [axes[i, 0] for i in range(n_plots)]
    posteriors_out = []

    for idx, ax in enumerate(axes):
        posterior = posterior_all[idx].cpu()
        posterior_dyck = posterior[dyck_positions, :]
        if max_positions is not None:
            posterior_dyck = posterior_dyck[:max_positions]
        n_dyck = len(posterior_dyck)
        posteriors_out.append(posterior_dyck)

        x_axis = torch.arange(n_dyck)
        major_labeled = False
        minor_labeled = False
        for k in range(T):
            if k < n_major:
                color = major_colors[k]
                label = ("Major" if not major_labeled else None) if idx == 0 else None
                major_labeled = True
            else:
                color = minor_colors[k - n_major]
                label = ("Minor" if not minor_labeled else None) if idx == 0 else None
                minor_labeled = True
            ax.plot(x_axis.numpy(), posterior_dyck[:, k].numpy(), label=label,
                    color=color, alpha=0.8, linewidth=1.5)

        ax.set_ylabel("P(Z=k | obs)")
        ax.set_xlim(0, max(n_dyck - 1, 1))
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("", fontsize=14)
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=max(1, T // 20))

    axes[-1].set_xlabel("Dyck token index")
    if title is None:
        title = f"Dyck task posterior over time — {exp_name}"
    fig.suptitle("", fontsize=18, y=1.01)
    fig.tight_layout()
    plt.show()

    sampler.p_minor = original_p_minor

    return {
        'posteriors': posteriors_out,
        'dyck_mask': dyck_mask.cpu(),
        'fig': fig,
        'axes': axes,
    }


def plot_dyck_task_posterior(
    exp_name: str,
    n_plots: int = 3,
    dyck_mask: Optional[torch.Tensor] = None,
    uniform_prior: bool = True,
    max_positions: Optional[int] = None,
    figsize: tuple = (10, 3.5),
    title: Optional[str] = None,
) -> dict:
    """
    Generate random Dyck samples and plot the task posterior over Dyck-token positions.

    Non-padded counterpart of ``plot_dyck_task_posterior_padded``.

    Parameters
    ----------
    exp_name : str
        Experiment name (folder under results/dyck/).
    n_plots : int, default=3
        Number of independent samples to plot (one subplot each).
    dyck_mask : torch.Tensor, optional
        1D binary mask over real-token positions.
        If None, one is sampled via ``sample_binary_mask``.
    uniform_prior : bool, default=True
        If True, sets p_minor for uniform task prior.
    max_positions : int, optional
        Maximum number of Dyck positions to show per subplot.
    figsize : tuple, default=(12, 4)
        Figure size per subplot row.
    title : str, optional
        Custom suptitle.

    Returns
    -------
    info : dict
        ``{'posteriors': [...], 'dyck_mask': Tensor, 'fig': Figure, 'axes': [...]}``.
    """
    import matplotlib.pyplot as plt
    from icl.dyck.dyck import dyck_task_posterior_over_time

    _, sampler, config = nu.load_everything("dyck", exp_name)
    device = config.device

    original_p_minor = getattr(sampler, 'p_minor', 0.0)
    if uniform_prior and sampler.n_minor_tasks > 0:
        sampler.p_minor = sampler.n_minor_tasks / (sampler.n_major_tasks + sampler.n_minor_tasks)

    if dyck_mask is None:
        dyck_mask = sample_binary_mask(config).to(device)
    else:
        dyck_mask = dyck_mask.to(device)

    samples_raw, masks_raw = sampler.generate(
        mode="train", task=None, num_samples=n_plots, epochs=1,
        dyck_mask=dyck_mask.clone(),
    )
    if samples_raw.dim() == 3:
        samples_raw = samples_raw.squeeze(0)
        masks_raw = masks_raw.squeeze(0)
    samples = samples_raw.to(device)
    masks = masks_raw.to(device)

    posterior_all = dyck_task_posterior_over_time(sampler, samples, masks)

    dyck_positions = torch.nonzero(dyck_mask == 1, as_tuple=True)[0].cpu()

    n_tasks = sampler.n_major_tasks + sampler.n_minor_tasks
    n_major = sampler.n_major_tasks
    n_minor_tasks = sampler.n_minor_tasks
    major_cmap = plt.cm.Blues
    minor_cmap = plt.cm.Reds
    major_colors = [major_cmap(0.3 + 0.6 * i / max(n_major - 1, 1)) for i in range(n_major)]
    minor_colors = [minor_cmap(0.3 + 0.6 * i / max(n_minor_tasks - 1, 1)) for i in range(n_minor_tasks)]
    T = n_tasks

    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots), squeeze=False)
    axes = [axes[i, 0] for i in range(n_plots)]
    posteriors_out = []

    for idx, ax in enumerate(axes):
        posterior = posterior_all[idx].cpu()
        posterior_dyck = posterior[dyck_positions, :]
        if max_positions is not None:
            posterior_dyck = posterior_dyck[:max_positions]
        n_dyck = len(posterior_dyck)
        posteriors_out.append(posterior_dyck)

        x_axis = torch.arange(n_dyck)
        major_labeled = False
        minor_labeled = False
        for k in range(T):
            if k < n_major:
                color = major_colors[k]
                label = ("Major" if not major_labeled else None) if idx == 0 else None
                major_labeled = True
            else:
                color = minor_colors[k - n_major]
                label = ("Minor" if not minor_labeled else None) if idx == 0 else None
                minor_labeled = True
            ax.plot(x_axis.numpy(), posterior_dyck[:, k].numpy(), label=label,
                    color=color, alpha=0.8, linewidth=1.5)

        ax.set_ylabel("P(Z=k | obs)")
        ax.set_xlim(0, max(n_dyck - 1, 1))
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("", fontsize=14)
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=max(1, T // 20))

    axes[-1].set_xlabel("Dyck token index")
    if title is None:
        title = f"Dyck task posterior over time (nonpadded) — {exp_name}"
    fig.suptitle("", fontsize=18, y=1.01)
    fig.tight_layout()
    plt.show()

    sampler.p_minor = original_p_minor

    return {
        'posteriors': posteriors_out,
        'dyck_mask': dyck_mask.cpu(),
        'fig': fig,
        'axes': axes,
    }
