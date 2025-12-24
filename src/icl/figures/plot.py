import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import torch
import numpy as np
import seaborn as sns
from datetime import datetime
from tqdm.notebook import tqdm
from scipy.interpolate import make_interp_spline
from IPython.display import display, HTML

def moving_average(y, window_size=5):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')


def get_loss_plots(config, train_results, folder="loss_plots", show=False, verbose=False):
    os.makedirs(folder, exist_ok=True)
    print("Loss plots saved at", folder)

    task_name = config.task.name
    log = train_results["log"]
    
    # Extract losses
    train_losses = log["train/loss"]
    eval_losses = log["eval/IDLoss"] if len(log["eval/IDLoss"]) > 0 else log["eval/loss"]
    ood_losses = log["eval/OODLoss"] if "eval/OODLoss" in log else []
    
    # Extract accuracies
    eval_accs = log["eval/IDAcc"]
    ood_accs = log["eval/OODAcc"]
    eval_steps = log["eval/step"]
    baseline = log["baseline"] if "baseline" in log else []

    # === Plot Loss ===
    fig_loss, ax_loss = plt.subplots(1, 2, figsize=(12, 6))  # linear and log scale

    range_vec = range(1, config.training.num_epochs + 1)
    train_losses_smoothed = moving_average(train_losses)

    for ax in ax_loss:
        ax.plot(range_vec[:len(train_losses_smoothed)], train_losses_smoothed, linestyle='-', color='lightblue', label='Training Loss')
        ax.plot(eval_steps, eval_losses, linestyle='--', color='palevioletred', label='Validation Loss')
        #if len(ood_losses) >= 1:
        #    ax.plot(eval_steps, ood_losses, linestyle='--', label='OOD Loss')
        for i, base in enumerate(baseline):
            color = cm.get_cmap('tab10')(i)
            ax.axhline(y=base, linestyle='-', label=f'{i+1}-gram Baseline', color=color)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.grid()
        ax.legend()

    ax_loss[0].set_title('Loss (Linear Scale)')
    ax_loss[1].set_title('Loss (Log Scale)')
    ax_loss[1].set_xscale('log')

    loss_path = os.path.join(folder, "loss.png")

    print(loss_path)

    plt.savefig(loss_path)

    if verbose:
        print("Loss plot saved at", loss_path)
    if show:
        plt.show()
    plt.close()

    # === Plot Accuracy ===

    if len(eval_accs) > 0 or len(ood_accs) > 0: fig_acc, ax_acc = plt.subplots(figsize=(8, 6))

    if len(eval_accs) > 0:
        ax_acc.plot(eval_steps, eval_accs, linestyle='--', color='lightgreen', label='Validation Accuracy')
    if len(ood_accs) > 0:
        ax_acc.plot(eval_steps, ood_accs, linestyle='--', color='orange', label='OOD Accuracy')

    if len(eval_accs) > 0 or len(ood_accs) > 0:
        ax_acc.set_xlabel('Steps')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title(f'{task_name}: Accuracy Curve')
        ax_acc.grid()
        ax_acc.legend()

        acc_path = os.path.join(folder, f"accuracy_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        plt.tight_layout()
        plt.savefig(acc_path)
        if verbose:
            print("Accuracy plot saved at", acc_path)
        if show: plt.show()
        plt.close()


def plot_attn_scores(train_results, config, folder="loss_plots", show=False, log=True):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    task_name = config.task.name

    range_vec = np.arange(1, min(config.training.eval_iter, 100), 5)
    range_vec = np.concatenate((range_vec, np.arange(1, config.training.num_epochs+1, config.training.eval_iter)[1:]), axis=0)

    if "eval/pth_score" in train_results["log"].keys():
        pth_score = train_results["log"]["eval/pth_score"]
        ax.plot(range_vec, pth_score, linestyle='--', label='PTH Score')
        if log:
            ax.set_xscale('log')
    
    if "eval/ih_score" in train_results["log"].keys():
        ih_score = train_results["log"]["eval/ih_score"]
        ax.plot(range_vec, ih_score, linestyle='--', label='IH Score')
        if log:
            ax.set_xscale('log')
        
    ax.set_xlabel('Steps')
    

    ax.set_title('Attention Scores')
    ax.grid()
    ax.legend()
    
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    image_path = os.path.join(folder, f"attn_scores_{curr_time}.png")
    plt.savefig(image_path)
    if show:
        plt.show()
    plt.close()



