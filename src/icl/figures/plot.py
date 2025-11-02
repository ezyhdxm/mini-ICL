import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import shutil
from PIL import Image
import torch
import numpy as np
import seaborn as sns
import stat
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


def plot_probes(train_results, config, folder="loss_plots", show=False, log=True):
    # print("Probes plots saved at ", folder)
    probes = train_results["probes"]
    if len(probes) == 0:
        return
    
    flag = "attn" in probes.keys()
    
    plot_labels = {"attn": "attn", "ff_icl": "ff_icl", "combined_icl": "ff+attn", "ff_mem_unif": "ff_unif", "ff_mem_true": "ff_true"}
    
    task_name = config.task.name
    if flag:
        fig, axes = plt.subplots(1, 2, figsize=(6*2, 6))
        ax = axes[0]
    
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6*1, 6))
    
    range_vec = np.arange(1, min(config.training.eval_iter, 100), 5)
    range_vec = np.concatenate((range_vec, np.arange(1, config.training.num_epochs+1, config.training.eval_iter)[1:]), axis=0)
    
    for pkey in probes.keys():
        if pkey in ["wk0", "wk1", "wo1", "ff", "emb", "ff_emb", "ff+res", "res"]:
            ax.plot(range_vec, probes[pkey], 
                    linestyle='-', label=f'{pkey}')
            if log:
                ax.set_xscale('log')
            
        elif pkey in ["attn", "ff_icl", "combined_icl", "ff_mem_unif", "ff_mem_true"]:
            axes[1].plot(range_vec, probes[pkey], 
                        linestyle='-', label=f'{plot_labels[pkey]}')
            if log:
                axes[1].set_xscale('log')
            
    
    ax.set_xlabel('Steps')
    if task_name == "bietti":
        ax.set_ylabel('Mempry Recall & KL divergence')
    else:
        ax.set_ylabel('KL divergence')
    is_mlp = any(config.model.mlp)
    mlp = "no" if not is_mlp else "with"
    linear = "(linear)" if is_mlp and not any(config.model.activation) else "" 
    ax.set_title(f'{",".join(map(str, config.model.num_heads))} Heads {config.model.num_layers} Layers {mlp} MLP {linear} Recall ({config.model.pos_enc})')
    ax.grid()
    ax.legend()
    if flag:
        axes[1].set_title(f'ICL Measure')
        axes[1].legend()
        axes[1].grid()
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Average TV distance')
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    image_path = os.path.join(folder, f"probe_{curr_time}.png")
    plt.savefig(image_path)
    if show:
        plt.show()
    plt.close()



def plot_bigram_icl_risk(config, train_results, folder="loss_plots", show=False):
    
    log = train_results["log"]

    if len(log["eval/IDLoss"]) == 0:
        return 

    id_losses, icl_losses = log["eval/IDLoss"], log["eval/ICLLoss"]
    ood_losses_smoothed = None
    copy_errors_smoothed = None
    if len(log["eval/OODLoss"]) > 0:
        ood_losses = log["eval/OODLoss"]
        ood_losses_smoothed = moving_average(ood_losses)
    
    if len(log["eval/CopyError"]) > 0:
        copy_errors = log["eval/CopyError"]
        copy_errors_smoothed = moving_average(copy_errors)

    id_losses_smoothed = moving_average(id_losses)
    icl_losses_smoothed = moving_average(icl_losses)
    
    task_name = config.task.name
    fig, axes = plt.subplots(1, 2, figsize=(6*2, 6))
    
    range_vec = np.arange(1, min(config.training.eval_iter, 100), 5)
    range_vec = np.concatenate((range_vec, np.arange(1, config.training.num_epochs+1, config.training.eval_iter)[1:]), axis=0)

    axes[0].plot(range_vec[:len(id_losses_smoothed)], id_losses_smoothed, linestyle='-', label='ID Risk')
    axes[1].plot(range_vec[:len(id_losses_smoothed)], id_losses_smoothed, linestyle='-', label='ID Risk')
    axes[0].plot(range_vec[:len(icl_losses_smoothed)], icl_losses_smoothed, linestyle='--', label='ICL Risk')
    axes[1].plot(range_vec[:len(icl_losses_smoothed)], icl_losses_smoothed, linestyle='--', label='ICL Risk')
    if ood_losses_smoothed is not None:
        axes[0].plot(range_vec[:len(ood_losses_smoothed)], ood_losses_smoothed, linestyle='-', label='OOD Risk')
        axes[1].plot(range_vec[:len(ood_losses_smoothed)], ood_losses_smoothed, linestyle='-', label='OOD Risk')
    
    if copy_errors_smoothed is not None:
        axes[0].plot(range_vec[:len(copy_errors_smoothed)], copy_errors_smoothed, linestyle='-', label='Copy Error')
        axes[1].plot(range_vec[:len(copy_errors_smoothed)], copy_errors_smoothed, linestyle='-', label='Copy Error')

    axes[1].set_xscale('log')
    axes[0].set_xlabel('Steps')
    axes[1].set_xlabel('Steps (Log Scale)')
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('Loss')
    mlp = "no" if config.model.mlp == False else "with"
    linear = "(linear)" if config.model.activation == False else "" 
    axes[0].set_title(f'{task_name}: {config.model.num_heads} Heads {config.model.num_layers} Layers {mlp} MLP {linear} Loss Over Epochs ({config.model.pos_enc})')
    axes[0].legend()
    axes[0].grid()
    axes[1].legend()
    axes[1].grid()
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    image_path = os.path.join(folder, f"icl_{curr_time}.png")
    plt.savefig(image_path)
    if show:
        plt.show()
    plt.close()



def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.
    
    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

def remove_readonly(func, path, exc_info):
    """Handle read-only files while deleting"""
    os.chmod(path, stat.S_IWRITE)  # Change to writable
    func(path)  # Retry removal


def get_attn_gif(layer, head, train_results, config, dag=None, folder="attns", out_folder="attns_plot", show=False, verbose=False):
    task_name = config.task.name
    attn_maps = train_results["attn_maps"]
    image_paths = []
    if os.path.exists(folder):
        shutil.rmtree(folder, onerror=remove_readonly)  # Handle read-only files
        print(f"Deleted: {folder}")
    
    os.makedirs(folder)
    steps = 0
    n_layer, n_heads, n_voc = config.model.num_layers, config.model.num_heads[layer], config.vocab_size
    
    for i, attn in tqdm(attn_maps.items(), mininterval=1, desc="Creating images", leave=False):
        if i < steps:
            continue
        
        if steps < 3000:
            steps += config.training.get_attn
        elif steps < 6000:
            steps += max(500, config.training.get_attn)
        else:
            steps += max(1000, config.training.get_attn)

        if dag is None:
            if head != "all" or config.model.num_heads[layer]==1:
                head = 0
                plt.figure(figsize=(6, 6))
                sns.heatmap(attn[layer][head].cpu(), cmap="viridis", annot=False, cbar=False)
                plt.title(f"Layer {layer}, Head {head}, Epoch {i + 1}")
            else:
                h = config.model.num_heads[layer]
                fig, axes = plt.subplots(1, h, figsize=(6*h, 6))
                for j in range(h):
                    sns.heatmap(attn[layer][j].cpu(), ax=axes[j], cmap="viridis", annot=False, cbar=False)
                    axes[j].set_title(f"Head {j+1} at Epoch {i + 1}")
                plt.tight_layout()
        else:
            adj_mat = dag_to_adj(dag, task_name)
            if head != "both":
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

                sns.heatmap(attn[layer][head].cpu(), ax=axes[0], cmap="viridis", annot=False, linewidths=0.05)
                axes[0].set_title(f"Attention Map at Epoch {i + 1}")
                sns.heatmap(adj_mat, ax=axes[1], cmap="viridis", annot=False, linewidths=0.05)
                axes[1].set_title("Adjacency Matrix of DAG")
                plt.tight_layout()
            else:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

                sns.heatmap(attn[layer][0].cpu(), ax=axes[0], cmap="viridis", annot=False, linewidths=0.05)
                axes[0].set_title(f"Head 1 Attention Map at Epoch {i + 1}")
                sns.heatmap(attn[layer][1].cpu(), ax=axes[1], cmap="viridis", annot=False, linewidths=0.05)
                axes[1].set_title(f"Head 2 Attention Map at Epoch {i + 1}")
                sns.heatmap(adj_mat, ax=axes[2], cmap="viridis", annot=False, linewidths=0.05)
                axes[2].set_title("Adjacency Matrix of DAG")
                plt.tight_layout()


        # Save image
        image_path = os.path.join(folder, f"attn_l{n_layer}h{n_heads}v{n_voc}ep{i}_L{layer}H{head}{task_name}.png")
        plt.savefig(image_path)
        plt.close()
        image_paths.append(image_path)

    # Step 2: Combine images into a GIF
    frames = [Image.open(image_path) for image_path in image_paths]
    os.makedirs(out_folder, exist_ok=True)
    # Get current time
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_gif_path = os.path.join(out_folder, f"attn_l{layer}_h{head}_{curr_time}.gif")
    
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,  # Duration between frames in milliseconds
        loop=0  # Infinite loop
    )
    
    if verbose:
        print(f"GIF saved at {output_gif_path}")
    shutil.rmtree(folder, onerror=remove_readonly)
    if verbose:
        print(f"Folder '{folder}' and its contents removed.")
    if show:
        display(HTML(f'<img src="{output_gif_path}" width="500px">'))
    return output_gif_path

def get_pos_sim(seq_len, model, device, pos_enc):
    
    if pos_enc == "abs":
        range_pos_toks = torch.arange(seq_len).to(device)
        pos_emb = model.positional_encoding(range_pos_toks)
    elif pos_enc == "rpe":
        pos_emb = model.layers[0].MHA.PEK.pe[:(seq_len+1)]
    

    similar = pos_emb @ pos_emb.t()
    similar = similar.detach().cpu()
    plt.imshow(np.abs(similar))
    plt.show()

def get_emb_sim(vocab_size, model, device):
    range_toks = torch.arange(vocab_size).to(device)
    toks = model.embed(range_toks)
    similar = toks @ toks.t()
    similar = similar.detach().cpu()
    plt.imshow(np.abs(similar))
    plt.show()

def plot_adj_heatmap(adj_mat):
    plt.figure(figsize=(6, 6))
    sns.heatmap(adj_mat, annot=False, cmap='viridis', fmt='.2f', linewidths=0.05, cbar=False)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.title('Matrix Heatmap')
    plt.show()



