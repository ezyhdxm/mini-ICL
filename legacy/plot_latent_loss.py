"""
Script to plot eval/Latent_true "Transformer | True" against evaluation steps.
"""

import json
import plotly.graph_objects as go
import numpy as np
import sys
import os

def plot_latent_loss(log_path, save_path=None, show=True):
    """
    Plot eval/Latent_true "Transformer | True" against evaluation steps.
    
    Args:
        log_path: Path to the log.json file
        save_path: Optional path to save the plot (HTML format for plotly)
        show: Whether to display the plot
    
    Returns:
        plotly.graph_objects.Figure: The plotly figure object
    """
    # Load the log file
    print(f"Loading log file from: {log_path}")
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # Extract the metric
    if 'eval/Latent_true' not in log_data:
        raise KeyError("'eval/Latent_true' not found in log file")
    
    latent_true = log_data['eval/Latent_true']
    if 'Transformer | True' not in latent_true:
        raise KeyError("'Transformer | True' not found in eval/Latent_true")
    
    metric_data = latent_true['Transformer | True']
    print(f"Found metric data with {len(metric_data)} evaluation points")
    
    # Check if metric_data is a list of lists (multiple values per step) or a flat list
    if metric_data and isinstance(metric_data[0], list):
        # If it's a list of lists, we need to aggregate (e.g., take mean)
        # Each inner list might be values for different batches or runs
        print(f"Metric data is nested. First element has {len(metric_data[0])} values")
        # Take mean across the inner dimension
        metric_values = [np.mean(values) if values else np.nan for values in metric_data]
        print(f"Aggregating by taking mean across inner dimension")
    else:
        metric_values = metric_data
        print(f"Metric data is flat with {len(metric_values)} values")
    
    # Get evaluation steps - use train/step which contains the training step numbers
    # at which evaluations occurred
    if 'train/step' in log_data:
        eval_steps = log_data['train/step']
        print(f"Using train/step for evaluation steps. Found {len(eval_steps)} evaluation points")
    elif 'eval/step' in log_data:
        eval_steps = log_data['eval/step']
        print(f"Using eval/step. Found {len(eval_steps)} steps")
    else:
        # Fallback to indices if neither is available
        print("Warning: No train/step or eval/step found. Using indices as evaluation steps.")
        eval_steps = list(range(len(metric_values)))
    
    # Ensure lengths match
    min_len = min(len(eval_steps), len(metric_values))
    eval_steps = eval_steps[:min_len]
    metric_values = metric_values[:min_len]
    
    if len(eval_steps) != len(metric_values):
        print(f"Warning: Mismatch in lengths. Steps: {len(eval_steps)}, Values: {len(metric_values)}")
    
    print(f"Plotting {min_len} data points")
    
    # Create the plotly plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=eval_steps,
        y=metric_values,
        mode='lines+markers',
        name='eval/Latent_true "Transformer | True"',
        marker=dict(size=4),
        line=dict(width=2),
        hovertemplate='Step: %{x}<br>Loss: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Loss: eval/Latent_true "Transformer | True" vs Evaluation Steps',
        xaxis_title='Training Step (Evaluation Step)',
        yaxis_title='Loss: eval/Latent_true "Transformer | True"',
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=600,
    )
    
    # Save if requested
    if save_path:
        if save_path.endswith('.html') or not save_path.endswith(('.png', '.jpg', '.pdf')):
            # Default to HTML for plotly
            if not save_path.endswith('.html'):
                save_path = save_path + '.html'
            fig.write_html(save_path)
        else:
            # For image formats, use plotly's image export (requires kaleido)
            fig.write_image(save_path)
        print(f"Plot saved to: {save_path}")
    
    if show:
        fig.show()
    
    return fig

if __name__ == '__main__':
    # Default to the experiment the user has open
    default_log_path = 'results/linear/train_6ab65809d5e5b5fe12b3488cb7cc0ede/log.json'
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = default_log_path
    
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        print(f"Usage: python plot_latent_loss.py <log_path> [save_path]")
        sys.exit(1)
    
    plot_latent_loss(log_path, save_path=save_path, show=True)

