import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from ipywidgets import interact, widgets
import IPython.display as display

def create_interactive_attention_viewer(attns, include_rollout=True, include_partial_rollouts=True, residual_mode='both'):
    """
    Create an interactive attention viewer with dropdown layer selection and attention rollout.
    
    Args:
        attns: Dictionary of attention tensors {layer_idx: attention_tensor}
        include_rollout: Whether to include full attention rollout visualization
        include_partial_rollouts: Whether to include partial rollouts (0->1, 0->1->2, etc.)
        residual_mode: 'both', 'with', 'without', or 'none' - controls residual connection in rollouts
    """
    
    # Convert attention data to numpy arrays
    attention_data = {}
    for layer_key, tensor in attns.items():
        attention_data[layer_key] = tensor.squeeze(0).cpu().numpy()
    
    layer_keys = sorted(attention_data.keys())
    
    # Create Plotly figure with dropdown
    fig = go.Figure()
    
    # Add traces for each individual layer
    for i, layer_key in enumerate(layer_keys):
        fig.add_trace(
            go.Heatmap(
                z=attention_data[layer_key],
                colorscale='Viridis',
                colorbar=dict(title="Attention Weight"),
                hovertemplate='Layer ' + str(layer_key) + '<br>Query: %{y}<br>Key: %{x}<br>Attention: %{z:.4f}<extra></extra>',
                visible=(i == 0),  # Only first layer visible initially
                name=f'Layer {layer_key}'
            )
        )
    
    # Keep track of how many traces we've added
    num_traces = len(layer_keys)
    
    # Determine which rollout modes to include
    rollout_modes = []
    if residual_mode == 'both':
        rollout_modes = [(True, 'w/ residual'), (False, 'w/o residual')]
    elif residual_mode == 'with':
        rollout_modes = [(True, 'w/ residual')]
    elif residual_mode == 'without':
        rollout_modes = [(False, 'w/o residual')]
    
    # Add partial rollout traces if requested
    partial_rollouts = {}
    if include_partial_rollouts and len(layer_keys) > 1:
        for include_res, res_label in rollout_modes:
            for i in range(1, len(layer_keys)-1):
                up_to_layer = layer_keys[i]
                layers_included = layer_keys[:i+1]
                rollout_name = f'Rollout {layers_included[0]}→{"→".join(str(l) for l in layers_included[1:])} ({res_label})'
                partial_rollouts[rollout_name] = compute_attention_rollout(attns, up_to_layer, include_residual=include_res)
                
                colorscale = 'Blues' if include_res else 'Purples'
                
                fig.add_trace(
                    go.Heatmap(
                        z=partial_rollouts[rollout_name],
                        colorscale=colorscale,
                        colorbar=dict(title="Rollout Weight"),
                        hovertemplate=rollout_name + '<br>Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>',
                        visible=False,
                        name=rollout_name
                    )
                )
    
    # Add full attention rollout traces if requested
    if include_rollout:
        for include_res, res_label in rollout_modes:
            full_rollout = compute_attention_rollout(attns, include_residual=include_res)
            rollout_name = f'Full Rollout ({res_label})'
            colorscale = 'Reds' if include_res else 'Oranges'
            
            fig.add_trace(
                go.Heatmap(
                    z=full_rollout,
                    colorscale=colorscale,
                    colorbar=dict(title="Rollout Weight"),
                    hovertemplate=rollout_name + '<br>Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>',
                    visible=False,
                    name=rollout_name
                )
            )
    
    # Create dropdown menu
    dropdown_buttons = []
    
    # Add buttons for each individual layer
    for i, layer_key in enumerate(layer_keys):
        visibility = [j == i for j in range(num_traces)]
        # Add False for all rollout traces
        visibility.extend([False] * (len(fig.data) - num_traces))
        
        dropdown_buttons.append(
            dict(
                label=f'Layer {layer_key}',
                method='update',
                args=[
                    {'visible': visibility},
                    {'title': f'Attention Heatmap - Layer {layer_key}'}
                ]
            )
        )
    
    # Add buttons for partial rollouts
    if include_partial_rollouts:
        trace_idx = num_traces
        for rollout_name in partial_rollouts.keys():
            visibility = [False] * len(fig.data)
            visibility[trace_idx] = True
            trace_idx += 1
            
            dropdown_buttons.append(
                dict(
                    label=rollout_name,
                    method='update',
                    args=[
                        {'visible': visibility},
                        {'title': f'Attention {rollout_name}'}
                    ]
                )
            )
    
    # Add buttons for full attention rollout
    if include_rollout:
        # Count how many full rollout traces we have
        num_full_rollouts = len(rollout_modes)
        start_idx = len(fig.data) - num_full_rollouts
        
        for i, (include_res, res_label) in enumerate(rollout_modes):
            visibility = [False] * len(fig.data)
            visibility[start_idx + i] = True
            rollout_name = f'Full Rollout ({res_label})'
            
            dropdown_buttons.append(
                dict(
                    label=rollout_name,
                    method='update',
                    args=[
                        {'visible': visibility},
                        {'title': rollout_name}
                    ]
                )
            )
    
    # Update layout with dropdown
    fig.update_layout(
        title=f'Attention Heatmap - Layer {layer_keys[0]}',
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        width=800,
        height=800,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ]
    )
    
    return fig


def compute_attention_rollout(attns, up_to_layer=None, include_residual=True):
    """
    Compute attention rollout by multiplying attention matrices across layers.
    
    Args:
        attns: Dictionary of attention tensors {layer_idx: attention_tensor}
        up_to_layer: If specified, compute rollout only up to this layer (inclusive)
        include_residual: Whether to include residual connections in the rollout
    
    Returns:
        numpy array: Attention rollout matrix
    """
    layer_keys = sorted(attns.keys())
    
    if up_to_layer is not None:
        # Find the index of up_to_layer in sorted keys
        if up_to_layer in layer_keys:
            end_idx = layer_keys.index(up_to_layer) + 1
            layer_keys = layer_keys[:end_idx]
        else:
            print(f"Warning: Layer {up_to_layer} not found. Using all layers.")
    
    # Initialize with first layer
    rollout = attns[layer_keys[0]].squeeze(0).cpu().numpy()
    
    if include_residual:
        # Add residual connection (identity matrix)
        eye = np.eye(rollout.shape[0])
        rollout = 0.5 * (rollout + eye)
    
    # Multiply through layers
    for layer_key in layer_keys[1:]:
        attention = attns[layer_key].squeeze(0).cpu().numpy()
        if include_residual:
            attention = 0.5 * (attention + eye)  # Add residual connection
        rollout = np.matmul(attention, rollout)
    
    return rollout


def create_side_by_side_comparison(attns, layer1=None, layer2=None):
    """
    Create side-by-side comparison of two layers or layer vs rollout.
    
    Args:
        attns: Dictionary of attention tensors
        layer1: First layer index (if None, uses first available)
        layer2: Second layer index (if None, shows rollout)
    """
    layer_keys = sorted(attns.keys())
    
    if layer1 is None:
        layer1 = layer_keys[0]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'Layer {layer1}', 
                       f'Layer {layer2}' if layer2 else 'Attention Rollout'],
        shared_xaxes=True,
        shared_yaxes=True
    )
    
    # First subplot - Layer 1
    attention1 = attns[layer1].squeeze(0).cpu().numpy()
    fig.add_trace(
        go.Heatmap(
            z=attention1,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(x=0.45, len=0.9)
        ),
        row=1, col=1
    )
    
    # Second subplot - Layer 2 or Rollout
    if layer2:
        attention2 = attns[layer2].squeeze(0).cpu().numpy()
        colorscale = 'Viridis'
    else:
        attention2 = compute_attention_rollout(attns)
        colorscale = 'Reds'
    
    fig.add_trace(
        go.Heatmap(
            z=attention2,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(x=1.02, len=0.9)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Attention Comparison",
        height=500,
        width=1100
    )
    
    return fig


def create_jupyter_widget_viewer(attns):
    """
    Create an interactive Jupyter widget for attention visualization.
    Perfect for Jupyter notebooks with real-time interaction.
    """
    layer_keys = sorted(attns.keys())
    
    # Create output widget
    output = widgets.Output()
    
    # Create layer dropdown
    layer_options = [(f'Layer {k}', k) for k in layer_keys]
    
    # Add partial rollout options
    for i in range(1, len(layer_keys)-1):
        layers_included = layer_keys[:i+1]
        rollout_name = f'Rollout {layers_included[0]}→{"→".join(str(l) for l in layers_included[1:])}'
        layer_options.append((rollout_name, f'rollout_{i}'))
    
    # Add full rollout option
    layer_options.append(('Full Rollout (All Layers)', 'rollout_full'))
    
    layer_dropdown = widgets.Dropdown(
        options=layer_options,
        value=layer_keys[0],
        description='Select Layer:',
        style={'description_width': 'initial'}
    )
    
    # Create statistics checkbox
    show_stats = widgets.Checkbox(
        value=False,
        description='Show Statistics',
        disabled=False
    )
    
    # Create colormap dropdown
    colormap_dropdown = widgets.Dropdown(
        options=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'blues', 'reds', 
                 'greens', 'purples', 'oranges', 'greys', 'ylorbr', 'ylgnbu', 'gnbu', 
                 'bugn', 'pubugn', 'purd', 'rdpu', 'orrd', 'ylorrd', 'ylgn', 'solar',
                 'spectral', 'rdylbu', 'rdbu', 'prgn', 'piyg', 'brbg', 'rdgy',
                 'jet', 'turbo', 'rainbow', 'twilight', 'sunset', 'sunsetdark'],
        value='viridis',
        description='Colormap:',
        style={'description_width': 'initial'}
    )
    
    def update_plot(layer, show_statistics, colormap):
        with output:
            output.clear_output(wait=True)
            
            if isinstance(layer, str) and layer.startswith('rollout'):
                if layer == 'rollout_full':
                    data = compute_attention_rollout(attns)
                    title = 'Full Attention Rollout (All Layers)'
                else:
                    # Extract the number from rollout_N
                    rollout_idx = int(layer.split('_')[1])
                    up_to_layer = layer_keys[rollout_idx]
                    data = compute_attention_rollout(attns, up_to_layer)
                    layers_included = layer_keys[:rollout_idx+1]
                    title = f'Rollout {layers_included[0]}→{"→".join(str(l) for l in layers_included[1:])}'
            else:
                data = attns[layer].squeeze(0).cpu().numpy()
                title = f'Attention Layer {layer}'
            
            # Create figure
            fig = go.Figure(data=go.Heatmap(
                z=data,
                colorscale=colormap,
                colorbar=dict(title="Weight"),
                hovertemplate='Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Key Position",
                yaxis_title="Query Position",
                width=700,
                height=700,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
            
            fig.show()
            
            # Show statistics if requested
            if show_statistics:
                print(f"\n{title} Statistics:")
                print(f"Shape: {data.shape}")
                print(f"Min: {data.min():.4f}")
                print(f"Max: {data.max():.4f}")
                print(f"Mean: {data.mean():.4f}")
                print(f"Std: {data.std():.4f}")
                
                # Find most attended positions
                max_attention_per_query = data.max(axis=1)
                top_queries = np.argsort(max_attention_per_query)[-5:][::-1]
                print(f"\nTop 5 queries with highest attention:")
                for q in top_queries:
                    key_pos = np.argmax(data[q])
                    print(f"  Query {q} → Key {key_pos} (weight: {data[q, key_pos]:.4f})")
    
    # Create interactive widget
    interactive_plot = widgets.interactive(
        update_plot,
        layer=layer_dropdown,
        show_statistics=show_stats,
        colormap=colormap_dropdown
    )
    
    # Create layout
    controls = widgets.HBox([layer_dropdown, colormap_dropdown, show_stats])
    app = widgets.VBox([controls, output])
    
    # Initial plot
    update_plot(layer_keys[0], True, 'Viridis')
    
    return app


def analyze_attention_patterns(attns):
    """
    Analyze and visualize attention patterns across all layers.
    """
    layer_keys = sorted(attns.keys())
    num_layers = len(layer_keys)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Average Attention per Layer', 
                       'Max Attention per Layer',
                       'Attention Entropy per Layer',
                       'Diagonal Dominance per Layer'],
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Calculate metrics for each layer
    avg_attentions = []
    max_attentions = []
    entropies = []
    diagonal_dominances = []
    
    for layer_key in layer_keys:
        attention = attns[layer_key].squeeze(0).cpu().numpy()
        
        # Average attention
        avg_attentions.append(attention.mean())
        
        # Max attention
        max_attentions.append(attention.max())
        
        # Entropy (measure of attention dispersion)
        # Add small epsilon to avoid log(0)
        attention_norm = attention + 1e-10
        entropy = -np.sum(attention_norm * np.log(attention_norm), axis=1).mean()
        entropies.append(entropy)
        
        # Diagonal dominance (how much attention focuses on same position)
        diagonal_sum = np.trace(attention)
        total_sum = attention.sum()
        diagonal_dominances.append(diagonal_sum / total_sum if total_sum > 0 else 0)
    
    # Plot metrics
    fig.add_trace(
        go.Scatter(x=layer_keys, y=avg_attentions, mode='lines+markers', name='Avg Attention'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=layer_keys, y=max_attentions, mode='lines+markers', name='Max Attention'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=layer_keys, y=entropies, mode='lines+markers', name='Entropy'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=layer_keys, y=diagonal_dominances, mode='lines+markers', name='Diagonal Dom.'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Attention Pattern Analysis Across Layers",
        height=800,
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Layer", row=2, col=1)
    fig.update_xaxes(title_text="Layer", row=2, col=2)
    fig.update_yaxes(title_text="Average Attention", row=1, col=1)
    fig.update_yaxes(title_text="Max Attention", row=1, col=2)
    fig.update_yaxes(title_text="Entropy", row=2, col=1)
    fig.update_yaxes(title_text="Diagonal Dominance", row=2, col=2)
    
    return fig


# Main function to use
def visualize_attention(attns, mode='dropdown', include_partial_rollouts=True, residual_mode='both'):
    """
    Main function to visualize attention with various options.
    
    Args:
        attns: Dictionary of attention tensors
        mode: 'dropdown', 'widget', 'analysis', 'comparison', or 'all'
        include_partial_rollouts: Whether to include partial rollouts in dropdown/widget modes
        residual_mode: How to handle residual connections in rollouts:
            - 'both': Show rollouts with and without residual connections
            - 'with': Only show rollouts with residual connections
            - 'without': Only show rollouts without residual connections
            - 'none': Don't include any rollouts
    """
    if mode == 'dropdown':
        fig = create_interactive_attention_viewer(attns, include_rollout=True, 
                                                include_partial_rollouts=include_partial_rollouts,
                                                residual_mode=residual_mode)
        fig.show()
    
    elif mode == 'widget':
        # For Jupyter notebooks
        return create_jupyter_widget_viewer(attns)
    
    elif mode == 'analysis':
        fig = analyze_attention_patterns(attns)
        fig.show()
    
    elif mode == 'comparison':
        layer_keys = sorted(attns.keys())
        if len(layer_keys) >= 2:
            fig = create_side_by_side_comparison(attns, layer_keys[0], layer_keys[-1])
        else:
            fig = create_side_by_side_comparison(attns)  # Shows layer vs rollout
        fig.show()
    
    elif mode == 'all':
        print("1. Interactive Dropdown Viewer:")
        fig1 = create_interactive_attention_viewer(attns, include_rollout=True,
                                                 include_partial_rollouts=include_partial_rollouts,
                                                 residual_mode=residual_mode)
        fig1.show()
        
        print("\n2. Attention Pattern Analysis:")
        fig2 = analyze_attention_patterns(attns)
        fig2.show()
        
        print("\n3. Side-by-side Comparison:")
        fig3 = create_side_by_side_comparison(attns)
        fig3.show()
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: 'dropdown', 'widget', 'analysis', 'comparison', 'all'")


# Example usage:
# Show both with and without residual connections (default)
# visualize_attention(attns, mode='dropdown')

# Show only rollouts WITHOUT residual connections
# visualize_attention(attns, mode='dropdown', residual_mode='without')

# Show only rollouts WITH residual connections
# visualize_attention(attns, mode='dropdown', residual_mode='with')

# Disable partial rollouts but show full rollout comparison
# visualize_attention(attns, mode='dropdown', include_partial_rollouts=False, residual_mode='both')

# For Jupyter widget
# widget = visualize_attention(attns, mode='widget')

# Pattern analysis
# visualize_attention(attns, mode='analysis')

# Show everything
# visualize_attention(attns, mode='all', residual_mode='both')