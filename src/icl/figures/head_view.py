import json
import os
import uuid
import torch
from IPython.display import display, HTML, Javascript

from icl.utils.train_utils import get_attn_base



# Adapted from https://github.com/jessevig/bertviz/tree/master
def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    max_heads = 0
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        max_heads = max(max_heads, layer_attention.shape[0])
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    for i in range(len(squeezed)):
        if squeezed[i].shape[0] < max_heads:
            padding = torch.zeros((max_heads - squeezed[i].shape[0], squeezed[i].shape[1], squeezed[i].shape[2]), device=squeezed[i].device)
            squeezed[i] = torch.cat((squeezed[i], padding))
            print(squeezed[i].shape)
    return torch.stack(squeezed)


def num_heads(attention):
    return attention[0][0].size(0)




# Adapted from https://github.com/jessevig/bertviz/tree/master
def head_view(attention, mask=None, tokens=None, layer=None, heads=None, include_layers=None, html_action='view'):

    attn_data = []
    if tokens is None: raise ValueError("'tokens' is required")
    if include_layers is None: include_layers = list(range(len(attention))) # include all layers
    attention = format_attention(attention, include_layers)
    if mask is None:
        mask = [[0] * len(tokens)]
    
    attn_data.append(
        {
            'name': None,
            'attn': attention.tolist(),
            'left_text': tokens,
            'right_text': tokens,
            'mask': mask
        }
    )


    if layer is not None and layer not in include_layers:
        raise ValueError(f"Layer {layer} is not in include_layers: {include_layers}")

    # Generate unique div id to enable multiple visualizations in one notebook
    vis_id = 'bertviz-%s'%(uuid.uuid4().hex)

    # Compose html
    if len(attn_data) > 1:
        options = '\n'.join(
            f'<option value="{i}">{attn_data[i]["name"]}</option>'
            for i, d in enumerate(attn_data)
        )
        select_html = f'Attention: <select id="filter">{options}</select>'
    else:
        select_html = ""
    
    vis_html = f"""      
        <div id="{vis_id}" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
            <span style="user-select:none">
                Layer: <select id="layer"></select>
                {select_html}
            </span>
            <div id='vis'></div>
        </div>
    """

    for d in attn_data:
        attn_seq_len_left = len(d['attn'][0][0])
        if attn_seq_len_left != len(d['left_text']):
            raise ValueError(
                f"Attention has {attn_seq_len_left} positions, while number of tokens is {len(d['left_text'])} "
                f"for tokens: {' '.join(d['left_text'])}"
            )



    params = {
        'attention': attn_data,
        'default_filter': "0",
        'root_div_id': vis_id,
        'layer': layer,
        'heads': heads,
        'include_layers': include_layers
    }

    # require.js must be imported for Colab or JupyterLab:
    if html_action == 'view':
        display(HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'))
        display(HTML(vis_html))
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(os.path.join(__location__, 'head_view.js'), encoding='utf-8') as f:
            vis_js = f.read().replace("PYTHON_PARAMS", json.dumps(params))  
        display(Javascript(vis_js))

    elif html_action == 'return':
        html1 = HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>')

        html2 = HTML(vis_html)

        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(os.path.join(__location__, 'head_view.js'), encoding='utf-8') as f:
            vis_js = f.read().replace("PYTHON_PARAMS", json.dumps(params))
        html3 = Javascript(vis_js)
        script = '\n<script type="text/javascript">\n' + html3.data + '\n</script>\n'

        head_html = html1.data + html2.data + script
        return head_html

    else:
        raise ValueError("'html_action' parameter must be 'view' or 'return")


def get_head_view(model, config, mask=None, train_results=None, sampler=None, trunc=None, action='view', batch=None):
    if train_results is None:
        if sampler is None:
            raise ValueError("Either 'train_results' or 'sampler' must be provided.")
        else:
            train_results = {'sampler': sampler}
    else:
        sampler = train_results['sampler']
    if batch is None:
        batch, mask = sampler.generate(mode="test", num_samples=1)
    attn_map = get_attn_base(model, batch)

    if trunc is None:
        trunc = config.seq_len
    
    attn_tensors = [torch.zeros((1,config.model.num_heads[l],config.seq_len,config.seq_len)) for l in range(config.model.num_layers)]
    for l, attn in attn_map.items():
        attn = attn.unsqueeze(0)  # Number of heads in this layer
        attn_tensors[l][:1, :config.model.num_heads[l], :, :] = attn  # Fill attention tensor

    trunc_attn = [attn_tensors[l][:1, :config.model.num_heads[l], :trunc, :trunc] for l in range(config.model.num_layers)]
    # trunc_attn = [trunc_attn[l]/trunc_attn[l].sum(dim=-1, keepdims=True) for l in range(config.model.num_layers)]

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            if mask.dim() >= 2:
                mask = mask.squeeze()
            mask = mask.cpu().numpy().tolist()
            mask = [0 if m == 0 else 1 for m in mask]  # Convert to binary mask

        mask = mask[:trunc]

    if action == 'view':
        head_view(trunc_attn, mask, batch[0].tolist()[:trunc], html_action=action)
    else:
        return head_view(trunc_attn, mask, batch[0].tolist()[trunc:], html_action=action)