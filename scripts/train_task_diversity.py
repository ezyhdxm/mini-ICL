import torch
from absl import logging
import numpy as np

from icl.linear.train_linear import train
from icl.linear.lr_config import get_config
from icl.utils.linear_ood_analysis import process_ood_evolve_task_diversity

logging.set_verbosity(logging.INFO)
torch.set_printoptions(precision=3, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)


def main():
    exp_names = ['train_244beee6c29be8ab4f96771fdb79342f',
                'train_596c3de4dd0c1f079982dec494417f92',
                'train_eb3b586c12d500ad1d3a3262c313a6a9',
                'train_cbcd1c0ddd2348f736f6c6098cb69faa',
                'train_17121002a85d5cddfec36dcbeae81c10',
                'train_1f47a24f3c5b1a0a4a20aee54df9750c',
                'train_cf0da7951f42f3b180a424ca5d8a48cd',
                'train_f30c3e6ff072ca8d5aba56abcd002f44',
                'train_96d53e0d0887655ce2a40bc86f4ff567',
                'train_4de979b8d7914639afda3b23ee7d5ba9',
                'train_ccf7f6f1a7749d22a6a053f39affa348',
                'train_6ab65809d5e5b5fe12b3488cb7cc0ede']
    
    print(f"\nProcessing OOD evolution for {len(exp_names)} experiments...")
    result_dict_task = process_ood_evolve_task_diversity(exp_names)
    
    print("\nResults:")
    print(result_dict_task)
    
    return result_dict_task


if __name__ == "__main__":
    main()
