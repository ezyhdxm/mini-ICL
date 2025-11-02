import torch
import torch.nn.functional as F
from typing import Tuple
import pandas as pd
from itertools import product
from IPython.display import display


from icl.latent_markov.random_distributions import *

# TODO: maybe switch to JAX in the future?

# Simple Markov chain sampler
class MarkovSampler:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.trans = {}
        self.order = config.task.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.device = config.device
        dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device)*config.task.alpha)
        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        # Sample all transition probabilities in one go
        self.trans_matrix = dirichlet_dist.sample((self.num_states_order,))  # Shape: (num_states_order, num_states)
        self.trans_matrix /= self.trans_matrix.sum(dim=1, keepdim=True)
    
    def generate(self, epochs=1, mode:str="train")-> torch.Tensor:
        num_samples = self.batch_size if mode == "train" else self.test_size
        num_samples *= epochs
        
        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        
        # Initialize the state (randomly choose starting states for each sequence)
        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)
        samples[:, :self.order] = state
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state*self.powers, dim=1)
            probs = self.trans_matrix[state_indices]  # Shape: (num_samples, num_states)
            
            # Sample the next states for the entire batch
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            
            # Update the state window (shift left and append the new state)
            # state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
            state[:, :-1] = state[:, 1:]  # Shift left
            state[:, -1] = next_states    # Append new state
            
        return samples.reshape(epochs, -1, self.seq_len), probs.reshape(epochs, -1, self.num_states)

def markov_generate_unjitted(trans_matrix:torch.Tensor, num_samples:int, seq_len:int, num_states:int, order:int, device:str, epochs:int=1)->Tuple[torch.Tensor, torch.Tensor]:
    # num_samples = self.batch_size if mode == "train" else self.test_size
        
    # Initialize the samples tensor
    num_samples *= epochs
    powers = (num_states ** torch.arange(order - 1, -1, -1, device=device)).long()
    samples = torch.zeros((num_samples, seq_len), dtype=torch.long, device=device)
    
    # Initialize the state (randomly choose starting states for each sequence)
    state = torch.randint(high=num_states, size=(num_samples, order), device=device)
    samples[:, :order] = state
    probs = torch.zeros((num_samples, num_states), device=device)
        
    for t in range(order, seq_len):
        state_indices = torch.sum(state*powers, dim=1)
        probs = trans_matrix[state_indices]  # Shape: (num_samples, num_states)
        
        # Sample the next states for the entire batch
        next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        # Update the sequence with the sampled next states
        samples[:, t] = next_states
        
        # Update the state window (shift left and append the new state)
        # state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
        state[:, :-1] = state[:, 1:]  # Shift left
        state[:, -1] = next_states    # Append new state
        
    return samples.reshape(epochs, -1, seq_len), probs.reshape(epochs, -1, num_states)


# ICL Markov chain sampler
class ICLMarkovSampler:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.order = config.task.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.device = config.device
        self.alpha = config.task.alpha
        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        self.dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device)*self.alpha)
    
    def get_stationary(self, pi: torch.Tensor)->torch.Tensor:
        pi_t = pi.transpose(1, 2)  # Transpose each matrix, Shape: (num_samples, num_states, num_states_order)
        svd_input = pi_t - torch.eye(self.num_states, device=self.device).unsqueeze(0)
        _, _, v = torch.linalg.svd(svd_input)
        mu = torch.abs(v[:, -1, :])  # Last singular vector for each matrix
        return mu / mu.sum(dim=1, keepdim=True)

    def generate(self, mode="train", epochs=1):
        num_samples = self.batch_size if mode == "train" else self.test_size
        num_samples *= epochs
        range_vecs = torch.arange(num_samples, device=self.device)

        # Sample all transition probabilities in one go
        trans_matrix = self.dirichlet_dist.sample((num_samples, self.num_states_order,))  # Shape: (num_samples, num_states_order, num_states)
        trans_matrix /= trans_matrix.sum(dim=-1, keepdim=True)

        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)

        if self.order == 1:
            mu = self.get_stationary(trans_matrix) # Shape: (num_samples, num_states)
            state = torch.multinomial(mu, num_samples=1) # Shape: (num_samples,1)
            samples[:, :self.order] = state
        else:
            # Initialize the state (randomly choose starting states for each sequence)
            state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device) # Shape: (num_samples, order)
            samples[:, :self.order] = state
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state * self.powers, dim=1) #shape: (num_samples,)
            probs = trans_matrix[range_vecs, state_indices, :]  # Shape: (num_samples, num_states)
            
            # Sample the next states for the entire batch
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            
            # Update the state window (shift left and append the new state)
            state[:, :-1] = state[:, 1:]  # Shift left
            state[:, -1] = next_states    # Append new state
        
        return samples.reshape(epochs, -1, self.seq_len), probs.reshape(epochs, -1, self.num_states)













        