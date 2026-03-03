import torch
import torch.nn.functional as F
from icl.latent_markov.markov import *
# import pandas as pd
# from itertools import product
# from IPython.display import display
from tqdm.notebook import trange, tqdm

#TODO: OUTDATED!

# Empirical n-gram learner
class ngramLearner:
    def __init__(self, config, order, is_icl=False):
        self.order = order
        self.vocab_size = config.vocab_size
        self.alpha = config.task.alpha
        self.num_states_order = config.vocab_size**self.order
        self.device = config.device
        self.is_icl = is_icl
        
        if self.order > 0:
            if not is_icl:
                self.trans_mat_est = self.alpha * torch.ones((self.num_states_order, self.vocab_size), device=self.device) # (num_states_order, num_states)
            self.state_powers = self.vocab_size ** torch.arange(self.order - 1, -1, -1, device=self.device)
            
        else:
            self.trans_mat_est = self.alpha*torch.ones((self.vocab_size,), device=self.device)
    
    def update(self, batch): # batch: (B,T)
        batch_size, seq_len = batch.shape
        if self.order > 0:
            if self.is_icl:
                self.trans_mat_est = self.alpha * torch.ones((batch_size, self.num_states_order, self.vocab_size), device=self.device)
            states = torch.stack([batch[:, t:t + self.order] for t in range(seq_len - self.order)], dim=1)  # (B, T-O, O)
            next_states = batch[:, self.order:]  # (B, T-O)

            # Compute state indices as base-vocab_size numbers
            state_indices = torch.sum(states * self.state_powers, dim=2)  # (B, T-O)
            values = torch.ones_like(state_indices[:,0], dtype=torch.float, device=self.device)  # Same size as positions
            # Update transition matrix
            for t in range(state_indices.size(1)):  # Loop over sequence length (T-O)
                # Add values to the specified positions
                if not self.is_icl:
                    self.trans_mat_est.index_put_((state_indices[:,t], next_states[:,t]), values, accumulate=True)
                else:
                    self.trans_mat_est.index_put_((torch.arange(batch_size), state_indices[:,t], next_states[:,t]), values, accumulate=True)
            if self.is_icl:   
                self.trans_mat_est /= self.trans_mat_est.sum(dim=-1, keepdim=True)
        else:
            if not self.is_icl:
                self.trans_mat_est += torch.bincount(batch.flatten(), minlength=self.vocab_size)
            else:
                bin_counts = torch.stack([torch.bincount(batch[i], minlength=self.vocab_size) for i in range(batch_size)])
                self.trans_mat_est = bin_counts / (bin_counts.sum(dim=-1, keepdim=True)+1e-6)
                
    def predict(self, batch):
        batch_size, seq_len = batch.size()
        if self.order > 0:
            probs = torch.zeros((batch_size, seq_len, self.vocab_size), device=self.device) # (B, T, N)
            uniform = torch.ones((self.vocab_size,), device=self.device) / self.vocab_size # N
            probs[:,:self.order,:] = uniform.repeat(batch_size, self.order, 1)
            states = torch.stack([batch[:, t:t+self.order] for t in range(seq_len-self.order)], dim=1) # (B, T-O, O)
            state_indices = torch.sum(states * self.state_powers, dim=2)  # (B, T-O)
            if not self.is_icl:
                probs[:, self.order:] = self.trans_mat_est[state_indices] / self.trans_mat_est[state_indices].sum(dim=-1, keepdim=True)
            else:
                batch_indices = torch.arange(batch_size).unsqueeze(1)
                probs[:, self.order:] = self.trans_mat_est[batch_indices, state_indices] 
            return probs

        else:
            if not self.is_icl:
                targets = batch.reshape(-1)
                probs = self.trans_mat_est / self.trans_mat_est.sum()
                probs = probs.unsqueeze(0).repeat(targets.size(0), 1)
                return probs.reshape(batch_size, seq_len, self.vocab_size)
            else:
                probs = self.trans_mat_est.unsqueeze(1).repeat(1, seq_len, 1)
                return probs
            
    def loss(self, batch):
        probs = self.predict(batch)
        one_hot_labels = F.one_hot(batch, num_classes=self.vocab_size).float()
        loss = -torch.sum(one_hot_labels * torch.log(probs+1e-13)) / (batch.size(0) * batch.size(1))
        return loss

# n-gram learner for latent markov chain
class many_ngramLearners:
    def __init__(self, config, order, sampler):
        self.order = order
        self.sampler = sampler
        self.markov_sampler = MarkovSampler(config)
        self.config = config
    
    def loss(self):
        total_trans = self.sampler.total_trans
        task_inds = torch.arange(total_trans)
        if total_trans > 16:
            task_inds = torch.randperm(total_trans)[:16]
        
        loss = 0
        for i in tqdm(task_inds, leave=False, desc=f"Evaluating Baseline with order {self.order}"):
            ngram_learner = ngramLearner(self.config, self.order, is_icl=False)
            batch, _ = self.sampler.generate(num_samples=256, mode="eval", task=i)
            ngram_learner.update(batch[:128])
            loss += ngram_learner.loss(batch[128:]).item()
        return loss / len(task_inds)



