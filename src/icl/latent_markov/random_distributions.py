import torch
from torch.distributions import Distribution

class RandomHotDistribution(Distribution):
    arg_constraints = {}

    def __init__(self, config, validate_args=None):
        super().__init__(batch_shape=torch.Size(), validate_args=validate_args)
        self.num_states = config.vocab_size
        self.card = config.task.cardinality
        self.prob = 1.0 / self.card
        assert self.card < self.num_states, "Cardinality must be less than number of states"

    def sample_index_comb_vec(self, total_samples):
        # Generate random values and sort to get top-card indices
        rand = torch.rand((total_samples, self.num_states))
        _, indices = torch.topk(rand, k=self.card, dim=1)
        return indices

    def sample(self, sample_shape=torch.Size()):
        shape = sample_shape if isinstance(sample_shape, torch.Size) else torch.Size(sample_shape)
        # Total number of samples to generate
        total_samples = int(torch.tensor(shape).prod())

        # Sample unique indices per distribution
        indices = self.sample_index_comb_vec(total_samples)

        # Create the output tensor
        output = torch.zeros(total_samples, self.num_states)
        for i in range(total_samples):
            output[i, indices[i]] = self.prob

        # Reshape to desired sample shape + num_states
        return output.view(*shape, self.num_states)
    
    def estimate_entropy(self, num_samples=3000):
        # Sample and flatten into 2D: (num_samples, num_states)
        samples = self.sample((num_samples,))
        
        # Turn each row into a hashable tuple
        tuples = [tuple(row.tolist()) for row in samples]

        # Count frequencies
        freq = {}
        for t in tuples:
            freq[t] = freq.get(t, 0) + 1

        probs = torch.tensor([count / num_samples for count in freq.values()])
        entropy = -torch.sum(probs * torch.log(probs + 1e-12))  # Add epsilon for stability
        return entropy.item()

    

class FiniteDirichletDistribution(Distribution):
    arg_constraints = {}

    def __init__(self, config, ood=False, validate_args=None):
        super().__init__(batch_shape=torch.Size(), validate_args=validate_args)
        self.alpha = config.task.random_alpha
        self.num_states = config.vocab_size
        if not ood:
            self.n_tasks = config.task.total_trans 
        else:
            self.n_tasks = 0
        if self.n_tasks > 0:
            self.task_pool = torch.distributions.dirichlet.Dirichlet(self.alpha*torch.ones(self.num_states)).sample((self.n_tasks,))

    def sample(self, sample_shape=torch.Size(), task=None):
        shape = sample_shape if isinstance(sample_shape, torch.Size) else torch.Size(sample_shape)
        total_samples = int(torch.tensor(shape).prod())

        # Sample from Dirichlet distribution
        
        if self.n_tasks > 0:
            if task is None:
                idxs = torch.randint(low=0, high=self.n_tasks, size=(total_samples,))
                samples = self.task_pool[idxs]
            else:
                assert task < self.n_tasks, "Task index out of bounds"
                samples = self.task_pool[task].unsqueeze(0).expand(total_samples, -1)
        else:
            samples = torch.distributions.dirichlet.Dirichlet(self.alpha*torch.ones(self.num_states)).sample((total_samples,))

        # Reshape to desired sample shape + num_states
        return samples.view(*shape, self.num_states)
    
    def estimate_entropy(self, num_samples=3000):
        # Sample and flatten into 2D: (num_samples, num_states)
        samples = self.sample((num_samples,))
        
        # Turn each row into a hashable tuple
        tuples = [tuple(row.tolist()) for row in samples]

        # Count frequencies
        freq = {}
        for t in tuples:
            freq[t] = freq.get(t, 0) + 1

        probs = torch.tensor([count / num_samples for count in freq.values()])
        entropy = -torch.sum(probs * torch.log(probs + 1e-12))  # Add epsilon for stability
        return entropy.item()
    


def get_dist(config, ood=False) -> Distribution:
    name = config.task.dist_name
    dists = {"finite_dirichlet": FiniteDirichletDistribution, "random_support": RandomHotDistribution}
    return dists[name](config, ood=ood)
