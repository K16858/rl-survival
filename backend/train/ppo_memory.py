import numpy as np

class PPOMemory:
    """Memory buffer for PPO"""
    
    def __init__(self, batch_size: int):
        self.visions = []
        self.statuses = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, vision, status, action, prob, val, reward, done):
        """Store episode data"""
        self.visions.append(vision)
        self.statuses.append(status)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        """Clear the memory"""
        self.visions = []
        self.statuses = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        """Return indices for batch generation"""
        n_states = len(self.visions)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches
