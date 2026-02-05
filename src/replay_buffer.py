
import torch
import random
import numpy as np 
from collections import deque
from operator import itemgetter
import sys


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, store_intrisic):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            device (torch.device)
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.store_intrisic = store_intrisic
        
    
    def add(self, state, action, reward, next_state, done, intrisic=None):
        """Add a new experience to memory."""
        if self.store_intrisic:
            if intrisic is None : sys.exit(-1)
            self.memory.append([state, action, reward, next_state, done, intrisic])
            return
        self.memory.append([state, action, reward, next_state, done])
    
    
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if(len(self.memory)<self.batch_size): return False
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        if self.store_intrisic:
            intr_rewards = torch.from_numpy(np.vstack([e[5] for e in experiences if e is not None])).float().to(self.device)
            return (states, actions, rewards, next_states, dones, intr_rewards)
        return (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Ensemble_Buffer(ReplayBuffer):
    def __init__(self, buffer_size, batch_size, device, beta,K):
        super().__init__(buffer_size, batch_size, device, store_intrisic = False)
        self.beta = beta
        self.K = K

    def add(self, state, action, reward, next_state, done):
        mask = np.random.binomial(1, self.beta, self.K)
        self.memory.append([state, action, reward, next_state, done,mask])
        return
    
    def sample(self):
        if(len(self.memory)<self.batch_size): return False
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        masks = torch.from_numpy(np.vstack([e[5] for e in experiences if e is not None])).float().to(self.device)
        return (states, actions, rewards, next_states, dones, masks)
        

class Priorized_Experience_Replay():
    def __init__(self,buffer_size, batch_size, device, offset):

        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.index= 0
        self.alpha = 0
        self.beta = 0
        self.offset = offset
        


    def set_alpha(self, alpha):
        self.alpha = alpha
    def set_beta(self, beta):
        self.beta = beta

    def add(self, state, action, reward, next_state, done):
        #insert into memory
        if len(self.memory)==self.buffer_size-1:
            self.memory.insert(self.index,[state, action, reward, next_state, done])
        else:
            self.memory.append([state, action, reward, next_state, done])  
        #set a maximum propabillity (in order to train at least one time)
        max_prio = self.priorities.max() if self.priorities.max() > 0 else 1.0
        self.priorities[self.index] = max_prio
        #update the index
        self.index = (self.index + 1)%self.buffer_size
        return
    
    def sample(self):
        if(len(self.memory)<self.batch_size): return False
        mem_len = len(self.memory)
        #sampling the indices
        prior = self.priorities if mem_len == self.buffer_size else self.priorities[:self.index]
        p = prior**self.alpha
        p = (p) /(p.sum())
  
        indices = np.random.choice(np.arange(mem_len) , self.batch_size, p=p, replace=False)
        #getting the tuples

        exp = [self.memory[i] for i in indices]

        #Compute importance-sampling weight
        weights  = (mem_len * p[indices]) ** (-self.beta)
        # normalize weights
        weights /= weights.max() 

        states = torch.from_numpy(np.stack([e[0] for e in exp if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in exp if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in exp if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e[3] for e in exp if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in exp if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones, indices, weights)
    
    def update_priorities(self, indices, new_priorities):
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = prio+self.offset 

    def store(self):
        return self.priorities, self.index

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    

