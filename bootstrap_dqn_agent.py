'''
In this file, the classic bootsrap dqn agent will be implemented. 
Then, we will also implement some extensions of the bootstrap dqn such as the ucb, the standard DQN with boltzmann, eps-greedy and even IM.
Finally we will also create the ADEU instances using the UCB, the IM and maybe an active exploration or even sth that replicates well known 
adaptive exploration frameowrks.
'''
from network import EnsembleNet
import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import normilize_state_atari, majority_vote_action
import numpy as np
from replay_buffer import Ensemble_Buffer

class BootDQN():

    def __init__(self,
                 K,
                 num_actions,
                 num_channels,
                 learning_rate,
                 device,
                 buffer_size,
                 batch_size,
                 beta, 
                 train_frequency, 
                 target_update,
                 head_start):
        
        self.policy_net = EnsembleNet(n_ensemble=K,
                                      n_actions=num_actions,
                                      num_channels=num_channels).to(device)
        self.target_net = EnsembleNet(n_ensemble=K,
                                      n_actions=num_actions,
                                      num_channels=num_channels).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.beta = beta
        self.num_actions = num_actions
        self.clip_gradient = 5
        self.device = device
        self.gamma = 0.99
        self.train_frequency = train_frequency
        self.K = K
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.head_start = head_start
        self.steps = 0 #counts the total steps
        self.opt = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.buffer = Ensemble_Buffer(buffer_size, batch_size, device, beta, K = self.K)


    def get_action(self, state, active_head):
        state = state.to(self.device)
        state = normilize_state_atari(state)
        qs = self.policy_net(x = state, k = active_head).detach().cpu().numpy()
        action = np.argmax(qs)
        return action

    def get_clear_action(self, state):
        state = state.to(self.device)
        state = normilize_state_atari(state)
        q_heads = self.policy_net(x = state)
        q_heads = [t.detach().cpu().numpy() for t in q_heads]
        action = majority_vote_action(q_heads)
        return action

    def step(self, state, action, reward, next_state, done):
        loss = 0
        self.buffer.add(state, action, reward, next_state, done)
        self.steps+=1
        #perform a training step
        if (self.steps % self.train_frequency != 0) or (self.steps < self.head_start): return loss
        experience = self.buffer.sample()
        loss = self.train(experience)
        if (self.steps % self.target_update == 0) and (self.steps > self.head_start):
                    self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss
    
    def train(self, experience):
        states, actions, rewards, next_states, dones, masks = experience
        losses = [0.0 for _ in range(self.K)]
        self.opt.zero_grad()
        q_policy_vals = self.policy_net(states, None)
        next_q_target_vals = self.target_net(next_states, None)
        next_q_policy_vals = self.policy_net(next_states, None)
        cnt_losses = []
        for k in range(self.K):
            total_used = torch.sum(masks[:,k])
            if total_used > 0.0:
                next_q_vals = next_q_target_vals[k].data
                
                next_actions = next_q_policy_vals[k].data.max(1, True)[1]
                next_qs = next_q_vals.gather(1, next_actions).squeeze(1)
                

                preds = q_policy_vals[k].gather(1, actions).squeeze(1)
                targets = rewards + self.gamma * next_qs * (1-dones)
                l1loss = F.smooth_l1_loss(preds, targets, reduction='mean')
                full_loss = masks[:,k]*l1loss
                loss = torch.sum(full_loss/total_used)
                cnt_losses.append(loss)
                losses[k] = loss.cpu().detach().item()

        loss = sum(cnt_losses)/self.K
        loss.backward()
        for param in self.policy_net.core_net.parameters():
            if param.grad is not None:
                # divide grads in core
                param.grad.data *=1.0/float(self.K)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_gradient)
        self.opt.step()
        return np.mean(losses)
        


    
    

if __name__ == '__main__':
     
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    print(f'this experiment will run in {device}')
    
    parameters = {
                'K':10,
                'num_actions':3,
                'num_channels':4,
                'learning_rate':6.25e-6,
                'device':device,
                'buffer_size':int(1e6),
                'batch_size':32,
                'beta':0.99, 
                'train_frequency':4, 
                'target_update':10000,
                'head_start':64,
    }


    i = 0
    agent = BootDQN(**parameters)

    while i<15_000_000:
        state = np.random.rand(4,84,84).astype(np.float32)
        state = torch.from_numpy(state)
        action = agent.get_action(state, active_head = 3)
        action = agent.get_clear_action(state)
        next_state = np.random.rand(4,84,84).astype(np.float32)
        next_state = torch.from_numpy(next_state)
        agent.step(state, action, 0.5, next_state, False)
        i+=1
