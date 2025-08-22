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
import numpy as np
from replay_buffer import Ensemble_Buffer
from atari_utils import *

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
        
        self.variance = 0
        # this will measure the steps of calculating the variance before update
        # we are tracking this steps in a different variable than the self.steps
        #since we will reset that at the begining of every evaluatino episode
        self.steps_til_update = 0 

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
        self.current_head = np.random.choice(self.K, size = 1)[0]
        self.buffer = Ensemble_Buffer(buffer_size, batch_size, device, beta, K = self.K)

    def update_variance(self, variance):    
        '''method that keeps track of a running average
        of the variance. In some algorithms (e.g., DQN BDQN etc)
        this will be zero. However, in some other is usefull.'''

        self.variance = (self.variance * self.steps_til_update + variance)/(self.steps_til_update + 1)
        self.steps_til_update += 1
        return
    
    def get_variance(self):
        variance = self.variance
        self.reset_variance()
        return variance
    
    def reset_variance(self,):
        self.variance = 0
        self.steps_til_update = 0
        return

    def get_action(self, state):
        '''
        A method that uses as an input a normilized state and
        returns an action for exploration. In Bootstrap DQN this
        action is greedily selected by the currevt head
        '''
        #warmup action 
        if self.steps < self.head_start:
            return np.random.choice(self.num_actions, size = 1)[0] #in that way we do not allow the agent to use the do nothing action
        state = torch.from_numpy(state).to(self.device)
        qs = self.policy_net(x = state, k = self.current_head).detach().cpu().numpy()
        action = np.argmax(qs) 
        return action

    def get_clear_action(self, state):
        '''
        A method that uses as an input a normilized state and
        returns an action for evaluation. In Bootstrap DQN this
        action is the result of the majority voting
        between all heads.
        '''
        if self.steps < self.head_start:
            return np.random.choice(self.num_actions, size = 1)[0]
        state = torch.from_numpy(state).to(self.device)
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
                next_qs = next_q_vals.gather(1, next_actions)
                
                next_qs = ensure_flat(next_qs)
                rewards = ensure_flat(rewards)
                dones = ensure_flat(dones)

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
        


    def reset(self, ):
        self.current_head = np.random.choice(self.K, size = 1)[0]


    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename + "_policy_network")
        torch.save(self.opt.state_dict(), filename + "_policy_network_optimizer")
        
class DQN(BootDQN):
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
                 head_start,
                 #for the Bpolztamnn exploration,
                 exploration_duration,
                 initial_tau = 1_000,
                 final_tau = 0.5,
                 epsilon_greedy = True):
        '''
        DQN is a special case of BootDQN. To enhance exploration, we also use the Boltzmann exploration.
        Hence, some more arguments are required to set up the Boltzmann exploration
        '''
        super().__init__(   K=1,
                            num_actions=num_actions, 
                            num_channels = num_channels, 
                            learning_rate = learning_rate, 
                            device = device, 
                            buffer_size = buffer_size, 
                            batch_size = batch_size, 
                            beta = 1, 
                            train_frequency=train_frequency, 
                            target_update=target_update, 
                            head_start = head_start)

        self.exploration_duration = exploration_duration
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.epsilon_greedy = epsilon_greedy
        if self.epsilon_greedy:
            self.epsilon = np.linspace(1, 0.05, self.exploration_duration)
        
        else:
            self.temperature = np.linspace(self.initial_tau, self.final_tau, self.exploration_duration)

    
    def get_action(self, state):    
        '''
        A method that uses as an input a normilized state and
        returns an action for exploration. In Bootstrap DQN this
        action is greedily selected by the currevt head
        '''
        #warmup action 
        if self.steps < self.head_start:
            return np.random.choice(self.num_actions, size = 1)[0]
        
        if self.epsilon_greedy:
            eps = self.epsilon[self.steps if self.steps < self.epsilon.shape[0] else -1]
            if  np.random.rand() < eps:
                #return a random action
                return np.random.choice(self.num_actions, size = 1)[0]
            else:
                return self.get_clear_action(state)
        #otherwise, we have boltzamm exploration
        state = torch.from_numpy(state).to(self.device)
        qs = self.policy_net(x = state, k = self.current_head).detach().cpu().numpy()
        qs = qs.reshape(-1)
        
        tau = self.temperature[self.steps if self.steps < self.temperature.shape[0] else -1]
        probs = np.exp(qs/tau)
        probs/=np.sum(probs)
        action = np.random.choice(self.num_actions, p = probs, size = 1)[0]
        return action


    def get_clear_action(self, state):
        '''
        return the action fpr the evaluation. In DQN, this action is the one
        that maximizes the Q function
        '''
        if self.steps < self.head_start:
            return np.random.choice(self.num_actions, size = 1)[0]
        state = torch.from_numpy(state).to(self.device)
        qs = self.policy_net(x = state, k = self.current_head).detach().cpu().numpy()
        return np.argmax(qs)



class BootUCB(BootDQN):
    def __init__(self, K, 
                 num_actions, 
                 num_channels, 
                 learning_rate, 
                 device, 
                 buffer_size, 
                 batch_size, 
                 beta, 
                 train_frequency, 
                 target_update, 
                 head_start,
                 lambda_value):
        super().__init__(K, 
                         num_actions, 
                         num_channels, 
                         learning_rate, 
                         device, 
                         buffer_size, 
                         batch_size, 
                         beta, 
                         train_frequency, 
                         target_update, 
                         head_start)

        self.lambda_value = lambda_value

    def get_action(self, state):
        if self.steps < self.head_start:
            return np.random.choice(self.num_actions, size = 1)[0]
        

        state = torch.from_numpy(state).to(self.device)
        qs = self.policy_net(x = state) #qs now is a list of torch.Tensors
        
        qs = torch.stack(qs, dim=0)  # shape: (10, 1, n_actions)
        qs_mean = qs.mean(dim=0)      # shape: (1, n_actions)
        qs_mean = qs_mean.squeeze(0)

        qs_std = qs.std(dim=0)
        qs_std = qs_std.squeeze(0)

        upper_bound = (qs_mean + self.lambda_value*qs_std).detach().cpu().numpy()
        action = np.argmax(upper_bound)
        
        self.update_variance(qs_std[action].detach().cpu().numpy())

        return action



def ensure_flat(x: torch.Tensor) -> torch.Tensor:
    """
    Ensures x has shape (N,) instead of (N,1).
    If already (N,), nothing happens.
    If (N,1), squeeze last dim.
    """
    if x.ndim == 2 and x.shape[1] == 1:
        return x.squeeze(1)   # (N,1) -> (N,)
    elif x.ndim == 1:
        return x              # already (N,)
    else:
        raise ValueError(f"Unexpected shape {x.shape}, expected (N,) or (N,1).")

    
    

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
