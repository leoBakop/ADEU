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
from tqdm import tqdm
from scipy.stats import zipf 

from torch.distributions.categorical import Categorical




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
                 head_start,
                 gamma, 
                 cnn = True,
                 state_size = -1):
        
        self.policy_net = EnsembleNet(n_ensemble=K,
                                      n_actions=num_actions,
                                      num_channels=num_channels,
                                      cnn = cnn,
                                      state_size=state_size).to(device)
        
        self.target_net = EnsembleNet(n_ensemble=K,
                                      n_actions=num_actions,
                                      num_channels=num_channels,
                                      cnn = cnn,
                                      state_size=state_size).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.variance = 0
        self.q_mean = 0
        # this will measure the steps of calculating the variance before update
        # we are tracking this steps in a different variable than the self.steps
        #since we will reset that at the begining of every evaluatino episode
        self.steps_til_update = 0 

        self.beta = beta
        self.num_actions = num_actions
        self.clip_gradient = 1
        self.device = device
        self.gamma = gamma
        self.train_frequency = train_frequency
        self.K = K
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.head_start = head_start
        self.steps = 0 #counts the total steps
        self.opt = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.current_head = np.random.choice(self.K, size = 1)[0]
        self.buffer = Ensemble_Buffer(buffer_size, batch_size, device, beta, K = self.K)

    def update_variance(self, variance, q):    
        '''method that keeps track of a running average
        of the variance. In some algorithms (e.g., DQN BDQN etc)
        this will be zero. However, in some other is usefull.'''

        self.variance = (self.variance * self.steps_til_update + variance)/(self.steps_til_update + 1)
        self.q_mean = (self.q_mean * self.steps_til_update + q)/(self.steps_til_update + 1)
        self.steps_til_update += 1
        return
    
    def get_variance(self, return_q = False):
        variance = self.variance
        q_mean = self.q_mean
        self.reset_variance()
        if return_q: return variance, q_mean
        return variance, 0
    
    def reset_variance(self,):
        self.variance = 0
        self.q_mean = 0
        self.steps_til_update = 0
        return

    def get_lambda_value(self,):
        return -1
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
        with torch.no_grad():
            qs = self.policy_net(x = state, k = self.current_head).detach().cpu().numpy()
        self.update_variance(variance = 0, q = 0)
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
        with torch.no_grad():
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
        if self.K  > 1:
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
                 gamma,
                 
                 #for the Bpolztamnn exploration,
                 exploration_duration,
                 initial_tau = 1_000,
                 final_tau = 0.5,
                 epsilon_greedy = True,
                 
                 #for teh MLP core
                 cnn = True,
                 state_size = -1,):
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
                            head_start = head_start,
                            gamma = gamma,
                            cnn = cnn,
                            state_size = state_size,)

        self.exploration_duration = exploration_duration
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.epsilon_greedy = epsilon_greedy
        if self.epsilon_greedy:
            self.epsilon = np.linspace(1, 0.01, self.exploration_duration)
        
        else:
            self.temperature = np.linspace(self.initial_tau, self.final_tau, self.exploration_duration)

    def update_variance(self, variance):
        return super().update_variance(variance, q = 0)
    def get_variance(self, return_q=False):
        return super().get_variance(return_q = False)

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
            self.update_variance(eps)
            if  np.random.rand() < eps:
                #return a random action
                return np.random.choice(self.num_actions, size = 1)[0]
            else:
                return self.get_clear_action(state)
        #otherwise, we have boltzamm exploration
        state = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            qs = self.policy_net(x = state, k = self.current_head).detach().cpu().numpy()
        qs = qs.reshape(-1)
        
        tau = self.temperature[self.steps if self.steps < self.temperature.shape[0] else -1]
        self.update_variance(tau)
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
        with torch.no_grad():
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
                 gamma,
                 lambda_value,
                 #for teh MLP core
                 cnn = True,
                 state_size = -1,):
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
                         head_start,
                         gamma,
                         #for teh MLP core
                        cnn = cnn,
                        state_size = state_size,)

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
        
        self.update_variance(qs_std[action].detach().cpu().numpy(), qs_mean[action].detach().cpu().numpy())

        return action



class Adeu_Std(BootUCB):
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
                 lambda_value,
                 rho,
                 initial_tau,
                 #for teh MLP core
                 cnn = True,
                 state_size = -1,):
        
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
                         head_start,
                         lambda_value,
                         #for teh MLP core
                         cnn = cnn,
                         state_size = state_size,
                         )
        self.rho = rho
        self.initial_tau = initial_tau
    def get_action(self, state):
        
        if self.steps < self.head_start:
            self.fix_lambda(state)
            return np.random.choice(self.num_actions, size = 1)[0]
        
        if self.steps == self.head_start:
            self.lambda_value = float(self.lambda_value)
            tqdm.write(f'the final lambda value for that run is {round(self.lambda_value, ndigits=3)}')

        if np.random.rand() < self.rho:
            return np.random.choice(self.num_actions, size = 1)[0]

        state = torch.from_numpy(state).to(self.device)
        qs = self.policy_net(x = state) #qs now is a list of torch.Tensors
        
        qs = torch.stack(qs, dim=0)  # shape: (10, 1, n_actions)
        qs_mean = qs.mean(dim=0)      # shape: (1, n_actions)
        qs_mean = qs_mean.squeeze(0)

        qs_std = qs.std(dim=0)
        qs_std = qs_std.squeeze(0)

        #I can chane this max. I should even consider leaving it as qs_mean/qs_std allowing to keep track of differents tds for each action, so allowing for a f(s,a)
        variance = self.lambda_value*torch.sqrt(torch.max(qs_std))
        print(' I should check what type of g() should be used')
        probabilities = (qs_mean/variance).detach().cpu().numpy()
        probabilities = np.exp(probabilities)
        probabilities/=np.sum(probabilities)
        if np.max(probabilities)  > 0.4:
            tqdm.write('exploitation')

        action = np.random.choice(self.num_actions, size=1, p = probabilities)[0]
        
        self.update_variance(variance.detach().cpu().numpy(), qs_mean[action].detach().cpu().numpy())

        return action

    def fix_lambda(self, state):
        #I will calculare the mean std in order to fine tune lambda
        state = torch.from_numpy(state).to(self.device)
        
        qs = self.policy_net(x = state)
        qs = torch.stack(qs, dim=0)  # shape: (10, 1, n_actions)
        qs_std = qs.std(dim=0)
        qs_std = qs_std.squeeze(0)

        variance = torch.sqrt(torch.max(qs_std)).detach().cpu().numpy()
        #Ideally I would like the maximum variance at the early stages of the training (i.e., head start)
        #to be around self.initial_tau
        l = self.initial_tau/variance
        self.lambda_value = (self.lambda_value*self.steps + l)/(self.steps + 1 )
        return 

    def get_lambda_value(self, ):
        return self.lambda_value
    

class EZ_ADEU(DQN):
    '''
    In this class, the adeu instance that emulates the ez-greedy algorithm has been implemented. 
    For more details on the implementation check both the ADEU and the dabney et al papers.
    '''
    
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
                 gamma,
                 N,               
                 #for the Bpolztamnn exploration,
                 exploration_duration,
                 initial_tau = 1_000,
                 final_tau = 0.5,
                 state_size = -1,
                 cnn = True,
                 ):
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
                 head_start,
                 gamma,
                 exploration_duration,
                 initial_tau = initial_tau,
                 final_tau = final_tau,
                 epsilon_greedy = True, #we will need an anealing epsilon since it will serve as our rho
                 
                 #for teh MLP core
                 cnn = False,
                 state_size = state_size,)
        
        self.zeta = zipf(a = 1.25)#a is the mu parameter defined in dabney
        self.n = 0 #let's say reverse length of the option
        self.current_options = -1 #what action is replicated when simulating the current option
        self.explored = False #stores whether agent aexplored or not in the previous step see paper for more details
        self.eps = 1/(N + 1)

    def calculate_uncertainty(self, ):

        eps = self.eps
        self.update_variance(eps)

        if self.n == 0 and not self.explored:
            return eps, 0
        if self.n == 0 and self.explored:
            n = self.zeta.rvs(size=1)
            return 1, n -1 
        
        if self.n > 0:
            return 1, self.n-1
        
        return 
    

    def get_action(self, state):
        #warmup action 
        if self.steps < self.head_start:
            return np.random.choice(self.num_actions, size = 1)[0]
        f, n = self.calculate_uncertainty()
        self.n = n
        probs = [1 - f, f]
        clear_action = self.get_clear_action(state)
        scenarios = ['ploit', 'plore']
        s = np.random.choice(scenarios, size = 1, p = probs)
        if s == 'ploit':
            self.explored = False
            return clear_action
        
        self.explored = True
        if self.n > 0:
            #use teh previously selected action
            return self.current_options
        omega = np.random.choice(self.num_actions, size = 1)
        self.current_options = omega

        return omega
    
    def reset(self,):
        self.n = 0
        self.explored = False


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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class Random_Exploration_Agent(nn.Module):
    def __init__(self,num_actions):
        super().__init__()
        self.num_actions = num_actions

    def get_action(self, state):
        return np.random.choice(self.num_actions, size = 1)[0]
    def train(self, experiences):
        return 0
    

class Two_Modes_ADEU_DQN():
    def __init__(   self, 
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
                    gamma, 
                    exploration_duration, 
                    initial_tau, 
                    final_tau, 
                    epsilon_greedy, 
                    cnn, 
                    state_size,
                    maximum_timesteps,
                    starting_mode):
        
        self.exploitation_agent = DQN(
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
                                        gamma, 
                                        exploration_duration, 
                                        initial_tau, 
                                        final_tau, 
                                        epsilon_greedy, 
                                        cnn, 
                                        state_size,
                                        
        )

        self.exploitation_agent.opt = torch.optim.AdamW(self.exploitation_agent.policy_net.parameters(), lr=self.exploitation_agent.learning_rate)
        self.exploration_agent = Random_Exploration_Agent(num_actions)
        self.exploration = 0
        self.exploitation = 0
        self.maximum_timesteps = maximum_timesteps

        self.k = 5
        self.past_values = np.zeros([maximum_timesteps,] )
        self.past_rewards = np.zeros([maximum_timesteps,])
        self.previous_d_promises = np.zeros([maximum_timesteps,])
        self.gamma = gamma
        self.homeostasis_threshold = np.zeros([maximum_timesteps,]) # we inityiallize it to 0 since we would assume that the uncertainty for the first ever episode is high and we would like the threshold to be zero in order to explore
        self.homeostasis_rho = 0.01
        self.uncertainty = 0
        self.episodes_with_no_evaluation = 1
        self.starting_mode = starting_mode
        self.device = device
        self.reset()
        
    def explore_vs_exploit(self, ):
        exploration = self.exploitation
        exploitation = self.exploitation

        self.exploitation = 0
        self.exploration = 0
        return exploration/(exploitation+ exploration + 1e-5)
    
    def reset (self, ):
        self.current_timestep = -1
        self.past_values = np.zeros([self.maximum_timesteps,] )
        self.past_rewards = np.zeros([self.maximum_timesteps,])
        self.homeostasis_threshold = self.homeostasis()
        self.episodes_with_no_evaluation+=1
    
    
    def get_clear_action(self, state, return_value = False):
        if not return_value:
            return self.exploitation_agent.get_clear_action(state)
        
        state = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            qs = self.exploitation_agent.policy_net(x = state, k = 0).detach().cpu().numpy()
        action = np.argmax(qs)
        return action, qs.flatten()[action] 

    def get_variance(self, return_q = None):
        retval = self.uncertainty/self.episodes_with_no_evaluation
        self.episodes_with_no_evaluation = 0
        return retval, 0
    
    def update_values_rewards(self, values, rewards):
        self.past_values[self.current_timestep, :] = values.detach().cpu().numpy()
        self.past_rewards[self.current_timestep, :] = rewards
        return



    def get_action(self, state):
        self.current_timestep+=1
        exploitation_action, exploitation_value = self.get_clear_action(state, return_value=True)
        uncertainty = self.calculate_uncertainty(exploitation_value, ) #uncertainty is (num_envs, 1)

        

        self.uncertainty += uncertainty
        self.previous_d_promises[self.current_timestep] = uncertainty

      


        if self.homeostasis_threshold[self.current_timestep] < 0.5:
            #uncertainty is low, exploit
            action = exploitation_action
            self.exploitation += 1
        else:
            #the action in the next couple of lines is THE SAME ONE (but one is in torch)
            action = self.exploration_agent.get_action(state)
            self.exploration +=1
            


        return action
    
    def calculate_uncertainty(self, value:np.array):

        if self.current_timestep < self.k:
            if self.starting_mode == 'exploration' :
                return 1
            return 0

        v_t_minus_k = self.past_values[self.current_timestep - self.k] if (self.current_timestep - self.k) >= 0 \
                                                                       else 0
        r_t_minus_k = self.past_rewards[self.current_timestep - self.k : self.current_timestep ] if (self.current_timestep - self.k) >= 0 \
                                                                       else 0
        
        s = (np.sum(np.array([self.gamma **i *r_t_minus_k[max(-i-1, -5)]  for i in range(r_t_minus_k.size)])))

        D_promise = v_t_minus_k - s - (self.gamma**self.k)*np.squeeze(value)
        D_promise = np.abs(D_promise)
        return D_promise
    
    def homeostasis(self, ):
        '''
        Docstring for homeostasis
        
        It returns an array with ether 1s or 0s. 
        1 means explore, otherwise exploit (similarly to epsilon greedy) 
        '''
        rho = self.homeostasis_rho
        x_values = self.previous_d_promises
        T = len(x_values)

        # Initialize variables
        x_bar = 0.0
        x2_bar = 1.0
        x_plus_bar = 1.0

        y = np.zeros([T])

        for t in range(T):
            x_t = x_values[t]
            tau = min(t + 1, 100 / rho)  # t starts at 0, so use (t+1)

            # Update moving average and variance
            x_bar = (1 - 1 / tau) * x_bar + (1 / tau) * x_t
            x2_bar = (1 - 1 / tau) * x2_bar + (1 / tau) * (x_t - x_bar) ** 2

            # Standardize and exponentiate
            x_plus = np.exp((x_t - x_bar) / np.sqrt(x2_bar + 1e-8))  # small epsilon for stability

            # Update transformed moving average
            x_plus_bar = (1 - 1 / tau) * x_plus_bar + (1 / tau) * x_plus

            # Sample from Bernoulli
            p = np.minimum(1, rho * x_plus / (x_plus_bar + 1e-8))
            y[t] = np.random.binomial(1, p)

        return y
    


    def train(self, experiences):
        self.exploitation_agent.train(experiences)
        self.exploration_agent.train(experiences)
        return 


    def step(self, state, action, reward, next_state, done):
        return self.exploitation_agent.step(state, action, reward, next_state, done)


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
