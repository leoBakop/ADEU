import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBuffer, Priorized_Experience_Replay
import network
from torch.distributions import MultivariateNormal
import welford
from gym.wrappers.normalize import RunningMeanStd
from torch.nn.utils import clip_grad_norm_
from torchrl.modules import NoisyLinear




# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, noisy_nets = False):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 256) if not noisy_nets else NoisyLinear(256, 256)

        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        seed,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr = 3e-4,
        train_frequency = 1,
        batch_size = 256,
        min_mem_size = 256,
        buffer_size =1e+5,
        per = False,
        mod_rnd = False,
        adaptive_exploration = True,
        eps = 0.01,
        use_rnd = False,
        use_noisy_nets = False
    ):
        if torch.cuda.is_available(): self.device = torch.device("cuda")
        else: self.device = torch.device("cpu")
        device = self.device
        print(f'the TD3 agent will run in the {self.device}')
        self.actor = Actor(state_dim, action_dim, max_action, noisy_nets=use_noisy_nets).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.batch_size = batch_size
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.min_mem_size = min_mem_size
        self.train_frequency = train_frequency
        self.per = per

        if not self.per:
            self.buffer = ReplayBuffer(buffer_size=int(buffer_size), batch_size=int(batch_size), device = self.device, store_intrisic=use_rnd)
        else:
             self.buffer = Priorized_Experience_Replay(buffer_size=int(buffer_size), batch_size=int(batch_size),device=self.device,offset=1e-6)

        self.total_it = 0
        #Algorithm used for this run
        self.adaptive_exploration = adaptive_exploration
        self.use_rnd = use_rnd
        self.use_noisy_nets = use_noisy_nets
        self.variance = self.policy_noise
        
        self.mod_rnd = mod_rnd
        if self.mod_rnd:
            self.rnd_low_threshold = 0.2
            self.rnd_high_threshold = 40
            #self.rnd_high_threshold should be in the range of [5,40]
            print(f'the rnd mean threshold is {self.rnd_high_threshold}')
            self.uncertainty_const = 1.5
            self.iter_mod_rnd = 0
            self.max_rnd_history = 20
            self.buff_rnds = np.zeros(self.max_rnd_history)
            self.full = False
            self.buff_mean_rnds = np.zeros(self.max_rnd_history*100000) #it is just a huge nymebr i don't care that much
            self.index_buff_mean_rnds = 0
        
        ####################################################
        if self.adaptive_exploration or self.use_rnd:
            hidden_layer = 256
            self.variance = 0
            self.rnd_target = network.RND_Network(device=self.device, input_size=state_dim, hidden_size=hidden_layer, max_speed=max_action).to(self.device)
            torch.manual_seed(10 * (seed+1))
            self.rnd_predictor = network.RND_NetworkPred(device=self.device, input_size=state_dim, hidden_size=hidden_layer, max_speed=max_action).to(self.device)
            torch.manual_seed(seed)
            
            self.rnd_optimizer = torch.optim.SGD(self.rnd_predictor.parameters(), lr=lr)
            self.rnd_loss = nn.MSELoss()
            self.eps = eps
            self.rollout_episode = False
            self.obss = RunningMeanStd(shape= (1, state_dim)) #it was inside the following if
            if self.use_rnd:

                self.intr_rewards = RunningMeanStd()                
                self.discount = discount
                self.discount = 0.999
                self.intr_gamma = 0.99
                self.w_index = 0

        

    def get_clear_action(self, state):  
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def get_variance(self):
        return self.variance

    def calculate_intrisic_reward(self, state):

        self.rnd_predictor.eval()
        self.rnd_target.eval()
        with torch.no_grad():
            variance = np.absolute(self.rnd_predictor(state).detach().cpu() - self.rnd_target(state).detach().cpu())
        self.rnd_predictor.train()
        self.rnd_target.train()
        variance = np.clip(variance, a_min = - 2*self.max_action, a_max = 2*self.max_action) #just to avoid the infinite variance
        if torch.isnan(variance).any():
            print('debug')
            variance = self.max_action
        return variance**2


    def get_action(self, state):
        """ method which returns the action suitable for exploration """
        """ action = (
				self.get_clear_action(np.array(state))
				+ np.random.normal(0, self.max_action * self.policy_noise, size=self.action_dim)
			).clip(-self.max_action, self.max_action) """
        clear_action = self.get_clear_action(np.array(state))
        if self.use_rnd: return clear_action
        if not self.adaptive_exploration or (self.adaptive_exploration and self.rollout_episode): 
            return (clear_action + np.random.normal(0, self.max_action * self.policy_noise, size=self.action_dim)).clip(-self.max_action, self.max_action)

        s = (state - torch.from_numpy(self.obss.mean) )/ torch.from_numpy(self.obss.var**2)
        s = s[0].float()
        s = torch.clamp(s, min = -5, max = 5)
        #s = self.scale(s) #new code, normilize the value between -1 1 iff the values are too low
        variance = self.calculate_intrisic_reward(s) #it was just state
        #variance = .1 in case of constant variance
        self.variance = variance #self.variance servs as a buffer
        #if np.random.rand()< self.eps: return MultivariateNormal(torch.from_numpy(clear_action), variance * torch.eye(clear_action.shape[0],clear_action.shape[0])).sample().detach().cpu().numpy() 
        
        #variance = max(variance, self.eps)
        if variance < 1e-2: return clear_action
        
        dist = MultivariateNormal(torch.from_numpy(clear_action), variance * torch.eye(clear_action.shape[0],clear_action.shape[0]))
        action = dist.sample()
        return action.detach().cpu().numpy()
    

    def step(self, state, action, reward, next_state, done, timestep):
        l = 0
        loss = 0
        if self.adaptive_exploration or self.use_rnd: self.obss.update(state.numpy())
        

        if self.use_rnd:
            #in case of the rnd, instead of the running average and runing std we are going to use
            #the 'sample average' and 'sample std' to increase the computational efficiency
            #instead of witing code we are going to use the welford algorithm via its package
            self.obss.update(state.numpy())
            #s = state #that was the correct now I am testing following line
            s = next_state
            s = (state - torch.from_numpy(self.obss.mean) )/ torch.from_numpy(self.obss.var**2)
            s = s[0].float()
            s = torch.clamp(s, min = -5, max = 5)
            
            #s = self.scale(s) #scale between -1 1 if the values are too low

            intrisic_reward = self.calculate_intrisic_reward(s)
            #self.welford.add(np.array([self.w_index, intrisic_reward.numpy()[0]]))
            self.variance = intrisic_reward**2 #self.variance servs as a buffer
            self.intr_rewards.update(self.variance.numpy())
            self.variance /= self.intr_rewards.mean
            self.w_index+=1
            #if self.w_index > 10: intrisic_reward/=self.welford.var_s[1]
            self.buffer.add(state, action, reward, next_state, done, self.variance)
        else:
            self.buffer.add(state, action, reward, next_state, done)
        
        
        # If enough samples are available in buffer, get random subset and learn
        if len(self.buffer) > self.min_mem_size and ((timestep % self.train_frequency) == 0):
            experiences = self.buffer.sample()
    
            if self.use_rnd:
                loss= self.train_using_rnd( experiences)
            elif self.per:
                self.train_basic_per(experiences)
            else:
                loss = self.train_basic(experiences)    
            
            if self.adaptive_exploration or self.use_rnd: l = self.train_rnd_network(experiences)
        return loss, l        

    
    
    def train_basic(self, experiences):
        self.total_it += 1

        # Sample replay buffer 
        state, action, reward, next_state, done = experiences
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action.float()) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss

    def evaluate_rnd_netowrk(self, mean_var, max_var, ):
        if not self.mod_rnd: 
            print('Something wrong with the mod rnd')
            return
        if self.rnd_predictor.get_uncertainty() != self.rnd_target.get_uncertainty():
            print('Something wrong with the synchronization of the rnd')
            return
        self.buff_rnds[self.iter_mod_rnd] = max_var
        self.iter_mod_rnd+=1
        self.iter_mod_rnd%=self.max_rnd_history

        if not self.full and (self.iter_mod_rnd + 1 == self.max_rnd_history): 
            self.full = True
        
        self.buff_mean_rnds[self.index_buff_mean_rnds]=mean_var 
        self.index_buff_mean_rnds+=1 
        
        if not self.full: return
        unc = self.rnd_predictor.get_uncertainty()

        
        mean_max_var = np.mean(self.buff_rnds)
        #it was max_var < self.rnd_low_threshold:
        if mean_max_var < self.rnd_low_threshold:
             unc = (unc * self.uncertainty_const)
             self.rnd_predictor.set_uncertainty(unc)
             self.rnd_target.set_uncertainty(unc)
             return
         
        #I should fucking find a way to trak how mach the mean var changes during the episode.
        #if it remains the same then I should decrease rge threshold.
        #in all the experiments the rng_high_threshold was set to 40 and in invertedpendulum to 10. 
        if self.index_buff_mean_rnds > 60:
            last = self.buff_mean_rnds[self.index_buff_mean_rnds-(self.index_buff_mean_rnds // 10):self.index_buff_mean_rnds]  # Last n/10 elements
            first = self.buff_mean_rnds[:self.index_buff_mean_rnds-(self.index_buff_mean_rnds // 10)]  # First 9n/10 elements


            #if mean_var > self.rnd_high_threshold :
            if np.abs(np.mean(last) - np.mean(first)) < 0.05*np.mean(last):
                unc = (unc/self.uncertainty_const)
                self.rnd_predictor.set_uncertainty(unc)
                self.rnd_target.set_uncertainty(unc)
        
        return



    def train_basic_per(self, experiences):
        self.total_it += 1

        # Sample replay buffer 
        state, action, reward, next_state, done, idx, weights = experiences
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action.float()) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        td_error = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        weights = torch.from_numpy(weights).to(self.device)
        critic_loss = (td_error*weights).mean().to(self.device)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        prob_error = current_Q1+current_Q2 - 2*target_Q
        self.buffer.update_priorities(idx, abs(prob_error.data.cpu().numpy()))
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss



    def train_using_rnd(self, experiences):
        self.total_it += 1

        # Sample replay buffer 
        state, action, reward, next_state, done, intrisic_reward = experiences
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action.float()) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            extr_Q = reward + (1 - done) * self.discount * target_Q
            intr_Q = intrisic_reward + self.intr_gamma * target_Q
            target_Q = extr_Q+intr_Q
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss



    def train_rnd_network(self, experiences):
        # Set the networks to the appropriate mode
        self.rnd_predictor.train()
        self.rnd_target.eval()

        # Extract states from experiences
        states, *_ = experiences
        if self.use_rnd or self.adaptive_exploration: #it was only if self.use_rnd
            states = ((states.cpu() - self.obss.mean) / self.obss.var**2)[0].float()
            states= torch.clamp(states, -5, 5)
            #new code, if the state containts low values, I scale them between -1 1
            
            #states = self.scale(state=states)

        # Forward pass through the target network
        with torch.no_grad():
            target_outputs = self.rnd_target(states)

        # Forward pass through the predictor network
        predictor_outputs = self.rnd_predictor(states)

        # Compute the loss between predictor and target outputs
        l = self.rnd_loss(predictor_outputs, target_outputs)

        # Backward pass and optimization step
        self.rnd_optimizer.zero_grad()
        l.backward()
        clip_grad_norm_(self.rnd_predictor.parameters(), 1.0)  # Gradient clipping
        self.rnd_optimizer.step()
        return l

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


    def scale(self, state, max_value = 5, min_value = - 5):
        if torch.max(state) > 1: return state
        t_min = torch.min(state)
        t_max = torch.max(state)

        # Apply min-max scaling to range [0, 1]
        normalized_state = (state - t_min) / (t_max - t_min)

        # Scale to range [min_value, max_value]
        scaled_state = (max_value - min_value) * normalized_state + min_value

        return scaled_state
    
    
    
    
class UCB_TD3_Agent:
    def __init__(self, N, lambda_parameter, agent_params, beta= 0.99):
        self.N = N
        self.agents  = [TD3(**agent_params) for _ in range(self.N)]
        self.lambda_value = lambda_parameter
        self.beta = beta
        if torch.cuda.is_available(): self.device = torch.device("cuda")
        else: self.device = torch.device("cpu")
        self.variance = 0
        return
    
    def get_variance(self):
        return self.variance
    
    def step(self, state, action, reward, next_state, done, timestep):
        loss = []
        mask = np.random.binomial(1, self.beta, self.N)
        for i, agent in enumerate(self.agents):
            #add experience in the buffer of each agent
            if mask[i] == 1:
                agent.buffer.add(state, action, reward, next_state, done)
            #train every agent in some experienve derived by its buffer
            if len(agent.buffer) > agent.min_mem_size and ((timestep % agent.train_frequency) == 0):
                experiences = agent.buffer.sample()
                l = agent.train_basic(experiences).detach().cpu().numpy()
                loss.append(l)
        
        return np.mean(np.array(loss)), 0
    
    def get_action(self, state):
        modified = False
        action_dim = self.agents[0].action_dim
        actions = torch.Tensor([a.get_action(state) for a in self.agents])
        state = state.to(self.device).unsqueeze(0)        
        def obj(action):
            action = torch.from_numpy(np.array([action], dtype=np.float32)).to(self.device)
            with torch.no_grad():
                q_values = np.array([np.min(a.critic(state, action)[0].detach().cpu().numpy()) for a in self.agents])
            retval = np.mean(q_values) + self.lambda_value*np.std(q_values) if self.N > 0 else 0    
            return retval, np.std(q_values)
        
        
        max_action = np.zeros(action_dim) - 153
        max_ucb = -np.inf
        max_std = -np.inf
        #actions = np.arange(self.action_size)
        for action in actions:
            temp_obj, std = obj(action)
            if temp_obj > max_ucb:
                modified = True
                max_action = action
                max_ucb = temp_obj
                max_std = std
        if not modified:
            raise ValueError      
        
        self.variance = max_std  
        return max_action.numpy()
    
    def get_clear_action(self, obs):
        modified = False
        action_dim = self.agents[0].action_dim
        actions = [a.get_action(obs) for a in self.agents]
        obs = obs.to(self.device).unsqueeze(0)
        def obj(action):
            action = torch.from_numpy(np.array([action], dtype=np.float32)).to(self.device)
            with torch.no_grad():
                q_values = np.array([np.min(a.critic(obs, action)[0].detach().cpu().numpy()) for a in self.agents])
            retval = np.mean(q_values)  
            return retval
        
        
        max_action = np.zeros(action_dim) - 153
        max_ucb = -np.inf
        max_std = -np.inf
        #actions = np.arange(self.action_size)
        for action in actions:
            temp_obj = obj(action)
            if temp_obj > max_ucb:
                modified = True
                max_action = action
                max_ucb = temp_obj
        if not modified:
            raise ValueError        
        return max_action