import network
import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from torch.nn import SmoothL1Loss
from torch.nn.utils import clip_grad_norm_
import time
from replay_buffer import ReplayBuffer, Priorized_Experience_Replay
import utils
from torch.distributions import MultivariateNormal
from sklearn.metrics import root_mean_squared_error


class Agent:
    def __init__(self, inputs_shape, hidden_layer_1,hidden_layer_2,
                    use_priorized_exp_replay, max_speed,
                    batch_size = 128, min_mem_size = 98, 
                    learning_rate = 0.001, gamma = 0.99,
                    update_target_every = 5, clip_grad = 1.0,
                    c = 1.0,action_size = 2,
                    memory_size = 100_000,
                    seed = 0,
                    cut_training = 150,
                    end_training = 550,
                    evaluation_c = 10 ** (-5),
                    train_frequency = 1,
                    #variables for the priorzed experience replay
                    tau = .005,
                    alpha = .7,
                    beta = 0.5,
                    #network variables for storing, and loading an existing network
                    path = None,
                    loaded_mem = None,
                    probabilities = None, 
                    index = 0,
                    #for the adaptive expploration
                    adaptive_eploration = True
                 ):
        
        #networks
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"{self.device} is used")
        self.action_size = action_size

        
        self.model = network.NAF(state_size = inputs_shape, action_size=action_size, layer_size_1 = hidden_layer_1,
                                    layer_size_2= hidden_layer_2, c=c,  max_speed = max_speed, seed = seed, device = self.device).to(self.device)
        
        #this predicts in every step and change its weights periodically
        self.target_model = network.NAF(state_size = inputs_shape, action_size=action_size, layer_size_1 = hidden_layer_1,
                                    layer_size_2= hidden_layer_2, c=c,  max_speed = max_speed, seed = seed, device = self.device).to(self.device)
        
        if path is not None:
            self.model.load_state_dict(torch.load(path))
            self.target_model.load_state_dict(torch.load(path))

        #miscellaneous for networks
        self.c = c
        self.evaluation_c = evaluation_c
        self.lr = learning_rate
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.loss = MSELoss()
        self.max_speed = max_speed
        #self.loss = SmoothL1Loss()
        self.gamma = gamma #gamma used in bellman equation
        self.update_target_every = update_target_every
        self.tau = tau 
        self.clip_grad = clip_grad
        
        #miscellaneous for the functionallity of the agent
        self.use_priorized_exp_replay = use_priorized_exp_replay
        self.cut_training = cut_training
        self.end_training = end_training
        self.exploration_terminated = False
        #memory
        
        self.batch_size = batch_size
        self.min_mem_size = min_mem_size if min_mem_size > batch_size else batch_size
        self.memory_size = memory_size
        self.train_frequency = train_frequency

        if self.use_priorized_exp_replay:
            
            self.replay_memory = Priorized_Experience_Replay(
                                            self.memory_size, 
                                            self.batch_size, 
                                            self.device,
                                            seed = seed,
                                            offset=1e-6,
                                            probabilities = probabilities,
                                            index = index 
                                            )
            self.alpha = alpha 
            self.beta = beta

        else:
            self.replay_memory = ReplayBuffer(
                                            self.memory_size, 
                                            self.batch_size, 
                                            self.device,
                                            seed = seed
                                            )
        if (loaded_mem is not None):
            self.replay_memory.memory = loaded_mem
        
        self.adaptive_exploration = adaptive_eploration
        if self.adaptive_exploration:
            
            self.rnd_target = network.RND_Network(device=self.device, input_size=inputs_shape, hidden_size=hidden_layer_1, max_speed=max_speed).to(self.device)
            self.rnd_predictor = network.RND_Network(device=self.device, input_size=inputs_shape, hidden_size=hidden_layer_1, max_speed=max_speed).to(self.device)
            self.rnd_optimizer = SGD(self.rnd_predictor.parameters(), lr=learning_rate * 10**(-2))
            self.rnd_loss = MSELoss()
        return


    
    def get_prediction(self, state):
        self.model.eval()
        with torch.no_grad():
            ret_val = self.model(state.unsqueeze(0))
        self.model.train()
        return ret_val

    def get_action(self, state):
        """ 
        returns always
        (the action that should be used/train the network with)
        """
        if not self.adaptive_exploration:return self.get_prediction(state)[0]
        action = self.get_prediction(state)[3].detach().cpu()

        self.rnd_predictor.eval()
        self.rnd_target.eval()
        with torch.no_grad():
            variance = np.absolute(self.rnd_predictor(state).detach().cpu() - self.rnd_target(state).detach().cpu())
        self.rnd_predictor.train()
        self.rnd_target.train()

        variance = np.clip(variance, a_min = - self.max_speed, a_max = self.max_speed)
        self.variance = variance #self.variance servs as a buffer
        if variance == 0: return action
        dist = MultivariateNormal(action.squeeze(-1), variance * torch.eye(action.squeeze(-1).shape[1],action.squeeze(-1).shape[1])) 
        expl_action = dist.sample()
        return expl_action
    

    def get_variance(self):
        return self.variance

    def get_clear_action(self,state):
        self.model.set_c(self.evaluation_c)
        return self.get_prediction(state)[0]
    
    
    def step(self, state, action, reward, next_state, done, episode, timestep):
        
        self.replay_memory.add(state, action, reward, next_state, done)
        
        if(episode == self.cut_training and not self.exploration_terminated): 
            
            print("--------------------10 episodes to testing--------------------")
            self.exploration_terminated = True
            self.optimizer.param_groups[0]['lr'] *=.1 
            self.terminate_training(episode)

        if (episode == self.end_training):   self.terminate_training(episode) 
       
        # If enough samples are available in replay_memory, get random subset and learn
        if len(self.replay_memory) > self.min_mem_size and timestep % self.train_frequency == 0:
            Q_losses = []
            for _ in range(self.update_target_every):
                experiences = self.replay_memory.sample()
                if self.use_priorized_exp_replay:
                    loss, self.model, self.target_model = self.train_per(  experiences, 
                                                        network = self.model,
                                                        target_network = self.target_model,
                                                        optim = self.optimizer)
                else:
                    loss, self.model, self.target_model = self.train_basic(     experiences, 
                                                                                network = self.model,
                                                                                target_network = self.target_model,
                                                                                optim = self.optimizer)
                Q_losses.append(loss)

            self.train_rnd(experiences)
            return np.mean(Q_losses)
        return 0.0

    def train_basic(self, experiences,network,target_network, optim, normal_training=True ):
        
        
        network.train()
        states, actions, rewards, next_states, dones = experiences

        #taking data from the network
        with torch.no_grad():
            future_vs_states = target_network(next_states)[2] # the V(s')

        
        current_qs_states = network(states,action=actions)[1] #the Q(s)
        if normal_training:
            rewards = torch.reshape(rewards, (self.batch_size, 1)) 
            new_qs = rewards + self.gamma * future_vs_states * (torch.reshape(1-dones, (self.batch_size, 1))) 
        else:
            rewards = torch.reshape(rewards, (self.R_f.batch_size, 1)) 
            new_qs = rewards + self.gamma * future_vs_states * (torch.reshape(1-dones, (self.R_f.batch_size, 1))) 
        #perform the optimization
        
        loss = self.loss(new_qs, current_qs_states)
        optim.zero_grad()
        loss.backward()
        clip_grad_norm_(network.parameters(), self.clip_grad) #check it
        optim.step()


        #update the target/(not crazy) model
        target_network=self.update_network(network, target_network)
        return float(loss.detach().cpu().numpy()), network, target_network

    def train_per(self,experiences, network, target_network, optim, normal_training = True):
        
        self.replay_memory.set_alpha(self.alpha)
        
        network.train()
        optim.zero_grad()
        states, actions, rewards, next_states, dones, idx, weights = experiences

        
        # get the Value for the next state from target model
        with torch.no_grad():
            _, _, future_vs_states, _ = target_network(next_states)
        
        rewards = torch.reshape(rewards, (self.batch_size, 1)) 
        new_qs = rewards + self.gamma * future_vs_states * (torch.reshape(1-dones, (self.batch_size, 1))) 
        
        # Get expected Q values from local model
        _, Q, _, _ = network(states, actions)

        # Compute loss
        td_error = Q - new_qs                
        #w = torch.Tensor((weights ** self.beta)/np.max(weights ** self.beta)).to(self.device)
        weights = torch.from_numpy(weights).to(self.device)
        loss = (self.loss(Q, new_qs)*weights).mean().to(self.device)
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(network.parameters(), self.clip_grad)
        optim.step()

        # ------------------- update target network ------------------- #
        target_network=self.update_network(network, target_network)
        # update per priorities
        self.replay_memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))

        
        return loss.detach().cpu().numpy(), network, target_network  

    def set_optimizers(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr

    def update_network(self, network, target_network):
        target_network.set_c(self.c)
        for target_param, model_param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(self.tau*model_param.data + (1.0-self.tau)*target_param.data)
        
        return target_network
    

    def check_if_training_was_started(self):
        return len(self.replay_memory) == self.min_mem_size
    
    def terminate_training(self, episode):
        self.set_c(self.evaluation_c, episode)
        self.model.set_testing_period(True)
        self.target_model.set_testing_period(True) #maybe there is no need but better safe than sorry
        self.pse = False #no need for extra layer noise in the evaluation phase
        self.update_target_every = 1 #this was 0 in the other experiments
    
    def set_c(self, c, episode):
        if episode >= self.end_training: 
            self.c = self.evaluation_c #for clear action in testing mode this should be 0.0
        else : self.c = c

    def set_beta(self, beta):
        self.beta = beta
    
    def train_rnd(self, experiences):
        # Set the networks to the appropriate mode
        self.rnd_predictor.train()
        self.rnd_target.eval()

        # Extract states from experiences
        states, *_ = experiences

        
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
        self.rnd_optimizer.step()


