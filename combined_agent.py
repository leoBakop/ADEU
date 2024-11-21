from agent import Agent
import sys
from td3_agent import TD3
import numpy as np
class Pretrained_Agent(Agent):

    """ This agent implemenets the `exploiting a known behaviour' intuition. In more detail,
     this agent will be trained for 50% (for the first $decay/2$ timesteps) of the horizon using the Boltzmann strategy and then it 
     will utilize the ade  """
    def __init__(self, inputs_shape, hidden_layer_1,hidden_layer_2,
                    use_priorized_exp_replay, max_speed,
                    batch_size = 128, min_mem_size = 98, 
                    learning_rate = 0.001, gamma = 0.99,
                    update_target_every = 5, clip_grad = 1.0,
                    c = 1.0,action_size = 2,
                    memory_size = 100_000,
                    seed = 0,
                    evaluation_c = 10 ** (-5),
                    train_frequency = 1,
                    #variables for the priorzed experience replay
                    tau = .005,
                    alpha = .7,
                    beta = 0.5,
                    #for the adaptive expploration
                    adaptive_exploration = True,
                    
                    mod_rnd = True,
                    use_rnd = False,
                    eps = 0.01,
                    use_noisy_nets = False,
                    timesteps = 1000,
                    horizon = 1000,
                    pretraining = 0.4
                 ):
        if (not adaptive_exploration):sys.exit(-1)
        
        super().__init__(   inputs_shape = inputs_shape,
                            hidden_layer_1= hidden_layer_1,
                            hidden_layer_2=hidden_layer_2,
                            max_speed=max_speed,
                            use_priorized_exp_replay=use_priorized_exp_replay,
                            batch_size = batch_size, 
                            min_mem_size = min_mem_size, 
                            learning_rate = learning_rate, 
                            gamma = gamma,
                            update_target_every = update_target_every ,
                            clip_grad = clip_grad ,
                            c = c ,
                            action_size = action_size ,
                            memory_size = memory_size,
                            seed = seed ,  
                            use_noisy_nets=False,
                            evaluation_c = evaluation_c,
                            train_frequency=train_frequency,
                            #variables for the priorzed experience replay
                            tau=tau,
                            alpha=alpha,
                            beta=beta,
                            ###

                            #for the adaptive expploration
                            adaptive_exploration = True,
                            eps = eps,
                            use_rnd = False,
                            mod_rnd=mod_rnd)
        
        if use_noisy_nets:
            self.use_noisy_nets = True
            self.noisy_agent = Agent(inputs_shape = inputs_shape,
                                hidden_layer_1= hidden_layer_1,
                                hidden_layer_2=hidden_layer_2,
                                max_speed=max_speed,
                                use_priorized_exp_replay=use_priorized_exp_replay,
                                batch_size = batch_size, 
                                min_mem_size = min_mem_size, 
                                learning_rate = learning_rate, 
                                gamma = gamma,
                                update_target_every = update_target_every ,
                                clip_grad = clip_grad ,
                                c = c ,
                                action_size = action_size ,
                                memory_size = memory_size,
                                seed = seed ,  
                                use_noisy_nets=True,
                                evaluation_c = evaluation_c,
                                train_frequency=train_frequency,
                                #variables for the priorzed experience replay
                                tau=tau,
                                alpha=alpha,
                                beta=beta,
                                ###
                            
                                #for the adaptive expploration
                                adaptive_exploration = False,
                                use_rnd = False,
                                mod_rnd=False)
        self.boltzman = True
        self.varience = 0
        self.horizon = horizon
        self.timesteps = timesteps
        self.index = 0
        self.pretraining = pretraining


    def get_action(self, state):
        #if we still use the 'boltzamn' action
        if self.index < int(self.pretraining*self.horizon*self.timesteps):
            self.index += 1
            if self.use_noisy_nets:
                return self.noisy_agent.get_action(state) #Boltzman action. 
            return super().get_prediction(state)[0]
        if(self.boltzman):
            print('Ade was just utilized')
            self.boltzman = False
        return super().get_action(state)

    def get_variance(self):
        if (self.boltzman):
            if self.use_noisy_nets:
                return  self.noisy_agent.get_variance()
            
            v = self.model.get_noise_cov()
            v = v.detach().cpu().numpy().reshape(v.shape[-1], v.shape[-1])
            v = np.array([v[i,i]for i in range(v.shape[0])])
            v = np.mean(v)
            return v,v

        return  super().get_variance()

    def step(self, state, action, reward, next_state, done, episode, timestep):
        if self.boltzman:
            retval = super().step(state, action, reward, next_state, done, episode, timestep)
            if self.use_noisy_nets: retval = self.noisy_agent.step(state, action, reward, next_state, done, episode, timestep)
        else:
            retval = super().step(state, action, reward, next_state, done, episode, timestep)
        return retval
    
    
class Pretrained_TD3_Agent(TD3):
    def __init__(self, 
                state_dim, 
                action_dim, 
                max_action, 
                seed, 
                horizon,
                timesteps,
                pretraining,
                discount=0.99, 
                tau=0.005, 
                policy_noise=0.2, 
                noise_clip=0.5, 
                policy_freq=2, 
                lr=0.0003, 
                train_frequency=1, 
                batch_size=256, 
                min_mem_size=256, 
                buffer_size=100000, 
                mod_rnd = False,
                per = False,
                adaptive_exploration=True, 
                eps = 0.01,
                use_rnd=False, 
                use_noisy_nets=False):
        if (not adaptive_exploration):sys.exit(-1)
        super().__init__(   state_dim=state_dim, 
                            action_dim=action_dim, 
                            max_action=max_action, 
                            seed=seed, 
                            discount=discount, 
                            tau=tau, 
                            policy_noise=policy_noise, 
                            noise_clip=noise_clip, 
                            policy_freq=policy_freq, 
                            lr=lr, 
                            mod_rnd=mod_rnd,
                            per=per,
                            train_frequency=train_frequency, 
                            batch_size=batch_size, 
                            min_mem_size=min_mem_size, 
                            buffer_size=buffer_size, 
                            adaptive_exploration=adaptive_exploration, 
                            eps=eps, 
                            use_rnd=use_rnd, 
                            use_noisy_nets=use_noisy_nets)
        self.baseline_exploration = True
        self.varience = 0
        self.horizon = horizon
        self.timesteps = timesteps
        self.index = 0
        self.pretraining = pretraining

    def get_action(self, state):
        if self.index < int(self.pretraining*self.horizon*self.timesteps):
            self.index += 1
            clear_action = self.get_clear_action(np.array(state))
            return (clear_action + np.random.normal(0, self.max_action * self.policy_noise, size=self.action_dim)).clip(-self.max_action, self.max_action) 
        if(self.baseline_exploration):
            print('Ade was just utilized')
            self.baseline_exploration = False
        return super().get_action(state)