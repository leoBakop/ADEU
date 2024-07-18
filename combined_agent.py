from agent import Agent
import sys

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
                    adaptive_exploration = True,
                    decay = 1_000_000,
                    use_rnd = False,
                    use_noisy_nets = False,
                    timesteps = 1000,
                    horizon = 1000,
                    pretraining = 0.4
                 ):
        if (not adaptive_exploration):sys.exit(-1)
        
        super().__init__(inputs_shape, hidden_layer_1,hidden_layer_2,
                    use_priorized_exp_replay, max_speed,
                    batch_size, min_mem_size, 
                    learning_rate , gamma ,
                    update_target_every , clip_grad ,
                    c ,action_size ,
                    memory_size,
                    seed,  
                    cut_training = cut_training,
                    end_training = end_training,
                    evaluation_c = evaluation_c,
                    train_frequency=train_frequency,
                    #variables for the priorzed experience replay
                    tau=tau,
                    alpha=alpha,
                    beta=beta,
                    ###
                  
                    #for the adaptive expploration
                    adaptive_exploration = True,
                    decay = decay,
                    use_rnd = False,
                    use_noisy_nets = False)
        self.boltzman = True
        self.varience = 0
        self.horizon = horizon
        self.timesteps = timesteps
        self.index = index
        self.pretraining = pretraining


    def get_action(self, state):
        if self.index < int(self.pretraining*self.horizon*self.timesteps):
            self.index += 1
            return self.get_prediction(state)[0] #Boltzman action. 
        if(self.boltzman):
            print('Ade was just utilized')
            self.boltzman = False
        return super().get_action(state)