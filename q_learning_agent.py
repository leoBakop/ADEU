import numpy as np
class Q_Learning_Agent():
    
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        self.q_table =  np.random.rand(self.state_size, self.action_size)
        self.eps = 1
        self.learning_rate = 0.8
        self.gamma = 0.95
        
    def get_action(self, obs):
        p = np.random.rand()
        if p < self.eps:
            #return a random move
            return np.random.randint(0, self.action_size)
        return np.argmax(self.q_table[obs])
    
    def step(self, tuple):
        obs, action, reward, new_obs, done  = tuple 
        self.q_table[obs, action] += self.learning_rate * (reward + self.gamma *np.max(self.q_table[new_obs])*(1-done) - self.q_table[obs, action])

    def set_eps(self, new_eps):
        self.eps = new_eps
        
    def get_clear_action(self, obs):
        return np.argmax(self.q_table[obs])
    
    
class IR_Q_Learning_Agent(Q_Learning_Agent):
    def __init__(self, state_size, action_size, beta):
        super().__init__(state_size, action_size)
        self.beta = beta
        self.n = 0.1*np.ones(state_size)
        
    def step(self, tuple):
        obs, action, reward, new_obs, done  = tuple 
        reward += self.beta/np.sqrt(self.n[new_obs])
        tuple = [obs, action, reward, new_obs, done]
        self.n[new_obs] += 1
        return super().step(tuple)
    
    
class UCB_Q_Learning():
    def __init__(self, state_size, action_size, num_of_agents, lambda_value, p):
        self.state_size = state_size
        self.action_size = action_size
        self.num_of_agents = num_of_agents
        self.lambda_value = lambda_value
        self.p = p
        self.agents = [Q_Learning_Agent(state_size, action_size=action_size) for _ in range(num_of_agents)]
        pass
    
    def set_eps(self, new_eps = None):
        new_eps = 0.1
        for agent in self.agents:
            agent.set_eps(new_eps)
        return
    
    
    def step(self, tuple):
        obs, action, reward, new_obs, done, masks  = tuple 
        for a in np.where(masks == 1)[0]:
            self.agents[a].step([obs, action, reward, new_obs, done])
    def get_clear_action(self, obs):
        def obj(action):
            q_values = np.array([a.q_table[obs, action] for a in self.agents])
            return np.mean(q_values)
        max_action = -1
        max_ucb = -np.inf
        actions = np.array([a.get_clear_action(obs) for a in self.agents])

        for action in actions:
            temp_obj = obj(action)
            if temp_obj > max_ucb:
                max_action = action
                max_ucb = temp_obj
                
        if max_action == -1:
            raise ValueError        
        return max_action
        
        
    def get_action(self, obs):
        self.set_eps()
        actions = np.array([a.get_action(obs) for a in self.agents])
        
        def obj(action):
            q_values = np.array([a.q_table[obs, action] for a in self.agents])
            retval = np.mean(q_values) + self.lambda_value*np.std(q_values)
            return retval
        max_action = -1
        max_ucb = -np.inf
        for action in actions:
            temp_obj = obj(action)
            if temp_obj > max_ucb:
                max_action = action
                max_ucb = temp_obj
                
        if max_action == -1:
            raise ValueError        
        return max_action