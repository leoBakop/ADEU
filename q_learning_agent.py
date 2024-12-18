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