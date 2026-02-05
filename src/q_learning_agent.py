import numpy as np
class Q_Learning_Agent():
    
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        self.q_table =  np.random.rand(self.state_size, self.action_size)
        #self.q_table = np.zeros([self.state_size, self.action_size])
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
        return 
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
        self.eps = 0.2 if self.eps > 0.2 else self.eps
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
        if new_eps is None:
            return
        for agent in self.agents:
            agent.set_eps(new_eps)
        return
    
    
    def step(self, tuple):
        obs, action, reward, new_obs, done, masks  = tuple 
        for a in np.where(masks == 1)[0]:
            self.agents[a].step([obs, action, reward, new_obs, done])
        return 
            
            
    """ def get_clear_action(self, obs):
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
        return max_action """
        
    def get_clear_action(self, obs):
        actions = np.array([a.get_clear_action(obs) for a in self.agents])
        most_common_action = np.bincount(actions).argmax()
        return most_common_action
        
    def get_action(self, obs):
        self.set_eps()
        actions = np.array([a.get_action(obs) for a in self.agents])
        actions = np.unique(actions)
        
        def obj(action):
            q_values = np.array([a.q_table[obs, action] for a in self.agents])
            retval = np.mean(q_values) + self.lambda_value*np.std(q_values) if self.num_of_agents > 0 else 0    
            return retval, np.std(q_values)
        
        max_action = -1
        max_ucb = -np.inf
        max_std = -np.inf
        #actions = np.arange(self.action_size)
        for action in actions:
            temp_obj, std = obj(action)
            if temp_obj > max_ucb:
                max_action = action
                max_ucb = temp_obj
                max_std = std
        if max_action == -1:
            raise ValueError        
        return max_action, max_std
    
    
    
class ADEU_Q_Learning(Q_Learning_Agent):
    
    
    def __init__(self, state_size, action_size, beta, rollout_frequency):
        super().__init__(state_size, action_size)
        self.beta = beta
        self.n = 0.1*np.ones(state_size)
        self.eps = 1
        self.rollout = False
        self.rollout_frequency = rollout_frequency
        
    def get_action(self, obs):
        
        if self.rollout:
            p = np.random.rand()
            if p < self.eps:
                #return a random move
                return np.random.randint(0, self.action_size)
            return np.argmax(self.q_table[obs])


        #getting all the q values for the corresponding state
        qs = self.q_table[obs]
        self.exploration_rate = sigmoid( np.sqrt(self.n[obs]))
        exploration_rate =  self.exploration_rate * qs
        
        exploration_rate = np.clip(exploration_rate, 0 , np.max(exploration_rate)) #we gurantee tat the exploration rate is positive
        
        #exploration_rate = (exploration_rate - np.min(exploration_rate))/ (np.max(exploration_rate) - np.min(exploration_rate))
        exploration_rate = exploration_rate/np.sum(exploration_rate)
        probs = exploration_rate
        
        '''probs = np.exp(exploration_rate)
        probs/= np.sum(probs) #it was probs instead of exploration rate'''
        
        
        action = np.random.choice(np.arange(0, self.action_size), size = 1, p = probs)
        return action[0]
    
    
    def set_eps(self, eps):
        self.eps = min(0.8, eps)
        self.eps = max(eps, 0.2)
        return
    
    def step(self, tuple):
        obs, action, reward, new_obs, done  = tuple 
        self.n[obs] += self.beta #it was n[new_obs]. I am not sure it makes sense but I should chek it on the non-sefate environment first
        return super().step(tuple)
    
    def get_exploration_rate(self):
         expl = 1/self.exploration_rate
         expl /= 40.033
         return np.round(expl, decimals=3)     



class sADEU_safety(ADEU_Q_Learning):
    ''' This class only implementes the sADEU that calculates the probabilities based only in the 
    expected cost and the Q values. '''

    def __init__(self, state_size, action_size, beta, initial_expected_cost, delta = .01):
        rollout_frequency = 0 # it was set to zero since we do not like to perform rolluts
        super().__init__(state_size, action_size, beta, rollout_frequency)
        self.gtd = np.zeros([self.state_size, self.action_size]) + initial_expected_cost
        self.delta = delta
        self.initial_expected_cost = initial_expected_cost

    def step(self, tuple):
        
        obs, action, reward, new_obs, cost, done  = tuple 
        if cost == -1: #we are in the background knowledge procedure
            cost = 5*self.initial_expected_cost
        self.gtd[obs, action] = (self.gtd[obs, action] * self.n[obs] + cost) / (self.n[obs] + self.beta) 
        return super().step([obs, action, reward, new_obs, done])
    
    def get_action(self, obs):
        
        #getting all the q values for the corresponding state
        qs = self.q_table[obs]
        self.exploration_rate = np.exp( - self.gtd[obs]/self.delta )
        exploration_rate =  self.exploration_rate 
        
        #exploration_rate = np.clip(exploration_rate, 0 , np.max(exploration_rate)) #we gurantee tat the exploration rate is positive
        
        #exploration_rate = (exploration_rate - np.min(exploration_rate))/ (np.max(exploration_rate) - np.min(exploration_rate))
        exploration_rate = exploration_rate/np.sum(exploration_rate)
        probs = exploration_rate
        
        '''probs = np.exp(exploration_rate)
        probs/= np.sum(probs) #it was probs instead of exploration rate'''
        
        
        action = np.random.choice(np.arange(0, self.action_size), size = 1, p = probs)
        return action[0]
    

class sADEU_safety_uncertainty(sADEU_safety):
    '''class that implements the safety (depending on expected cost)
     and uncertainty agent '''
    def __init__(self, state_size, action_size, beta, initial_expected_cost, delta=0.01):
        super().__init__(state_size, action_size, beta, initial_expected_cost, delta)

    def get_action(self, obs):
        qs = self.q_table[obs]
        f_unc = sigmoid(self.beta * np.sqrt(self.n[obs]))*qs
        unc_term = f_unc / ((2 + np.max(qs)*(np.max(self.gtd[obs]) > np.max(qs)) + np.max(- self.gtd[obs]/self.delta)) **2)
        
        
        if np.max(np.exp((- self.gtd[obs]/self.delta))) > np.max(np.exp(unc_term)) and np.max(self.gtd[obs]) < 0.1:
            print('d')

        #prevent from overfloating
        x = np.clip( (- self.gtd[obs]/self.delta) + unc_term, a_min=None, a_max=709)  # max float for exp
        self.exploration_rate = np.exp(x)
        exploration_rate =  self.exploration_rate 
        
        exploration_rate = np.clip(exploration_rate, 0 , np.max(exploration_rate)) #we gurantee tat the exploration rate is positive
        
        #exploration_rate = (exploration_rate - np.min(exploration_rate))/ (np.max(exploration_rate) - np.min(exploration_rate))
        exploration_rate = exploration_rate/np.sum(exploration_rate)
        probs = exploration_rate
        
        '''probs = np.exp(exploration_rate)
        probs/= np.sum(probs) #it was probs instead of exploration rate'''
        
        
        action = np.random.choice(np.arange(0, self.action_size), size = 1, p = probs)
        return action[0]

def sigmoid(x):
    x-=5 #0.5
    return 1/(1 + np.exp(-x)) #-0.5