import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.stats import entropy
from gymnasium.envs.registration import register


register(
    id="DeepSea",
    entry_point="deepsea:DeepSea"
)

class DeepSea(gym.Env):

    def __init__(self, N):
        super(DeepSea, self).__init__()
        
        if N < 3:
            print(f"N={N} is pointless, defaulting to N=3")
            N = 3
        self.observation_space = spaces.MultiBinary(N ** 2)
        self.action_space = spaces.Discrete(2)
        # create a NxN grid
        self.grid = np.zeros((N, N), dtype=int)
        # initialize starting position
        self.grid[0,0] = 1
        self.x = 0
        self.y = 0
        self.N = N
        self.needs_reset = True
        # set a cell visitation counter
        self.visit_counter = np.zeros((N, N), dtype=int)
        self.visit_counter[0,0] += 1
        # to check convergence based on regret
        self.optimal_reward = 0.99
        self.cumulative_regret = 0
        self.episode_regret = 0
        self.total_episodes = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.converged_after = -1
        self.consecutive_stuck_episodes = 0  # Track consecutive episodes in stuck range
    

    def get_onehot_state(self):
        return self.grid.ravel().copy().astype(np.float32) 
    # returns the percentage of state coverage, just for visualization purposes
    def get_coverage(self):
        grid = self.visit_counter
        lower_triangular_mask = np.tril(np.ones_like(grid, dtype=bool))
        lower_triangular_values = grid[lower_triangular_mask]
        total_visits = lower_triangular_values.sum()
        if total_visits > 0:
            normalized_values = lower_triangular_values / total_visits
        else:
            normalized_values = lower_triangular_values
        
        total_cells = lower_triangular_values.size
        visited_cells = np.count_nonzero(lower_triangular_values)
        coverage = visited_cells / total_cells
        return coverage

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            # initialize action meaning for each possible state
            self.action_map = {}
            for i in range(self.N):
                for j in range(self.N):
                    # we map a binary vector to each possible cell
                    # the first element corresponds to left 
                    # and the second element to right, by convention
                    # self.action_map[(i,j)] = [0,1]
                    self.action_map[(i,j)] = [np.random.randint(0, 2)]
                    if self.action_map[(i,j)][0] == 0:
                        self.action_map[(i,j)].append(1)
                    else:
                        self.action_map[(i,j)].append(0)
        
        self.grid = np.zeros((self.N, self.N), dtype=int)
        self.grid[0,0] = 1
        self.x = 0
        self.y = 0
        self.needs_reset = False

        avg_regret = self.average_regret()
        if avg_regret < 0.9 and self.total_episodes > 0:
            self.converged_after = self.total_episodes
            return self.get_onehot_state(), {'solved':self.total_episodes}
        
        # Check if agent is stuck in suboptimal policy (always going left)
        if 0.99 < avg_regret < 0.9919:
            self.consecutive_stuck_episodes += 1
            if self.consecutive_stuck_episodes >= 2000:
                print(f"Agent stuck in suboptimal policy for 1000 consecutive episodes (avg_regret: {avg_regret:.4f})")
                return self.get_onehot_state(), {'solved':100000}  # -1 indicates stuck/failed
        else:
            self.consecutive_stuck_episodes = 0  # Reset counter if not in stuck range

        return self.get_onehot_state(), {}

    def step(self, action):
        assert not self.needs_reset
        # initialize vars to return
        reward = .0; terminated = False; truncated = False
        # action is either 0 or 1
        # get the action meanings for the current state
        state = (self.x, self.y)
        if self.x == self.N - 1: # if we are currently in the bottom row, this is the final step
            terminated = True
            truncated = True
        left = self.action_map[state][0]
        right = self.action_map[state][1]

        # step logic
        if self.x < self.N - 1: # if we are not on the last row
            self.x += 1 # go to the next row, no matter the action
        if action == left:
            if self.y > 0:
                self.y -= 1
        elif action == right:
            if self.y < self.N - 1:
                self.y += 1

            reward -= 0.01 / self.N
            if self.x == self.y and self.x == self.N - 1 and terminated and state[0] == self.N-1 and state[1] == self.N-1:
                reward += 1
        
        # update the grid
        if not terminated:
            self.grid[state[0],state[1]] = 0
            self.grid[self.x,self.y] = 1
            # update the visit counter
            self.visit_counter[self.x,self.y] += 1
        else:
            self.grid[state[0],state[1]] = 0
            self.needs_reset = True
        self.episode_reward += reward
        self.episode_length += 1
        info = {}
        if terminated:
            self.episode_regret = self.optimal_reward - self.episode_reward
            self.cumulative_regret += self.episode_regret
            self.total_episodes += 1
            info = {'episode': {'r': self.episode_reward, 'l': self.episode_length}}
            self.episode_reward = 0
            self.episode_length = 0
            
        return self.get_onehot_state(), reward, terminated, truncated, info
    
    # stores or returns a heatmap that shows cell visitation
    def heatmap(self, filename=None):
        plt.figure(figsize=(12, 12))
        plt.imshow(self.visit_counter, cmap='hot', interpolation='nearest')
        plt.title("Cell Visitation Heatmap")

        # Add text labels for the counters
        # for i in range(self.visit_counter.shape[0]):
        #     for j in range(self.visit_counter.shape[1]):
        #         plt.text(j, i, self.visit_counter[i, j], ha='center', va='center', color="red")

        if filename is None:
            return plt
        else:
            plt.savefig(filename)
            plt.close()

    
    def average_regret(self):
        if self.total_episodes == 0:
            return 0
        return self.cumulative_regret / self.total_episodes

if __name__=="__main__":
    ds = DeepSea(6, seed=10)
    
    episodes = 10
    
    for _ in range(episodes):
        done = False
        total = 0
        obs, _ = ds.reset()
        while not done: 
            print(ds.grid)
            action = int(input("Enter action: "))
            obs, reward, terminated, truncated, info = ds.step(action)
            print("reward is ", reward)
            total += reward
            done = terminated or truncated
            if done: 
                print("total reward is ", total)
    print("finished!")
    ds.heatmap(f"{30}x{30}_heatmap.png")
