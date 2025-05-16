import numpy as np
import gymnasium as gym
import wandb
from gym.wrappers import RecordVideo
from q_learning_agent import Q_Learning_Agent, ADEU_Q_Learning
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import os


def evaluate(agent:Q_Learning_Agent | ADEU_Q_Learning, timesteps, evaluation_episodes, label, episode, iterations,desc, normalizer, domain, save_experience = False,  wandb_log = True, keep_trajectories = False):
    """ We should note that Keep_trajectory refers to keeping the trajectory within an episode. Thus, we can create the figure
        that outputs the policy, the exploration policy and the variance etc.
        save_experience saves the experience to reuse it in the safety example    
    """
    
    dim = len(desc)
    env = make_evaluation_env(iterations, episode, label, desc, domain)
    env.reset(seed = np.random.randint(low = 1, high = 9e+10))
    mean_reward = np.zeros(evaluation_episodes)
    mean_timesteps = np.zeros(evaluation_episodes)
    mean_sum_reward = np.zeros(evaluation_episodes)
    experience = []
    average_rewards = np.zeros(timesteps)
    #these variables will store the original policy, the actions proovided by the exploration strategy
    #and the variance of the agent.
    if keep_trajectories:
        path = f'./experiments/tabular/{domain}/adeu_figure/episode{episode}'
        policy = []
        explortaion_strategy = []
        variance = []
    for episode in range(evaluation_episodes):
        current_observation, info = env.reset()
        total_reward = 0
        for t in range(timesteps):
            action = agent.get_clear_action(current_observation)# User-defined policy function
            new_observation, reward, done, truncated, info = env.step(action) 
            row, col = reverse_hash(new_observation,dim)
            if keep_trajectories and episode == 0:
                policy.append(action)
                explortaion_strategy.append(agent.get_action(current_observation))
                variance.append(agent.get_exploration_rate())
            if desc[row][col] == 'H':
                reward = 0
                done = True
                truncated= True
            elif desc[row][col] == 'G':
                reward = 100_000
                done = True
                truncated= True
            else:
                reward = new_observation/normalizer
                done = False
                truncated = False
                
                
            
            total_reward += reward
            if save_experience:
                experience.append([current_observation, action, reward, new_observation, done or truncated])
            average_rewards[t] += reward/evaluation_episodes
            #if episode == 0:
                #wandb.log({domain+'evaluation_rewards':reward})
            mean_timesteps[episode] = t
            current_observation = new_observation

            if done or t == timesteps - 1:
                mean_timesteps[episode] = t+1
                mean_reward[episode] = total_reward/(t+1)
                mean_sum_reward[episode] = total_reward
                break
    if keep_trajectories:
        store_trajectory(path=path, policy = np.array(policy), exploration_strategy=np.array(explortaion_strategy), variance=np.array(variance))
    if wandb_log:
        wandb.log({"mean_evaluation_reward": np.mean(mean_reward), 'duration': np.mean(mean_timesteps), 'cumm_evaluation_reward': np.mean(mean_sum_reward)})
    env.close()
    return np.mean(mean_reward), average_rewards, np.mean(mean_timesteps), np.mean(mean_sum_reward), experience




def make_evaluation_env(iterations, episode, label, desc, domain, **kwargs):
    eval_env = gym.make('FrozenLake-v1',desc = desc, is_slippery = False, render_mode = 'rgb_array', **kwargs)
    return eval_env
    if episode % iterations != 0: return eval_env 
    if episode == 0: return eval_env
    eval_env = RecordVideo(eval_env, video_folder = f'./videos/final_experiments/tabular/{label}/{domain}/', episode_trigger= lambda x:  x % iterations == 0, name_prefix=f'true_epside_{episode}')
   
    
    return eval_env


def reverse_hash(hash_code, ncols):
    row = hash_code // ncols
    col = hash_code % ncols
    return row, col
    

def store_trajectory(path, policy, exploration_strategy, variance):
    if not os.path.exists(path):
        os.makedirs(path) 
    
    np.save(path+'/policy.npy', policy)
    np.save(path+'/exploration_strategy.npy', exploration_strategy)
    np.save(path+'/variance.npy', variance)