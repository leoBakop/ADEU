import numpy as np
import gymnasium as gym
import wandb
from gym.wrappers import RecordVideo
from q_learning_agent import Q_Learning_Agent
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def evaluate(agent:Q_Learning_Agent, timesteps, evaluation_episodes, label, episode, iterations,desc, normalizer):
    env = make_evaluation_env(iterations, episode, label, desc)
    env.reset(seed = np.random.randint(low = 1, high = 9e+10))
    mean_reward = np.zeros(evaluation_episodes)
    mean_timesteps = np.zeros(evaluation_episodes)
    mean_sum_reward = np.zeros(evaluation_episodes)

    average_rewards = np.zeros(timesteps)
    for episode in range(evaluation_episodes):
        current_observation, info = env.reset()
        total_reward = 0
        for t in range(timesteps):
            action = agent.get_clear_action(current_observation)# User-defined policy function
            current_observation, reward, done, truncated, info = env.step(action) 
            if done :
                reward = + 100000 if ( reward == 1) else -10 #yeah that is correct
            else:
                reward = current_observation/normalizer
            
            total_reward += reward
            average_rewards[t] += reward/evaluation_episodes
            #if episode == 0:
                #wandb.log({domain+'evaluation_rewards':reward})
            mean_timesteps[episode] = t
            if done:
                mean_timesteps[episode] = t+1
                mean_reward[episode] = total_reward/(t+1)
                mean_sum_reward[episode] = total_reward
                break
    print('evaluatoion was ended')
    wandb.log({"mean_evaluation_reward": np.mean(mean_reward), 'duration': np.mean(mean_timesteps), 'cumm_evaluation_reward': np.mean(mean_sum_reward)})
    env.close()
    return np.mean(mean_reward), average_rewards, np.mean(mean_timesteps), np.mean(mean_sum_reward)




def make_evaluation_env(iterations, episode, label, desc, **kwargs):
    eval_env = gym.make('FrozenLake-v1',desc = desc, is_slippery = False, render_mode = 'rgb_array', **kwargs)
    
    if episode % iterations != 0: return eval_env 
    eval_env = RecordVideo(eval_env, video_folder = f'./videos/final_experiments/tabular/{label}/FrozenLake-v1/', episode_trigger= lambda x:  x % iterations == 0, name_prefix=f'FrozenLake-v1_true_epside_{episode}')
    
    
    return eval_env