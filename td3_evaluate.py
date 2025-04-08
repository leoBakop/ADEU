import torch
import numpy as np
from td3_agent import TD3
import gymnasium as gym
import wandb
from gym.wrappers import RecordVideo
from utils import make_observation_franz_kitchen


def evaluate(agent:TD3, timesteps, evaluation_episodes, domain, label, episode, iterations, debug,kwargs,options = None ):
    env = make_evaluation_env(kwargs = kwargs, dom=domain, iterations=iterations, episode = episode, label = label, debug = debug)

    env.reset(seed = np.random.randint(low = 1, high = 9e+10))
    mean_reward = np.zeros(evaluation_episodes)
    mean_timesteps = np.zeros(evaluation_episodes)
    mean_sum_reward = np.zeros(evaluation_episodes)

    average_rewards = np.zeros(timesteps)
    for episode in range(evaluation_episodes):
        if options is not None:
            current_observation, info = env.reset(options = options)
        else:
            current_observation, info = env.reset()
        
        if domain == 'FrankaKitchen-v1':
            current_observation = make_observation_franz_kitchen(current_observation)
        elif isinstance(current_observation, dict):
            current_observation = np.concatenate([arr for arr in current_observation.values()])

        current_observation = torch.FloatTensor(current_observation)
        total_reward = 0
        for t in range(timesteps):
            action = agent.get_clear_action(current_observation)# User-defined policy function
            observation, reward, done, truncated, info = env.step(action)
            if domain == 'FrankaKitchen-v1':
                observation = make_observation_franz_kitchen(observation) 
            elif isinstance(observation, dict):
                observation = np.concatenate([arr for arr in observation.values()])
            current_observation = torch.FloatTensor(observation)
            total_reward += reward
            average_rewards[t] += reward/evaluation_episodes
            #if episode == 0:
                #wandb.log({domain+'evaluation_rewards':reward})
            if done or truncated or (t==timesteps -1):
                mean_timesteps[episode] = t+1
                mean_reward[episode] = total_reward/(t+1)
                mean_sum_reward[episode] = total_reward
                break
    wandb.log({domain+"mean_evaluation_reward": np.mean(mean_reward), 'duration': np.mean(mean_timesteps), 'cumm_evaluation_reward': np.mean(mean_sum_reward)})
    env.close()
    return np.mean(mean_reward), average_rewards, np.mean(mean_timesteps), np.mean(mean_sum_reward)

def make_evaluation_env(dom, iterations, episode, label, debug, kwargs):
    if debug:
        return gym.make(dom, render_mode = 'none', **kwargs)
    
    eval_env = gym.make(dom, render_mode = 'rgb_array', **kwargs)
    #having some problems with the rendering
    #from now on I will not generate videos
    return eval_env
    if episode % iterations != 0: return eval_env 
    eval_env = RecordVideo(eval_env, video_folder = f'./videos/final_experiments/TD3/{dom}/{label}/', episode_trigger= lambda x:  x % iterations == 0, name_prefix=f'{dom}_true_epside_{episode}')
    return eval_env

