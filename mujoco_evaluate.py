import torch
import numpy as np

from agent import Agent

import gymnasium as gym
import wandb
from gym.wrappers import RecordVideo



def evaluate(agent:Agent, timesteps, evaluation_episodes, domain, label, episode, iterations,**kwargs):
    env = make_evaluation_env(dom=domain, iterations=iterations, episode = episode, label = label, **kwargs)
    env.reset(seed = np.random.randint(low = 1, high = 9e+10))
    mean_reward = np.zeros(evaluation_episodes)
    mean_timesteps = np.zeros(evaluation_episodes)
    mean_sum_reward = np.zeros(evaluation_episodes)

    average_rewards = np.zeros(timesteps)
    for episode in range(evaluation_episodes):
        current_observation, info = env.reset()
        if isinstance(current_observation, dict):
            current_observation = np.concatenate([arr for arr in current_observation.values()])

        current_observation = torch.FloatTensor(current_observation)
        total_reward = 0
        for t in range(timesteps):
            action = agent.get_clear_action(current_observation)# User-defined policy function
            action=np.array(action.cpu().squeeze().numpy().reshape((agent.action_size,)))
            observation, reward, done, truncated, info = env.step(action) 
            if isinstance(observation, dict):
                observation = np.concatenate([arr for arr in observation.values()])
            current_observation = torch.FloatTensor(observation)
            total_reward += reward
            average_rewards[t] += reward/evaluation_episodes
            #if episode == 0:
                #wandb.log({domain+'evaluation_rewards':reward})
            if done or truncated:
                mean_timesteps[episode] = t+1
                mean_reward[episode] = total_reward/(t+1)
                mean_sum_reward[episode] = total_reward
                break
    wandb.log({domain+"mean_evaluation_reward": np.mean(mean_reward), 'duration': np.mean(mean_timesteps), 'cumm_evaluation_reward': np.mean(mean_sum_reward)})
    env.close()
    return np.mean(mean_reward), average_rewards, np.mean(mean_timesteps), np.mean(mean_sum_reward)

def make_evaluation_env(dom, iterations, episode, label, **kwargs):
    eval_env = gym.make(dom, render_mode = 'rgb_array', **kwargs)
    if episode % iterations != 0: return eval_env 
    eval_env = RecordVideo(eval_env, video_folder = f'./videos/final_experiments/naf/{dom}/{label}/', episode_trigger= lambda x:  x % iterations == 0, name_prefix=f'{dom}_true_epside_{episode}')
    return eval_env