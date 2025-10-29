
import gymnasium as gym
import ale_py
from atari_env import AtariEnv
import numpy as np
import torch
from bootstrap_dqn_agent import BootDQN
import deepsea


def evaluate(n, agent:BootDQN, seed, evaluation_episodes = 30):
    env = make_environment(n)
    obs, _ = env.reset(seed=seed)
    cumulative_rewards_per_episode = np.zeros(evaluation_episodes)
    
    for episode in range(evaluation_episodes):
        rewards = 0

        while True:
            action = agent.get_clear_action(obs)
            new_obs, r, terminated, truncated, info = env.step(action)
            terminated = terminated or truncated
            rewards+=r
            obs = new_obs
            #if info['lives'] == 0 and terminated:
            if terminated:
                obs, info = env.reset()
                cumulative_rewards_per_episode[episode] = rewards
                break

    env.close()
    return np.mean(cumulative_rewards_per_episode)


def make_environment(N):
    env = gym.make('DeepSea', N=N,)
    return env