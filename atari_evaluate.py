
import gymnasium as gym
import ale_py
from atari_env import AtariEnv
import numpy as np
import torch
from bootstrap_dqn_agent import BootDQN



def evaluate(agent:BootDQN, domain, timesteps, seed, evaluation_episodes = 30):
    env = make_environment(domain, timesteps)
    obs, _ = env.reset(seed=seed)
    cumulative_rewards_per_episode = np.zeros(evaluation_episodes)
    
    for episode in range(evaluation_episodes):
        rewards = 0

        while True:
            action = agent.get_clear_action(obs)
            new_obs, r, terminated, info = env.step(action)
            rewards+=r
            obs = new_obs
            #if info['lives'] == 0 and terminated:
            if terminated:
                obs, info = env.reset()
                cumulative_rewards_per_episode[episode] = rewards
                break

    env.close()
    return np.mean(cumulative_rewards_per_episode)


def make_environment(domain, timesteps):
    gym.register_envs(ale_py)
    env = gym.make(domain, repeat_action_probability = .0, max_episode_steps=timesteps, frameskip = 1)
    env = AtariEnv(env, frame_stacking=4, max_no_op_frames=30)
    return env