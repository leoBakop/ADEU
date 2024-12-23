""" This script tries to simulate the whole backround knowledge in a tabular environment.
In more detail, it generates some previous knowledge (i.e a q-table) using simple q-learning. 
Then, some rollouts are performed using the q-learning agent, to create some training tuples (i.e. dataset)
for some speciffic states.
The tested agents, are trained offline in that dataset and afterwards, the trained as in tabular_main.py"""

import gymnasium as gym
import numpy as np
import random
from q_learning_agent import Q_Learning_Agent, IR_Q_Learning_Agent, UCB_Q_Learning, ADEU_Q_Learning
from tabular_evaluate import evaluate
import wandb
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import argparse
import sys
import os

def set_seed(seed):
    print('setting seeds')
    random.seed(seed)
    np.random.seed(seed)
    
def wandb_session(domain , seed = 0, name = None, code = 0, debug = False, beta = 0, heads = 0):
    # start a new wandb run to track this script
    project = f"Mod RND| Tabular-Background knowledge, Adaptive Exploration {domain}"
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        name = name,
        # track hyperparameters and run metadata
        config={
        "code":code,
        'beta':beta,
        'heads':heads 
        }
    )
    
    
def generate_agent(map, env:gym.Env, agent:Q_Learning_Agent, timesteps, horizon, dim):
    epsilon = np.linspace(1, 0.1, int(0.5 * horizon))
    normalizer = dim*(dim+1)/2
    render = False
    states_count = np.zeros([dim,dim])
    for episode in tqdm(range(horizon)):
        #after some timesteps perform the evaluation
        agent.set_eps(epsilon[episode if episode < epsilon.size else -1])
        obs, _ = env.reset()
        for t in range(timesteps):
            states_count[reverse_hash(obs,dim)]+=1
            if render: env.render()
            
            action = agent.get_action(obs)
            
            new_obs, reward, done, truncated, info = env.step(action)
            if done :
                reward = + 100000 if ( reward == 1) else -10
            else:
                reward = new_obs/normalizer
               
            agent.step([obs, action, reward, new_obs, done or truncated])
            obs = new_obs
            
            if done or truncated:
                break
    env.close()
    return states_count, agent, normalizer

def reverse_hash(hash_code, ncols):
    row = hash_code // ncols
    col = hash_code % ncols
    return row, col


def rollout(agent:Q_Learning_Agent, normalizer, timesteps, horizon, map):
    """ creates the dataset """

    for ep in range(horizon):
        *_, experience = evaluate(agent = agent, timesteps = timesteps, evaluation_episodes = 1, label = 'creating_rollout', episode = ep, iterations = 10, desc=map, normalizer=normalizer, save_experience=True, wandb_log=False)
    


def main(seed):
    dim = 12
    desc = generate_random_map(size=dim, seed = seed)
    env = gym.make('FrozenLake-v1',  desc=desc, is_slippery = False, render_mode = "rgb_array")
    env.reset(seed = seed)
    agent_params = {
        'state_size':env.observation_space.n, 
        'action_size':env.action_space.n
    }
    agent = Q_Learning_Agent(**agent_params)
    timesteps = 300
    states_count, agent, normalizer = generate_agent (map = desc, env = env, agent = agent, timesteps = timesteps, horizon = int(1e4), dim=dim)
    rollout(agent=agent, normalizer=normalizer, timesteps=timesteps, horizon=1, map =desc)



if __name__=='__main__':
    main(seed = 1)