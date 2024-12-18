import gymnasium as gym
import numpy as np
import random
from q_learning_agent import Q_Learning_Agent, IR_Q_Learning_Agent
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
    
def wandb_session(domain , seed = 0, name = None, code = 0, debug = False, beta = 0):
    # start a new wandb run to track this script
    project = "debug" if debug else f"Mod RND| Tabular, Adaptive Exploration {domain}"
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        name = name,
        # track hyperparameters and run metadata
        config={
        "seed": seed,
        "code":code,
        'beta':beta
        }
    )
    

def main(seed, code, label):
    set_seed(seed)
    name = f'{label}, seed:{seed}'
    
    """ State code in frozen lake
        0  1   2   3
        4  5   6   7 
        8  9  10  11
        12 13 14  15"""
        
        
    """ or State code in frozen lake
        0  1   2   3  4  5  
        6  7   8   9  10 11  
        12 13  14  15 16 17
        18 19  20  21 22 23
        24 25  26  27 28 29
        30 31  32  33 34 35"""    
    
    render = False
    dim = 17
    desc = generate_random_map(size=dim)
    env = gym.make('FrozenLake-v1',  desc=desc, is_slippery = False, render_mode = "rgb_array")
    env.reset(seed = seed)
    agent_params = {
        'state_size':env.observation_space.n, 
        'action_size':env.action_space.n
    }
    if code == 2:
        agent_params['beta'] = 0.8
        agent = IR_Q_Learning_Agent(**agent_params)
    else: 
        agent = Q_Learning_Agent(**agent_params)
    wandb_session(domain = 'FrozenLake-v1', seed = seed, debug = False, code = code, name = name, beta = agent_params['beta'] if code ==2 else 0)

    state_count = run(env=env, agent=agent, render=render, horizon = 1_000_000, desc = desc, dim= dim, label=label)
    store(filename = f'experiments/tabular/FrozenLake/{label}/seed_{seed}', state_count = state_count, agent_params=agent_params)
    
def store(filename, state_count, agent_params):
    if not os.path.exists(filename+'/stored_data'):
        os.makedirs(filename+'/stored_data')
    
    np.save(filename+'/stored_data/state_count.npy', state_count)
    full_path = filename + "/hyperparameters.txt"

    with open(full_path, 'w') as f:
        for key, value in agent_params.items():
            f.write(f"{key}: {value}\n")


def run(env:gym.Env, agent:Q_Learning_Agent, horizon = 1_000, timesteps = 300, render = False, label = 0, desc = None, dim = 50):
    epsilon = np.linspace(1, 0.1, int(0.5 * horizon))
    percentage = 1
    normalizer = dim*(dim+1)/2
    states_count = np.zeros([dim,dim])
    for episode in tqdm(range(horizon)):
        #after some timesteps perform the evaluation
        if episode % int(round(100/percentage)) == 0:
            evaluate(agent, timesteps, evaluation_episodes = 3, label = label, episode = episode, iterations = int(0.2*horizon), desc = desc, normalizer=normalizer)
            
        agent.set_eps(epsilon[episode if episode < epsilon.size else -1])
        obs, _ = env.reset()
        for t in range(timesteps):
            states_count[reverse_hash(obs,dim)]+=1
            if render: env.render()
            action = agent.get_action(obs)
            new_obs, reward, done, truncated, info = env.step(action)
            if done :
                reward = + 100000 if ( reward == 1) else -10 #yeah that is correct
            else:
                reward = new_obs/normalizer
                 
            agent.step([obs, action, reward, new_obs, done or truncated])
            obs = new_obs
            
            if done or truncated:
                break
    env.close()
    wandb.finish()
    return states_count
 
 
def reverse_hash(hash_code, ncols):
    row = hash_code // ncols
    col = hash_code % ncols
    return row, col
    
def code_creation(adaptive_exploration=True, use_rnd = False, ucb = False,):
    
    retval = 0
    if ucb:return retval + 5
    if adaptive_exploration: return retval + 1
    if use_rnd: return retval + 2
    return retval
    
if  __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Your script description here')
    # Define hyperparameters as command-line arguments
    parser.add_argument('--ade', action='store_true', default=False, help='Would you like to use Adaptive Exploration')
    parser.add_argument('--im', action='store_true', default=False, help='Would you like to use the standrad RND')
    parser.add_argument('--ucb', action='store_true', default=False, help='Would you like to use the standrad RND')

    args = parser.parse_args()

    # Access hyperparameters in your script
    use_im = args.im
    use_ade = args.ade
    use_ucb = args.ucb
    
    arguments = {
        'ade': use_ade,
        'intrisic motivation':use_im,
        'ucb':use_ucb,
        'Baseline': (not use_ade) and (not use_im) and (not use_ucb)
    }
    
    
    label = [a for a in arguments.keys() if arguments[a] ][0]
    
    seed = 10
    code = code_creation(adaptive_exploration=use_ade, use_rnd = use_im, ucb=use_ucb )
    for s in range(seed):
        main(seed = s, code = code, label = label)