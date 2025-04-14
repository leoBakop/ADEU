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



KEEP_TRAJECTORY = True


def set_seed(seed):
    print('setting seeds')
    random.seed(seed)
    np.random.seed(seed)
    
def wandb_session(domain , seed = 0, name = None, code = 0, debug = False, beta = 0, heads = 0):
    # start a new wandb run to track this script
    
    project = "debug" if debug else f"ECAI-2025| Tabular, Adaptive Exploration {domain}"
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


def generate_custom_map(dim):
    desc = []
    initial_line = "SF" + "".join(["H" for _ in range(dim - 2)])
    desc.append(initial_line)
    #constructing the vertical path
    for _ in range(3, dim): 
        desc.append("HF" + "".join(["H" for _ in range(dim - 2)]))   
    line_before= ("HF" + "".join(["F" for _ in range(dim - 2)]))     
    #cosntructing the horizontal path
    desc.append(line_before)
    final_line = "".join(["H" for _ in range(dim - 1)])+"G"
    desc.append(final_line)
    return desc

def main(seed, code, label, heads = 0, type_of_map = 0):
    set_seed(seed)
    beta = 0
    name = f'{label}, seed:{seed}'
    
    """ State code in frozen lake
        0  1   2   3
        4  5   6   7 
        8  9  10  11
        12 13 14  15"""
        
        
    """ or another state code in frozen lake
        0  1   2   3  4  5  
        6  7   8   9  10 11  
        12 13  14  15 16 17
        18 19  20  21 22 23
        24 25  26  27 28 29
        30 31  32  33 34 35"""    
    
    render = False
    dim =  1500 if tofl > 0 else 50
    if type_of_map == 0:
        desc = generate_random_map(size=dim, seed = seed)
    else:
        desc = generate_custom_map(dim = dim)
        
    env = gym.make('FrozenLake-v1',  desc=desc, is_slippery = False, render_mode = None)
    env.reset(seed = seed)
    agent_params = {
        'state_size':env.observation_space.n, 
        'action_size':env.action_space.n
    }
    if code == 2:
        beta = 0.8
        agent_params['beta'] = beta
        agent = IR_Q_Learning_Agent(**agent_params)
    elif code == 5:
        if heads == 1: beta = 0
        else: beta = 10
        agent_params['num_of_agents'] = heads
        agent_params['lambda_value'] = beta
        agent_params['p'] = 0.99
        agent = UCB_Q_Learning(**agent_params)
    elif code == 1:
        beta = 0.01
        agent_params['beta'] = beta
        agent_params['rollout_frequency'] = 0.05
        agent = ADEU_Q_Learning(**agent_params)
    else: 
        agent = Q_Learning_Agent(**agent_params)
    
    domain = "FrozenLake" if type_of_map == 0 else "CustomFrozenLake"
    wandb_session(domain = domain, seed = seed, debug = False, code = code, name = name, beta = beta, heads = heads)

    state_count = run(env=env, 
                      domain = domain, 
                      agent=agent, 
                      render=render, 
                      horizon = int(1e5), 
                      desc = desc, 
                      dim= dim, 
                      label=label, 
                      code = code, 
                      tofl = tofl, 
                      timesteps = int(3*dim))
    store(filename = f'experiments/tabular/{domain}/{label}/seed_{seed}', state_count = state_count, agent_params=agent_params)
    
def store(filename, state_count, agent_params):
    if not os.path.exists(filename+'/stored_data'):
        os.makedirs(filename+'/stored_data')
    
    np.save(filename+'/stored_data/state_count.npy', state_count)
    full_path = filename + "/hyperparameters.txt"

    with open(full_path, 'w') as f:
        for key, value in agent_params.items():
            f.write(f"{key}: {value}\n")


def run(env:gym.Env, agent:Q_Learning_Agent|ADEU_Q_Learning, code, domain, horizon = 1_000, timesteps = 1000, render = False, label = 0, desc = None, dim = 50, tofl = 0, seed = 0):
    initial_exploration_rate = 1 
    epsilon = np.linspace(initial_exploration_rate, 0.05, int(0.6 * horizon))
    percentage = 1
    normalizer = dim*(dim+1)/2
    mean_std = 0
    states_count = np.zeros([dim,dim])
    for episode in tqdm(range(horizon)):
        if code == 1 and np.random.rand()<agent.rollout_frequency:
            agent.rollout = True
        
        #after some timesteps perform the evaluation
        if episode % int(round(100/percentage)) == 0:
            keep_trajectory =( seed == 0 and type(agent) == ADEU_Q_Learning and KEEP_TRAJECTORY)
            evaluate(agent, timesteps, domain = domain, evaluation_episodes = 3, label = label, episode = episode, iterations = int(0.2*horizon), desc = desc, normalizer=normalizer, keep_trajectories=keep_trajectory)
            wandb.log({"std error during training": mean_std})
            mean_std = 0
        agent.set_eps(epsilon[episode if episode < epsilon.size else -1])
        obs, _ = env.reset()
        row, col = reverse_hash(obs,dim)
        states_count[row, col]+=1
        for t in range(timesteps):
            
            if render: env.render()
            
            if code == 5: 
                action, std = agent.get_action(obs)
                mean_std += std*percentage/(timesteps*100)
            else: action = agent.get_action(obs)
            
            if tofl != 0 and action == 3:
                action = np.random.choice(np.arange(2), size = 1)[0]
            new_obs, reward, done, truncated, info = env.step(action)

            row, col = reverse_hash(new_obs,dim)
            states_count[row, col]+=1
            if desc[row][col] == 'H':
                reward = -10
                done = True
            elif desc[row][col] == 'G':
                print ('target was founded')
                reward = 100_000
                done = True
            elif desc[row][col] == 'S':
                reward = 0
                done = False
                truncated = False
            else:
                reward = dim*new_obs/normalizer
                done = False
                truncated = False
                
            if code == 5:
                mask = np.random.binomial(1, agent.p, agent.num_of_agents)
                agent.step([obs, action, reward, new_obs, done or truncated, mask])
            else:
                agent.step([obs, action, reward, new_obs, done or truncated])
            obs = new_obs
            
            if (done):
                break
        if code == 1: 
            agent.rollout = False
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
    parser.add_argument('--im', action='store_true', default=False, help='Would you like to use the standrad RND')
    parser.add_argument('--ucb', action='store_true', default=False, help='Would you like to use the standrad RND')
    parser.add_argument('--adeu', action='store_true', default=True, help='Would you like to use the standrad RND')
    parser.add_argument('--heads', type=int, default=-1, help='Would you like to use the standrad RND')
    #tofl means type of frozen lake
    parser.add_argument('--tofl', type=int, default=1, help='Would you like to use the standrad RND')
    args = parser.parse_args()

    # Access hyperparameters in your script
    use_im = args.im
    use_ade = args.adeu
    use_ucb = args.ucb
    heads = args.heads
    tofl = args.tofl
    arguments = {
        'adeu': use_ade,
        'intrisic motivation':use_im,
        'ucb':use_ucb,
        'Baseline': (not use_ade) and (not use_im) and (not use_ucb)
    }
    
    
    label = [a for a in arguments.keys() if arguments[a]][0]
    
    seed = 10
    code = code_creation(adaptive_exploration=use_ade, use_rnd = use_im, ucb=use_ucb )
    for s in range(seed):
        main(seed = s, code = code, label = label, heads = heads, type_of_map=tofl)
