import gymnasium as gym
import ale_py
import os
os.environ["SDL_RENDER_DRIVER"] = "software"
import time
from atari_env import AtariEnv
import wandb
import argparse
from tqdm import tqdm
import random
import numpy as np
import torch
from bootstrap_dqn_agent import *
from atari_evaluate import evaluate


def wandb_session(domain:str , rho, name = None, code = 0, debug = False,):
    # start a new wandb run to track this script
    domain = domain.replace("/", "_")
    project = f"Neurips-2025| Atari, Adaptive Exploration {domain}"
    project = "debug" if debug else project

    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        name = name,
        # track hyperparameters and run metadata
        config={
        "code":code,
        'rho':rho,
        }
    )
    




def main(domain, 
         seed,
         code,
         timesteps,
         rho):
    #at first let's seed everything except the environment
    set_seed(seed)
    debug = False
    #environment
    horizon = int(20e6)
    gym.register_envs(ale_py)
    env = gym.make(domain, repeat_action_probability = 0.25, max_episode_steps=timesteps,)# render_mode = 'human')
    env = AtariEnv(env, frame_stacking=4, max_no_op_frames=30)
    #agent
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    rho = 0 if code < 0 else rho
    agents_parameters = {
                'K':10,
                'num_actions':env.action_space.n,
                'num_channels':4,
                'learning_rate':6.25e-6,
                'device':device,
                'buffer_size':int(1e5),
                'batch_size':32,
                'beta':0.99, 
                'train_frequency':4, 
                'target_update':10_000,
                'head_start':50_000,
                #'head_start' : 64
    }


    #aent construction
    if code == -1:
        agents_parameters['initial_tau'] = 100
        agents_parameters['final_tau'] = 0.5
        agents_parameters['exploration_duration'] = int(horizon/1.5)
        agents_parameters['epsilon_greedy'] = True
        agent = DQN(**agents_parameters)
    if code == -2:
        agent = BootDQN(**agents_parameters)
    name = reverse_code(code)+f', seed {seed}'
    wandb_session(domain = domain, code=code, debug = debug, rho = rho, name = name)
    loss = 0
    rewards = 0
    noop = 0
    print( '----------------------------------------')
    print( '-----begining of the training process---')
    print(f'-----this agent will run in {device}----')
    print( '----------------------------------------')
    obs, info = env.reset(seed=seed)
    evaluation_every = int(horizon//20)

    for t in tqdm(range(horizon)):

        if t%evaluation_every == 0:
            cumulative_evaluation_reward = evaluate(agent, domain, timesteps, seed, evaluation_episodes=30)
            wandb.log({ 'Cumulative episodic reward':cumulative_evaluation_reward,
                        'mean loss during training': loss/evaluation_every,
                        'mean training reward': rewards/evaluation_every,
                        'frequency of noop': noop 
                        })

            rewards = 0
            loss = 0
            noop = 0
        action = agent.get_action(obs)

        noop = noop+1 if action == 0 else noop

        new_obs, reward, terminated, info = env.step(action)
        l = agent.step(state = obs,
                   action = action,
                   reward = reward,
                   next_state=new_obs,
                   done = terminated)
        loss+=l
        rewards+=reward
        obs = new_obs
        #if info['lives'] == 0 and terminated:
        if terminated:
            obs, info = env.reset()
            agent.reset()

        #env.render()
    env.close()



def store(path,agent:BootDQN, label,seed, cumulative_rewards, agent_parameters):
    path = f"./experiments/ATARI/{domain}/{label}/seed_{seed}"
    if not os.path.exists(path+'/stored_data'):
        os.makedirs(path+'/stored_data')
    np.save(path+'/stored_data/cumulative_rewards.npy', cumulative_rewards)

    full_path = path + "/hyperparameters.txt"

    with open(full_path, 'w') as f:
        for key, value in agent_parameters.items():
            f.write(f"{key}: {value}\n")

    agent.save(path)

def set_seed(seed):
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    print("Setting seeds ...... \n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=  True

def code_creation(agent_args):
    if agent_args['dqn']: return -1
    if agent_args['bdqn']: return -2
    if agent_args['ucb']: return -3
    return 0    

def reverse_code(code):
    if code == -1: return 'dqn'
    if code == -2: return 'bdqn'
    if code == -3: return 'ucb'
    return None
if __name__ == '__main__':

    seeds = 3



    parser = argparse.ArgumentParser(description='Your script description here')
    # Define hyperparameters as command-line arguments
    parser.add_argument('--DQN', action='store_true', default=False, help='Use the standard DQN')
    parser.add_argument('--BDQN', action='store_true', default=True, help='Use bootstrap DQN')
    parser.add_argument('--UCB', action='store_true', default=False, help='Use UCB built on top of BDQN')
    parser.add_argument('--adeu_ensemble', action='store_true', default=False, help='Use the ensebmle method for measuring uncertainty')
    parser.add_argument('--include_pos', action='store_true', default=False, help='Whether position of the agents should be included in the obs')
    parser.add_argument('--per', action='store_true', default=False, help='Would you like to Use the prioritized Experience Replay Buffer')
    parser.add_argument('--dom', type=str, default='ALE/Pong-v5', help='In what domain would you like to the agent to be tested')
    parser.add_argument('--steps', type=int, default=10_000, help='what are the maximum timesteps per episode')
    parser.add_argument('--rho', type=float, default=.0, help='what are the maximum timesteps per episode')
    args = parser.parse_args()

    # Access hyperparameters in your script
    dqn = args.DQN
    bdqn = args.BDQN
    ucb = args.UCB

    rho = args.rho

    domain = args.dom
    timesteps = args.steps
    
    agent_args = {
        'dqn': dqn,
        'bdqn': bdqn,
        'ucb': ucb
    }


   
    code = code_creation(agent_args)
    
    for s in range(seeds):
        main(domain, seed=s, code = code, timesteps = timesteps, rho = rho)