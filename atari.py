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
from bootstrap_dqn_agent import BootDQN



def main(domain, 
         seed,
         code,
         timesteps):
    #at first let's seed everything except the environment
    set_seed(seed)
    #environment
    horizon = int(40e6)
    gym.register_envs(ale_py)
    env = gym.make(domain, render_mode = 'human', repeat_action_probability = 0.25, max_episode_steps=timesteps)
    env = AtariEnv(env, frame_stacking=4, max_no_op_frames=5)
    #agent
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    agents_parameters = {
                'K':10,
                'num_actions':env.action_space.n,
                'num_channels':4,
                'learning_rate':6.25e-6,
                'device':device,
                'buffer_size':int(1e6),
                'batch_size':32,
                'beta':0.99, 
                'train_frequency':4, 
                'target_update':10000,
                #'head_start':50000,
                'head_start' : 64
    }


    i = 0
    agent = BootDQN(**agents_parameters)



    obs, info = env.reset(seed=seed)


    for t in range(horizon):
        action = agent.get_action(obs, active_head = np.random.choice(agent.K))
        new_obs, reward, terminated, info = env.step(action)
        agent.step(state = obs,
                   action = action,
                   reward = reward,
                   next_state=new_obs,
                   done = terminated)
        obs = new_obs
        if info['lives'] == 0 and terminated:
            obs, info = env.reset()

        env.render()
        time.sleep(0.1)
    env.close()



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
    parser.add_argument('--dom', type=str, default='ALE/Breakout-v5', help='In what domain would you like to the agent to be tested')
    parser.add_argument('--steps', type=int, default=280, help='what are the maximum timesteps per episode')

    args = parser.parse_args()

    # Access hyperparameters in your script
    dqn = args.DQN
    bdqn = args.BDQN
    ucb = args.UCB

    domain = args.dom
    timesteps = args.steps
    
    agent_args = {
        'dqn': dqn,
        'bdqn': bdqn,
        'ucb': ucb
    }


   
    code = code_creation(agent_args)
    
    for s in range(seeds):
        main(domain, seed=s, code = code, timesteps = timesteps)