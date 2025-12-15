import gymnasium as gym
import ale_py
import os
os.environ["SDL_RENDER_DRIVER"] = "software"
import time
from atari_env import AtariEnv, MAXIMUM_TIMESTEPS
import wandb
import argparse
from tqdm import tqdm
import random
import numpy as np
import torch
from bootstrap_dqn_agent import *
from atari_evaluate import evaluate


def wandb_session(domain:str , rho, lr, name = None, code = 0, debug = False, ):
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
        'learning_rate': lr,
        }
    )
    




def main(domain, 
         seed,
         code,
         rho):
    #at first let's seed everything except the environment
    set_seed(seed)
    debug = True
    #environment
    horizon = int(20e6)
    #horizon = int(10e4)
    gym.register_envs(ale_py)
    #we are setting the frameskip equal to 1 (so in practice no frameskip). We then apply it using the Wrapper
    env = gym.make(domain, repeat_action_probability = 0.25, frameskip = 1,  max_episode_steps=MAXIMUM_TIMESTEPS,)# render_mode = 'human')
    env = AtariEnv(env, frame_stacking=4, max_no_op_frames=30)
    #agent
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    print( '----------------------------------------')
    print( '-----begining of the training process---')
    print(f'-----this agent will run in {device}----')
    print( '----------------------------------------')


    rho = 0 if code < 0 else rho
    agents_parameters = {
                'K':10,
                'num_actions':env.action_space.n,
                'num_channels':4,
                #'learning_rate':6.25e-6,
                'learning_rate':5e-5,
                'device':device,
                'buffer_size':int(1e5),
                'batch_size':32,
                'beta':0.99, 
                'train_frequency':4, 
                'target_update':10_000,
                #'head_start':50_000,
                'head_start' : 64,
                'gamma':0.99,
                'cnn' : True,
                'state_size':-1

    }


    #aent construction
    if code == -1:
        maximum_reward = 1 #we know that from the Wrapper
        agents_parameters['initial_tau'] = 1 #This is a known upper bound (see Sutton & Barto 2018, Bertsekas 2019 etc)
        agents_parameters['final_tau'] = 0.01
        agents_parameters['exploration_duration'] = int(0.5*horizon)
        agents_parameters['epsilon_greedy'] = False
        agent = DQN(**agents_parameters)
    if code == -2:
        agent = BootDQN(**agents_parameters)
    if code == -3:
        agents_parameters['lambda_value'] = 0.1
        agent = BootUCB(**agents_parameters)

    if code == 3:
        agents_parameters['lambda_value'] = 10
        agents_parameters['rho'] = 0.1
        maximum_reward = 1 #we now that from the Wrapper
        agents_parameters['initial_tau'] = 1 #This is a known upper bound (see Sutton & Barto 2018, Bertsekas 2019 etc)
        agent = Adeu_Std(**agents_parameters)
    if code == 4:
        agents_parameters['initial_tau'] = 1 #This is a known upper bound (see Sutton & Barto 2018, Bertsekas 2019 etc)
        agents_parameters['final_tau'] = 0.01
        agents_parameters['exploration_duration'] = int(0.5*horizon)
        agents_parameters['epsilon_greedy'] = False
        
        agents_parameters['maximum_timesteps'] = MAXIMUM_TIMESTEPS
        agents_parameters['starting_mode'] = 'exploration'
        agent = Two_Modes_ADEU_DQN(**agents_parameters)
    
    name = reverse_code(code)+f', seed {seed}'
    wandb_session(domain = domain, code=code, debug = debug, rho = rho, name = name, lr = agents_parameters['learning_rate'])
    agent, cumm_eval_rewards = run( seed, 
                                    agent, 
                                    env, 
                                    horizon,
                                    code)
    
    path = f"./experiments/ATARI/{domain}/{name}/seed_{seed}"
    store(path, agent, cumm_eval_rewards, agents_parameters)



def store(path,agent:BootDQN, cumulative_rewards, agent_parameters:dict):
    
    if not os.path.exists(path+'/stored_data'):
        os.makedirs(path+'/stored_data')
    np.save(path+'/stored_data/cumulative_rewards.npy', cumulative_rewards)

    if agent_parameters.get('lambda_value', -1) != -1:
        agent_parameters['lambda_value'] == agent.get_lambda_value()

    full_path = path + "/hyperparameters.txt"

    with open(full_path, 'w') as f:
        for key, value in agent_parameters.items():
            f.write(f"{key}: {value}\n")

    agent.save(path)

def run(seed, agent:BootDQN | Two_Modes_ADEU_DQN, env:AtariEnv, horizon,code):
    
    loss = 0
    rewards = 0
    episodes_till_evaluate = 1
    noop = 0
    cumm_eval_rewards = []
    
    obs, info = env.reset(seed=seed)
    evaluation_every = int(horizon//50)
    
    for t in tqdm(range(horizon)):
        
        if t%evaluation_every == 0:
            cumulative_evaluation_reward = evaluate(agent, domain, seed, evaluation_episodes=30)
            return_q = (code == -3 ) or (code == 3) #in case of the adeu std or the Bootucb I would like also to get the mean q value among the training
            variance, q_mean = agent.get_variance(return_q)
            wandb.log({ 'Cumulative episodic reward':cumulative_evaluation_reward,
                        'mean episodic loss during training': loss/episodes_till_evaluate,
                        'mean episodic training reward': rewards/episodes_till_evaluate,
                        'frequency of noop (episodic)': noop/episodes_till_evaluate,
                        'mean variance at every timestep' : variance,
                        'variance over greediness': variance/q_mean if q_mean!=0 else -1
                        })
            episodes_till_evaluate = 0
            cumm_eval_rewards.append(cumulative_evaluation_reward)
            rewards = 0
            loss = 0
            noop = 0

        action = agent.get_action(obs)
        noop = noop+1 if action == 0 else noop

        new_obs, reward, terminated,truncated, info = env.step(action)
        terminated = terminated or truncated
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
            episodes_till_evaluate+=1
            obs, info = env.reset()
            agent.reset()

        #env.render()
    env.close()
    return agent, cumm_eval_rewards

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
    if agent_args.get('dqn', False): return -1
    if agent_args.get('bdqn', False): return -2
    if agent_args.get('ucb', False): return -3
    if agent_args.get('ppo', False):  return -4
    if agent_args.get('rnd', False) :  return -5

    if agent_args.get('adeu_std', False): return 3
    if agent_args.get('two_modes_adeu', False) : return 4
    return 0    

def reverse_code(code):
    if code == -1: return 'dqn'
    if code == -2: return 'bdqn'
    if code == -3: return 'ucb'
    if code == -4: return 'ppo'
    if code == -5 :return 'rnd'

    if code == 3: return 'adeu_std'
    if code == 4: return 'two_modes_adeu'
    return None
if __name__ == '__main__':

    seeds = 1



    parser = argparse.ArgumentParser(description='Your script description here')
    # Define hyperparameters as command-line arguments
    parser.add_argument('--DQN', action='store_true', default=False, help='Use the standard DQN')
    parser.add_argument('--BDQN', action='store_true', default=False, help='Use bootstrap DQN')
    parser.add_argument('--UCB', action='store_true', default=False, help='Use UCB built on top of BDQN')
    parser.add_argument('--adeu_std', action='store_true', default=False, help='Use the ensebmle method for measuring uncertainty')
    parser.add_argument('--two_modes_adeu', action='store_true', default=True, help='Use the ensebmle method for measuring uncertainty')
    parser.add_argument('--include_pos', action='store_true', default=False, help='Whether position of the agents should be included in the obs')
    parser.add_argument('--per', action='store_true', default=False, help='Would you like to Use the prioritized Experience Replay Buffer')
    parser.add_argument('--dom', type=str, default='ALE/Pong-v5', help='In what domain would you like to the agent to be tested')
    parser.add_argument('--rho', type=float, default=.0, help='what are the maximum timesteps per episode')
    args = parser.parse_args()

    # Access hyperparameters in your script
    dqn = args.DQN
    bdqn = args.BDQN
    ucb = args.UCB
    adeu_std = args.adeu_std
    two_modes_adeu = args.two_modes_adeu

    rho = args.rho

    domain = args.dom
    
    agent_args = {
        'dqn': dqn,
        'bdqn': bdqn,
        'ucb': ucb,
        'adeu_std': adeu_std,
        'two_modes_adeu' : two_modes_adeu
    }


   
    code = code_creation(agent_args)
    
    for s in range(seeds):
        main(domain, seed=s, code = code, rho = rho)