import gymnasium as gym
import os
os.environ["SDL_RENDER_DRIVER"] = "software"
import wandb
import argparse
from tqdm import tqdm
import random
import numpy as np
import torch
from bootstrap_dqn_agent import *
from deepsea_evaluate import evaluate
import deepsea



def wandb_session(rho, lr, n, name = None, code = 0, debug = False, ):
    # start a new wandb run to track this script
    
    project = f"Neurips-2025| DeepSea, Adaptive Exploration"
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
        'n':n
        }
    )


def main(
         n,
         seed,
         code,
         rho):
    #at first let's seed everything except the environment
    set_seed(seed)
    debug = False
    #environment
    horizon = int(1e6)
    #horizon = int(10e4)
    
    #we are setting the frameskip equal to 1 (so in practice no frameskip). We then apply it using the Wrapper
    env = gym.make('DeepSea', stochastic = False, N=n,)# render_mode = 'human')
    
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    print( '----------------------------------------')
    print( '-----begining of the training process---')
    print(f'-----this agent will run in {device}----')
    print(f'-----------------N = {n} ---------------')
    print( '----------------------------------------')

    obs, info = env.reset(seed=seed)
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
                'head_start':50_000,
                #'head_start' : 64,
                'gamma':0.99,
                'cnn': False,
                'state_size' : obs.shape[0],
    }


    #agent construction
    if code == -1:
        maximum_reward = 1 #we know that from the Wrapper
        agents_parameters['initial_tau'] = 1 #This is a known upper bound (see Sutton & Barto 2018, Bertsekas 2019 etc)
        agents_parameters['final_tau'] = 0.01
        agents_parameters['exploration_duration'] = int(0.5*horizon)
        agents_parameters['epsilon_greedy'] = True
        agent = DQN(**agents_parameters)
    if code == -2:
        agent = BootDQN(**agents_parameters)
    if code == -3:
        agents_parameters['lambda_value'] = 0.1
        agent = BootUCB(**agents_parameters)

    if code == 1:

        agents_parameters['exploration_duration'] = int(0.5*horizon)
        agents_parameters['N'] = n
        agent = EZ_ADEU(**agents_parameters)
        
    
    name = reverse_code(code)+f', seed {seed}'
    wandb_session(code=code, debug = debug, rho = rho, name = name, lr = agents_parameters['learning_rate'], n = n)
    agent, cumm_eval_rewards, total_timesteps = run( n,
                                    seed, 
                                    agent, 
                                    env, 
                                    horizon,
                                    code)
    
    path = f"./experiments/DEEPSEA/{reverse_code(code)}/{n}/seed_{seed}"
    store(path, agent, cumm_eval_rewards, agents_parameters, total_timesteps)
    wandb.finish()
    return total_timesteps == horizon


def store(path,agent:BootDQN, cumulative_rewards, agent_parameters:dict, total_timesteps):
    
    if not os.path.exists(path+'/stored_data'):
        os.makedirs(path+'/stored_data')
    np.save(path+'/stored_data/cumulative_rewards.npy', cumulative_rewards)

    if agent_parameters.get('lambda_value', -1) != -1:
        agent_parameters['lambda_value'] = agent.get_lambda_value()
    agent_parameters['solved_in'] = total_timesteps
    full_path = path + "/hyperparameters.txt"

    with open(full_path, 'w') as f:
        for key, value in agent_parameters.items():
            f.write(f"{key}: {value}\n")

    #agent.save(path)

def run(n, seed, agent:BootDQN, env:gym.Env, horizon,code):
    
    loss = 0
    rewards = 0
    episodes_till_evaluate = 1
    noop = 0
    cumm_eval_rewards = []
    
    obs, info = env.reset()
    evaluation_every = int(horizon//100)
    total_timesteps = 0
    consecutive_succesful_evaluations = 0

    for t in tqdm(range(horizon)):
        
        if t%evaluation_every == 0:
            cumulative_evaluation_reward, solved = evaluate(n, agent, seed, evaluation_episodes=30)
            
            return_q = (code == -3 ) or (code == 3) #in case of the adeu std or the Bootucb I would like also to get the mean q value among the training
            variance, q_mean = agent.get_variance(return_q)
            wandb.log({ 'Cumulative episodic reward':cumulative_evaluation_reward,
                        'mean episodic loss during training': loss/episodes_till_evaluate,
                        'mean episodic training reward': rewards/episodes_till_evaluate,
                        'frequency of selecting the wrong action (episodic)': noop/episodes_till_evaluate,
                        'mean variance at every timestep' : variance,
                        'variance over greediness': variance/q_mean if q_mean!=0 else -1
                        })
            episodes_till_evaluate = 0
            cumm_eval_rewards.append(cumulative_evaluation_reward)
            rewards = 0
            loss = 0
            noop = 0

            if solved:
                consecutive_succesful_evaluations +=1
                if consecutive_succesful_evaluations > 9:
                    break
            else: consecutive_succesful_evaluations = 0 

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
        total_timesteps+=1
        #env.render()
    env.close()
    return agent, cumm_eval_rewards, total_timesteps






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
    if agent_args['adeu_std']: return 3
    if agent_args['ez_adeu']: return 1
    return 0    

def reverse_code(code):
    if code == -1: return 'dqn'
    if code == -2: return 'bdqn'
    if code == -3: return 'ucb'
    if code == 3: return 'adeu_std'
    if code == 1: return 'ez_adeu'

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Your script description here')
    # Define hyperparameters as command-line arguments
    parser.add_argument('--DQN', action='store_true', default=False, help='Use the standard DQN')
    parser.add_argument('--BDQN', action='store_true', default=False, help='Use bootstrap DQN')
    parser.add_argument('--UCB', action='store_true', default=False, help='Use UCB built on top of BDQN')
    parser.add_argument('--adeu_std', action='store_true', default=False, help='Use the ensebmle method for measuring uncertainty')
    parser.add_argument('--ez_adeu', action='store_true', default=False, help='Use the ensebmle method for measuring uncertainty')
    parser.add_argument('--include_pos', action='store_true', default=False, help='Whether position of the agents should be included in the obs')
    parser.add_argument('--per', action='store_true', default=False, help='Would you like to Use the prioritized Experience Replay Buffer')
    parser.add_argument('--dom', type=str, default='ALE/Pong-v5', help='In what domain would you like to the agent to be tested')
    parser.add_argument('--rho', type=float, default=.0, help='what are the maximum timesteps per episode')
    parser.add_argument('--N', type=int, default=50, help='Maximum dimension of the grid')
    args = parser.parse_args()

    # Access hyperparameters in your script
    dqn = args.DQN
    bdqn = args.BDQN
    ucb = args.UCB
    adeu_std = args.adeu_std
    ez_adeu = args.ez_adeu
    N = args.N
    rho = args.rho

    domain = args.dom
    
    agent_args = {
        'dqn': dqn,
        'bdqn': bdqn,
        'ucb': ucb,
        'adeu_std': adeu_std,
        'ez_adeu' : ez_adeu
    }


   
    code = code_creation(agent_args)
    consecutive_fails = 0
    for s in range(7):
        for n in range(51, 100, 3):
            failed = main(n, seed=s, code = code, rho = rho)
            consecutive_fails = consecutive_fails + 1 if failed else 0
            if consecutive_fails > 2:
                print(f'{reverse_code(code)} agent failed to solve deep sea for n = {n - (3*3)}')
                break