import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from td3_agent import TD3, UCB_TD3_Agent, TD3_Ensemble_ADEU
from combined_agent import Pretrained_TD3_Agent
import os
from copy import copy
import gymnasium as gym
import wandb
import argparse
from gym.wrappers import RecordVideo
from tqdm import tqdm
import random
from gymnasium.wrappers import TransformObservation
from network import UNCERTAINTY
from td3_evaluate import evaluate
from utils import make_observation_franz_kitchen
import time
KEEP_TRAJECTORY = False

#defining some global variables
GYMNASIUM_ROBOTIC_DOMAINS =[
    'AntMaze_UMazeDense-v5',
    'PointMaze_LargeDense-v3',
    'FetchSlideDense-v3',
    'FetchSlide-v3',
    'FrankaKitchen-v1'
] 


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





def main(seed, horizon, timesteps, domain, ade, use_rnd, ucb, noisy_nets, comb, mod_rnd, per,include_pos,adeu_ensemble, rho, **kwargs):
    set_seed(seed)
    if KEEP_TRAJECTORY:
        print('In this ran, some statistics will be kept')
        print('If you would like to change it, please set KEEP TRAJECTORY to false')
        print('###################################################################')
        time.sleep(10)
        
    adaptive_exploration = ade
    maze = 'maze' in domain.lower()

    #just for printing
    arguments = {
        'Baseline+ade':comb,
        'UCB': ucb,
        'Adeu+ensemble':adeu_ensemble,
        'ade': ade,
        'rnd':use_rnd,
        'noisy_nets':noisy_nets,
        'Baseline': (not ade) and (not use_rnd) and (not noisy_nets) and (not comb)

    }
    
    
    label = [a for a in arguments.keys() if arguments[a] ][0]
    label = label+" per" if per else label
    label = label + " mod_rnd" if mod_rnd else label
    name = f'{label}, seed:{seed}'
    debug = False
    

    



    env = gym.make(domain, **kwargs)



    seed = seed
    timesteps = timesteps
    observation, info = env.reset(seed=seed)

    eps = rho if rho != -1 else 0
    eps = 0.3 if not adeu_ensemble else eps
    print(f'rho was set to {eps}')
    batch_size = 256
    lr = 5e-5 #it was 3e-4 ,
    uncertainty_const = 1.05
    print(f"actions are limited between [{-env.action_space.high[0]} and {env.action_space.high[0]}]")
    print(f"Shape of an observation: {len(observation)}")
    agent_params = {
        "state_dim": len(observation),
        "action_dim": env.action_space.shape[0],
        "max_action": env.action_space.high[0],
        'seed':seed,
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise":0.2, 
        "noise_clip":0.5,
        "policy_freq":2,
        "lr": lr,
        "train_frequency": 1, #it was 2, 
        "batch_size": batch_size,
        "min_mem_size": batch_size,
        "buffer_size": 1e+5,
        "mod_rnd":mod_rnd,
        "per":per,
        "adaptive_exploration": adaptive_exploration,
        "eps":eps, #it was 0.0005
        "use_rnd": use_rnd,
        "use_noisy_nets": noisy_nets,
        "uncertainty_const" : uncertainty_const
    }
    pretraining = 0.05
    filename = f"./experiments/TD3/new_mujoco_main/{domain}/{label}/seed_{seed}"
    if not os.path.exists(filename+'/stored_data'):
        os.makedirs(filename+'/stored_data')
    if adeu_ensemble:
        lambda_parameter = 1 #it was 1
        N = 5
        agent_params['train_frequency']*=N
        agent = TD3_Ensemble_ADEU(agent_params = agent_params, N = N, lambda_parameter=lambda_parameter, beta = 1)
    elif not ucb:
        lambda_parameter = None
        agent = (TD3(**agent_params) if not comb else Pretrained_TD3_Agent(**agent_params, timesteps = timesteps, horizon = horizon, pretraining = pretraining))
        if use_rnd: agent.set_uncertainty(1)
    else:
        lambda_parameter = 10 #it was 10
        N = 5
        agent_params['train_frequency']*=N
        agent = UCB_TD3_Agent(agent_params = agent_params, N = N, lambda_parameter=lambda_parameter, beta = 1)
    wandb_session(domain = domain, include_pos = include_pos, name=name, lambda_parameter = lambda_parameter, lr = lr, uc = uncertainty_const, rho =rho, code=code_creation(rho = rho,use_rnd = use_rnd, ucb  = ucb, adaptive_exploration=adaptive_exploration, noisy_nets=noisy_nets, comb=comb, mod_rnd=mod_rnd, per = per, adeu_ensemble=adeu_ensemble), debug = debug)
    mean_reward, rewards, mean_evaluation_rewards, evaluation_rewards, evaluation_duration, cum_evaluation_rewards, arg_var_per_epsiode, var_per_episode, total_timesteps = \
        run(kwargs=kwargs,
            maze = maze, 
            domain = domain, 
            env=env, 
            horizon = horizon, 
            agent = agent, 
            timesteps=timesteps, 
            adaptive_exploration = adaptive_exploration, 
            use_rnd=use_rnd, 
            label = label, 
            path = filename, 
            mod_rnd = mod_rnd,
            debug = debug,
            adeu_ensemble = adeu_ensemble,
            seed=seed
            )
    agent_params['horizon']=horizon
    agent_params['timesteps'] = timesteps
    agent_params['networks_uncertainty'] = UNCERTAINTY
    agent_params['pretraining_time'] = pretraining
    agent_params['total_timesteps'] = total_timesteps


    if code == 6: #this is used only to seperate the adeu+ensemble with rho = 0 and rhos = 0.1
        filename+='/rho0/'
        if not os.path.exists(filename+'/stored_data'):
            os.makedirs(filename+'/stored_data')


    store(filename, mean_reward, rewards, mean_evaluation_rewards, evaluation_rewards,  evaluation_duration, cum_evaluation_rewards, arg_var_per_epsiode, var_per_episode, params = agent_params)    
    
    if not comb or ucb: store_network(agent=agent, path = filename)
    env.close()
    return mean_evaluation_rewards





def run(env, domain, horizon, agent:TD3, timesteps, adaptive_exploration,use_rnd ,label, path, mod_rnd, debug, adeu_ensemble, maze,  kwargs, seed):

    
    options = None
   
        
    total_timesteps = 0
    max_timesteps_reached = False
    mean_reward = np.zeros(horizon)
    percentage = 0.01
    mean_evaluation_rewards = np.zeros(int(percentage*horizon)+100)
    cum_evaluation_rewards = np.zeros(int(percentage*horizon)+100)
    evaluation_duration = np.zeros(int(percentage*horizon)+100)
    rewards = np.zeros(horizon*timesteps)
    evaluation_rewards = np.zeros((int(percentage*horizon)+100, timesteps))
    evaluated_episodes = 0
    arg_var_per_episode = np.zeros(horizon) #stores in which timestep of each episode the agent has the biggest uncertainty
    var_per_episode = np.zeros(horizon)
    rnd_loss = 0
    summed_reward = 0
    tmp_network_loss = 0
    network_loss = 0
    mean_var_per_episode = 0 
    agent.rollout_episode = False
    tmp_mean, tmp_r, tmp_duration, tmp_cum = 0,0,0,0

    var = np.zeros(timesteps)
    episode = 0
    t = 0
    total_reward = 0

    for training_step in tqdm(range(horizon)):
        
        

        if training_step % int(horizon*percentage) == 0:
            wandb.log({
                    "timestep with max uncertanty": arg_var_per_episode[training_step -1 if training_step > 0 else 0],
                    "max uncertanty": var_per_episode[training_step -1 if training_step > 0 else 0],
                    'error of the rnd network': rnd_loss,
                    'reward during training': summed_reward /(100//percentage),
                    'Critic network loss': network_loss/(100//percentage),
                    'uncertainty of the agent': agent.rnd_predictor.get_uncertainty() if adaptive_exploration else 0
                    })

            summed_reward = 0
            network_loss = 0
            if mod_rnd:
                max_var = var_per_episode[training_step -1 if training_step > 0 else 0] if t >timesteps/20 else 100
                agent.evaluate_rnd_netowrk(mean_var = mean_var_per_episode/(100//percentage), max_var = max_var)
                mean_var_per_episode = 0
            keep_trajectory = ( seed == 0 and adaptive_exploration and KEEP_TRAJECTORY)
            tmp_mean, tmp_r, tmp_duration, tmp_cum= evaluate(kwargs=kwargs, iterations = int(0.05*horizon), agent=agent, timesteps=timesteps, evaluation_episodes = 30, domain = domain, episode = episode, label = label, debug = debug, options = options, keep_trajectories=keep_trajectory)
            if evaluated_episodes < mean_evaluation_rewards.shape[0]:
                mean_evaluation_rewards[evaluated_episodes], evaluation_rewards[evaluated_episodes], evaluation_duration[evaluated_episodes], cum_evaluation_rewards[evaluated_episodes]  = tmp_mean, tmp_r, tmp_duration, tmp_cum
                evaluated_episodes+=1 
        
        
            current_observation, info = env.reset() #seed is not used as the documentation indicates
            current_observation = torch.FloatTensor(current_observation)
        

        
        
        t+=1
        total_timesteps+=1
        action = agent.get_action(current_observation)# User-defined policy function
        var[t] = agent.get_variance()
        observation, reward, done, truncated, info = env.step(action)
        tmp_network_loss, tmp_rnd_loss = agent.step(current_observation, action, reward, torch.FloatTensor(observation), done or truncated, timestep=t) #push that in memory
        #just to have a good plotting
        rnd_loss = tmp_rnd_loss if (adaptive_exploration or use_rnd) and tmp_rnd_loss > 0 else rnd_loss
        network_loss += tmp_network_loss
        current_observation = torch.FloatTensor(observation)

        #statistics
        total_reward+=reward
        rewards[training_step] = reward
        #wandb.log({domain+"rewards": reward})
        if done or truncated or (t==timesteps-1):
            
            mean_reward[episode] = total_reward/(t+1)
            summed_reward+=total_reward
            total_reward = 0
            

            network_loss/= t #just to plot the average of the network loss. Otherwise it depends on the timesteps that the agent survived
            arg_var_per_episode[episode] = np.argmax(var)
            var_per_episode[episode] = np.max(var)
            mean_var_per_episode+= np.mean(var[:t])#that is a new line
            agent.rollout_episode = False
            episode+=1
            current_observation, info = env.reset()
            current_observation = torch.FloatTensor(current_observation)
            var = np.zeros(timesteps)
            t = 0
            done = False
            truncated = False
            if agent.rollout_episode: 
                    agent.rollout_episode = False
            #check for rollout episode
            if(adaptive_exploration or adeu_ensemble) and (np.random.rand() < agent.eps): 
                agent.rollout_episode = True
                   
            

        if (episode % int(.05 * horizon)) == 0:
            np.save(path + f'/stored_data/uncertainty_of_{episode}th_episode.npy', var)
    env.close()
    wandb.finish()
    return mean_reward, rewards, mean_evaluation_rewards, evaluation_rewards, evaluation_duration, cum_evaluation_rewards, arg_var_per_episode, var_per_episode, total_timesteps

def wandb_session(domain , include_pos, rho, uc=1.5, name = None, code = 0, lr = 0, debug = False, lambda_parameter = -1):
    # start a new wandb run to track this script
    project = f"AAAI-2026| TD3, Adaptive Exploration {domain}" if not include_pos \
            else f"AAAI-2026| TD3, Adaptive Exploration {domain}, including position" 
    project = "debug" if debug else project
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        name = name,
        # track hyperparameters and run metadata
        config={
        "code":code,
        "lr":lr,
        "unc_cost": uc,
        'rho':rho,
        'lambda':lambda_parameter,
        }
    )
    

def store_network(agent:TD3, path):
      if not os.path.exists(path): os.makedirs(path)
      agent.save(path)


def store(path, mean_reward, rewards, mean_evaluation_rewards,evaluation_rewards,  evaluation_duration, cum_evaluation_rewards, arg_var_per_epsiode, var_per_episode, params):
    
    np.save(path+'/stored_data/mean_reward.npy', mean_reward)
    np.save(path+'/stored_data/rewards.npy', rewards)
    np.save(path+'/stored_data/mean_evaluation_rewards.npy', mean_evaluation_rewards)
    np.save(path+'/stored_data/evaluation_rewards.npy', evaluation_rewards)
    np.save(path+'/stored_data/evaluation_duration.npy', evaluation_duration)
    np.save(path+'/stored_data/cum_evaluation_rewards.npy', cum_evaluation_rewards)
    np.save(path+'/stored_data/timestep_with_maximum_varience.npy', arg_var_per_epsiode)
    np.save(path+'/stored_data/maximum_varience', var_per_episode)
    
    full_path = path + "/hyperparameters.txt"

    with open(full_path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    return
def code_creation(rho, adaptive_exploration=True, use_rnd = False, noisy_nets = False, comb = False, mod_rnd = False, per = False, ucb = False, adeu_ensemble = False):
    retval = 20 if mod_rnd else 0
    retval+= 100 if per else 0
    
    #if mod_rnd and noisy_nets: sys.exit(-1)
    if per and use_rnd: sys.exit(-1)
    if comb:
        if not adaptive_exploration: sys.exit(-1)
        return retval + 4
    if noisy_nets:return retval + 3
    if adaptive_exploration: return retval + 1
    if use_rnd: return retval + 2
    if ucb: return retval + 5
    if adeu_ensemble: 
        if rho != -1:
            return retval + 7
        return retval + 6
    if mod_rnd: sys.exit(-1)
    return retval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Your script description here')
    # Define hyperparameters as command-line arguments
    parser.add_argument('--ade', action='store_true', default=True, help='Would you like to use Adaptive Exploration')
    parser.add_argument('--rnd', action='store_true', default=False, help='Would you like to use the standrad RND')
    parser.add_argument('--noisy_nets', action='store_true', default=False)
    parser.add_argument('--comb', action='store_true', default=False, help='Would you like to utilize both Bolztmann and ade')
    parser.add_argument('--mod_rnd', action='store_true', default=False, help='Heuristically modify the Uncertainty')
    parser.add_argument('--UCB', action='store_true', default=False, help='Create a UCB agent')
    parser.add_argument('--adeu_ensemble', action='store_true', default=False, help='Use the ensebmle method for measuring uncertainty')
    parser.add_argument('--include_pos', action='store_true', default=False, help='Whether position of the agents should be included in the obs')
    parser.add_argument('--per', action='store_true', default=False, help='Would you like to Use the prioritized Experience Replay Buffer')
    parser.add_argument('--dom', type=str, default='Hopper-v5', help='In what domain would you like to the agent to be tested')
    parser.add_argument('--steps', type=int, default=1000, help='what are the maximum timesteps per episode')
    parser.add_argument('--rho', type=float, default=-1.0, help='what are the maximum timesteps per episode')
   
    args = parser.parse_args()

    # Access hyperparameters in your script
    use_rnd = args.rnd
    domain = args.dom
    timesteps = args.steps
    ade = args.ade
    max_episode_steps = timesteps
    noisy_nets = args.noisy_nets
    comb = args.comb
    mod_rnd = args.mod_rnd
    per = args.per
    ucb = args.UCB
    rho = args.rho
    adeu_ensemble = args.adeu_ensemble
    include_pos = args.include_pos

    horizon = {
        'Hopper-v5':10e6,
        'Hopper':10e+6,
        'Humanoid-v5':80e6,
        'Walker2d-v5':10e6,
        'Walker2d':10e6,
        'HalfCheetah-v5':10e6,
        'Pusher-v5':5e6,
        'Reacher-v5':5e6,
        'InvertedDoublePendulum-v5':30e6,
        'InvertedPendulum-v5':10e6,
        'Ant-v5':5e6,
        'HumanoidStandup-v5':3.5e6,#it was 5 in neurips
        'Swimmer-v5':5e6,
        ################
        #for gymnasium robotics
        'AntMaze_UMazeDense-v5':5e6,
        'PointMaze_LargeDense-v3':2e6,
        'FetchSlideDense-v3':5e6,
        'FetchSlide-v3':5e6,
        'FrankaKitchen-v1':5e6,
        ######################
        'MountainCarContinuous-v0':5e6
    }

    horizon = int(1.2e6)
    

        

    seeds = 20
    
    mean_evaluation_rewards = []
    code = code_creation(adaptive_exploration=ade, 
                                           ucb = ucb, 
                                           use_rnd=use_rnd, 
                                           noisy_nets=noisy_nets, 
                                           comb=comb, 
                                           per=per, 
                                           mod_rnd=mod_rnd, 
                                           adeu_ensemble=adeu_ensemble,
                                           rho = rho)
    print(f"{domain} + code:{code}")

    for s in range( seeds):
        main(seed = s, 
             horizon=horizon, 
             timesteps=timesteps, 
             ucb = ucb, 
             domain=domain, 
             ade = ade, 
             use_rnd=use_rnd, 
             noisy_nets=noisy_nets, 
             comb =comb, 
             per=per, 
             mod_rnd = mod_rnd,
             include_pos = include_pos,
             adeu_ensemble = adeu_ensemble,
             rho = rho)
             