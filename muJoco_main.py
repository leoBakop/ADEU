import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from agent import Agent
from combined_agent import Pretrained_Agent
import pickle
import os
import datetime
from copy import copy
import gymnasium as gym
import wandb
import plotly.graph_objects as go
import argparse
from gym.wrappers import RecordVideo
from tqdm import tqdm
import random
from gymnasium.wrappers import TransformObservation



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





def main(seed, horizon, timesteps, domain, ade, use_rnd, noisy_nets, comb, **kwargs):
    set_seed(seed)
    adaptive_exploration = ade
    #kwargs={
    #    'max_episode_steps':timesteps,
    #}

    #just for printing
    arguments = {
        'Boltz+ade':comb,
        'ade': ade,
        'rnd':use_rnd,
        'noisy_nets':noisy_nets,
        'Bolz': (not ade) and (not use_rnd) and (not noisy_nets) and (not comb)
    }
    label = [a for a in arguments.keys() if arguments[a] ][0]
    name = f'naf, using {label}, seed:{seed}'
    wandb_session(domain, seed, name=name, code=code_creation(use_rnd = use_rnd, adaptive_exploration=adaptive_exploration, noisy_nets=noisy_nets, comb=comb))
    
    
    
    env = gym.make(domain, **kwargs)
    eval_env = gym.make(domain, render_mode = 'rgb_array', **kwargs)
    eval_env = RecordVideo(eval_env, video_folder = f'./videos/{domain}', episode_trigger= lambda x:  x % 1000 == 9, name_prefix=f'{domain}_')
    
    c = 1
    I = 1
    

    seed = seed
    timesteps = timesteps
    observation, info = env.reset(seed=seed)
    if isinstance(observation, dict):
            observation = np.concatenate([arr for arr in observation.values()])
    eval_env.reset(seed=seed)

    #horizon and network
    inside_layer_1 = 64
    inside_layer_2= 64
    batch_size = 64
    cut_training = -1
    horizon = horizon
    end_training = horizon 
    print(f"actions are limited between [{-env.action_space.high[0]} and {env.action_space.high[0]}]")

    agent_params = {
    "inputs_shape": len(observation),
    "hidden_layer_1": inside_layer_1,
    "hidden_layer_2": inside_layer_2,
    "use_priorized_exp_replay": False,
    "batch_size": batch_size,
    "update_target_every": I,
    "max_speed": env.action_space.high[0],
    "gamma": 0.99,
    "c": c,
    "min_mem_size": batch_size,
    "learning_rate": 1e-3,  # robotics is with e-3, cheetah with e-4
    "tau": 0.001,  # it was 0.005
    "action_size": env.action_space.shape[0],
    "path": None,
    "loaded_mem": None,
    "probabilities": None,
    "index": 0,
    "alpha": 0.7,  # for PER
    "beta": 0.5,   # for PER
    "seed": seed,  # for parameter space exploration
    "cut_training": cut_training,
    "end_training": end_training,
    "evaluation_c": 0,
    "train_frequency": 8,  # For all other domains it was 1, for cheetah it is 8
    "adaptive_exploration": adaptive_exploration,
    "decay": int(0.05 * horizon * timesteps),
    "use_rnd": use_rnd,
    "use_noisy_nets": noisy_nets
    }

    agent = (Agent(**agent_params) if not comb else Pretrained_Agent(**agent_params, timesteps = timesteps, horizon = horizon, pretraining = 0.03))

    mean_reward, rewards, mean_evaluation_rewards, evaluation_rewards, evaluation_duration, cum_evaluation_rewards, arg_var_per_epsiode, var_per_episode = \
        run(**kwargs, domain = domain, env=env, horizon = horizon, agent = agent, timesteps=timesteps, eval_env = eval_env, adaptive_exploration = adaptive_exploration, use_rnd=use_rnd, label = label)
    agent_params['horizon']=horizon
    agent_params['timesteps'] = timesteps
    store(f"./experiments/{domain}/{label}/seed_{seed}", mean_reward, rewards, mean_evaluation_rewards, evaluation_rewards,  evaluation_duration, cum_evaluation_rewards, arg_var_per_epsiode, var_per_episode, params = agent_params)    
    env.close()
    return mean_evaluation_rewards


def run(env, eval_env, domain, horizon, agent:Agent, timesteps, adaptive_exploration,use_rnd ,label, **kwargs):
    mean_reward = np.zeros(horizon)
    percentage = 1
    mean_evaluation_rewards = np.zeros(int(percentage*horizon/100)+100)
    cum_evaluation_rewards = np.zeros(int(percentage*horizon/100)+100)
    evaluation_duration = np.zeros(int(percentage*horizon/100)+100)
    rewards = np.zeros(horizon*timesteps)
    evaluation_rewards = np.zeros((int(percentage*horizon/100)+100, timesteps))
    evaluated_episodes = 0
    arg_var_per_episode = np.zeros(horizon) #stores in which timestep of each episode the agent has the biggest uncertainty
    var_per_episode = np.zeros(horizon)
    rnd_loss = 0

    

    for episode in tqdm(range(horizon)):
        var = np.zeros(timesteps)
        if episode % int(round(100/percentage)) == 0:
            wandb.log({"timestep with max uncertanty": arg_var_per_episode[episode -1 if episode > 0 else 0]})
            wandb.log({"max uncertanty": var_per_episode[episode -1 if episode > 0 else 0]})
            wandb.log({'error of the rnd network': rnd_loss})
            tmp_mean, tmp_r, tmp_duration, tmp_cum= evaluate(**kwargs, iterations = int(0.05*horizon), agent=agent, timesteps=timesteps, evaluation_episodes = 3, domain = domain, episode = episode, label = label)
            if evaluated_episodes < mean_evaluation_rewards.shape[0]:
                mean_evaluation_rewards[evaluated_episodes], evaluation_rewards[evaluated_episodes], evaluation_duration[evaluated_episodes], cum_evaluation_rewards[evaluated_episodes]  = tmp_mean, tmp_r, tmp_duration, tmp_cum
                evaluated_episodes+=1 
        current_observation, info = env.reset() #seed is not used as the documentation indicates
        if isinstance(current_observation, dict):
            current_observation = np.concatenate([arr for arr in current_observation.values()])
        current_observation = torch.FloatTensor(current_observation)
        total_reward = 0
        
        for t in range(timesteps):
        
            action = agent.get_action(current_observation)# User-defined policy function
            var[t] = agent.get_variance()
            action=np.array(action.cpu().squeeze().numpy().reshape((agent.action_size,)))
            observation, reward, done, truncated, info = env.step(action)
            if isinstance(observation, dict):
                observation = np.concatenate([arr for arr in observation.values()])
            _, tmp_rnd_loss = agent.step(current_observation, action, reward, torch.FloatTensor(observation), done, episode, timestep=t) #push that in memory
            #just to have a good plotting
            rnd_loss = tmp_rnd_loss if (adaptive_exploration or use_rnd) and tmp_rnd_loss > 0 else rnd_loss
            current_observation = torch.FloatTensor(observation)

            #statistics
            total_reward+=reward
            rewards[episode*timesteps + t] = reward
            #wandb.log({domain+"rewards": reward})
            if done or truncated:
                mean_reward[episode] = total_reward/(t+1)
                #wandb.log({domain+"mean_reward": total_reward/(t+1), "loss": loss/(t+1)})
                break
        arg_var_per_episode[episode] = np.argmax(var)
        var_per_episode[episode] = np.max(var)
    env.close()
    eval_env.close()
    wandb.finish()
    return mean_reward, rewards, mean_evaluation_rewards, evaluation_rewards, evaluation_duration, cum_evaluation_rewards, arg_var_per_episode, var_per_episode

def wandb_session(domain , seed = 0, name = None, code = 0):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"AAAI| NAF, Adaptive Exploration {domain}",
        name = name,
        # track hyperparameters and run metadata
        config={
        "domain": domain,
        "seed": seed,
        "code":code
        }
    )
    
def evaluate(agent:Agent, timesteps, evaluation_episodes, domain, label, episode, iterations,**kwargs):
    env = make_evaluation_env(dom=domain, iterations=iterations, episode = episode, label = label, **kwargs)
    mean_reward = np.zeros(evaluation_episodes)
    mean_timesteps = np.zeros(evaluation_episodes)
    mean_sum_reward = np.zeros(evaluation_episodes)

    average_rewards = np.zeros(timesteps)
    for episode in range(evaluation_episodes):
        current_observation, info = env.reset()
        if isinstance(current_observation, dict):
            current_observation = np.concatenate([arr for arr in current_observation.values()])

        current_observation = torch.FloatTensor(current_observation)
        total_reward = 0
        for t in range(timesteps):
            action = agent.get_clear_action(current_observation)# User-defined policy function
            action=np.array(action.cpu().squeeze().numpy().reshape((agent.action_size,)))
            observation, reward, done, truncated, info = env.step(action) 
            if isinstance(observation, dict):
                observation = np.concatenate([arr for arr in observation.values()])
            current_observation = torch.FloatTensor(observation)
            total_reward += reward
            average_rewards[t] += reward/evaluation_episodes
            #if episode == 0:
                #wandb.log({domain+'evaluation_rewards':reward})
            if done or truncated:
                mean_timesteps[episode] = t+1
                mean_reward[episode] = total_reward/(t+1)
                mean_sum_reward[episode] = total_reward
                break
    wandb.log({domain+"mean_evaluation_reward": np.mean(mean_reward), 'duration': np.mean(mean_timesteps), 'cumm_evaluation_reward': np.mean(mean_sum_reward)})
    env.close()
    return np.mean(mean_reward), average_rewards, np.mean(mean_timesteps), np.mean(mean_sum_reward)

def make_evaluation_env(dom, iterations, episode, label, **kwargs):
    eval_env = gym.make(dom, render_mode = 'rgb_array', **kwargs)
    if episode % iterations != 0: return eval_env 
    eval_env = RecordVideo(eval_env, video_folder = f'./videos/final_experiments/naf/{dom}/{label}/', episode_trigger= lambda x:  x % iterations == 0, name_prefix=f'{dom}_true_epside_{episode}')
    return eval_env

def store(path, mean_reward, rewards, mean_evaluation_rewards,evaluation_rewards,  evaluation_duration, cum_evaluation_rewards, arg_var_per_epsiode, var_per_episode, params):
    if not os.path.exists(path+'/stored_data'):
        os.makedirs(path+'/stored_data')
    np.save(path+'/stored_data/mean_reward.npy', mean_reward)
    np.save(path+'/stored_data/rewards.npy', rewards)
    np.save(path+'/stored_data/mean_evaluation_rewards.npy', mean_evaluation_rewards)
    np.save(path+'/stored_data/evaluation_rewards.npy', evaluation_rewards)
    np.save(path+'/stored_data/evaluation_duration.npy', evaluation_duration)
    np.save(path+'/stored_data/cum_evaluation_rewards.npy', cum_evaluation_rewards)
    np.save(path+'/stored_data/timestep_with_maximum_varience.npy', arg_var_per_epsiode)
    np.save(path+'/stored_data/maximum_varience', var_per_episode)
    
    full_path = f"{path}+/hyperparameters.txt"
    
    with open(full_path, 'wb') as f:
        pickle.dump(params, f)

    return
def code_creation(adaptive_exploration=True, use_rnd = False, noisy_nets = False, comb = False):
    if comb:
        if not adaptive_exploration: sys.exit(-1)
        return 4
    if noisy_nets:return 3
    if adaptive_exploration: return 1
    if use_rnd: return 2
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Your script description here')
    # Define hyperparameters as command-line arguments
    parser.add_argument('--ade', action='store_true', default=False, help='Would you like to use Adaptive Exploration')
    parser.add_argument('--rnd', action='store_true', default=False, help='Would you like to use the standrad RND')
    parser.add_argument('--noisy_nets', action='store_true', default=False)
    parser.add_argument('--comb', action='store_true', default=False, help='Would you like to utilize both Bolztmann and ade')
    parser.add_argument('--dom', type=str, default='Hopper-v4', help='In what domain would you like to the agent to be tested')
    parser.add_argument('--steps', type=int, default=1000, help='what are the maximum timesteps per episode')
   
    args = parser.parse_args()

    # Access hyperparameters in your script
    use_rnd = args.rnd
    domain = args.dom
    timesteps = args.steps
    ade = args.ade
    max_episode_steps = timesteps
    noisy_nets = args.noisy_nets
    comb = args.comb
    horizon = int(40_000_000 / timesteps) #hopeer:100M  fetch reach 15M, cheetah 7M
    if timesteps <= 200:
        horizon = int(horizon/10)

        

    seeds = 1
    
    mean_evaluation_rewards = []

    print(f"{domain}+ code:{code_creation(adaptive_exploration=ade, use_rnd=use_rnd, noisy_nets=noisy_nets, comb=comb)}")

    for s in range(seeds):
        main(seed = s, horizon=horizon, timesteps=timesteps, domain=domain, ade = ade, use_rnd=use_rnd, noisy_nets=noisy_nets, comb =comb)