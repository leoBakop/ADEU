import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agent import Agent
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=  True





def main(seed, horizon, timesteps, domain, per, psne, **kwargs):
    wandb_session(domain, seed, name=f'naf_{domain}_per_{per}_psne_{psne}_seed_{seed}', code=code_creation(per = per, psne = psne))
    env = gym.make(domain, **kwargs)
    eval_env = gym.make(domain, render_mode = 'rgb_array', **kwargs)
    eval_env = RecordVideo(eval_env, video_folder = f'./videos/{domain}', episode_trigger= lambda x:  x % 1000 == 9, name_prefix=f'{domain}_')
    c = 1
    I = 1
    terminate_psne = int(0.5*horizon)
    head_start_psne = int(0.11*horizon)
    
    sigma = 0

    seed = seed
    timesteps = timesteps
    observation, info = env.reset(seed=seed)
    if isinstance(observation, dict):
            observation = np.concatenate([arr for arr in observation.values()])
    eval_env.reset(seed=seed)

    #horizon and network
    inside_layer_1 = 256
    inside_layer_2= 256
    batch_size = 64
    cut_training = -1
    horizon = horizon
    end_training = horizon 
    print(f"actions are limited between [{-env.action_space.high[0]} and {env.action_space.high[0]}]")
    

    agent = Agent( len(observation),
                    hidden_layer_1=inside_layer_1,
                    hidden_layer_2=inside_layer_2,
                    use_priorized_exp_replay= per,
                    batch_size=batch_size,
                    update_target_every= I,
                    max_speed=env.action_space.high[0],
                    gamma= .99,
                    c=c,
                    min_mem_size= batch_size,
                    learning_rate= 1e-3, #robotics is with e-3 cheetah with e-4
                    tau=.001, #it was 0.005
                    action_size=env.action_space.shape[0],
                    path = None,
                    loaded_mem = None,
                    probabilities= None,
                    index = 0,
                    #for per
                    alpha = 0.7,
                    beta = 0.5,
                    #for the parameter space exploration
                    seed = seed,
                    cut_training = cut_training,
                    end_training = end_training,
                    evaluation_c=0,
                    train_frequency=1, #2 dor all in general except cheetha now I am trying for hopper #it was 8 for halg cheetah,
                    adaptive_eploration=True
                    )
    mean_reward, rewards, mean_evaluation_rewards, evaluation_rewards, evaluation_duration, cum_evaluation_rewards = \
        run(**kwargs, seed = seed, domain = domain, env=env, horizon = horizon, agent = agent, timesteps=timesteps, per = per, psne = psne, terminate_psne = terminate_psne, eval_env = eval_env, head_start_psne=head_start_psne) #the headstart was 200
    store(f"./experiments/{domain}/per_{per}_psne_{psne}/seed_{seed}", mean_reward, rewards, mean_evaluation_rewards, evaluation_rewards,  evaluation_duration, cum_evaluation_rewards)    
    env.close()
    return mean_evaluation_rewards


def run(env, eval_env, domain,  seed, horizon = 10_000, agent:Agent = None, timesteps = 50, per=False, psne = False, terminate_psne = 4_000, head_start_psne = 200, **kwargs):
    mean_reward = np.zeros(horizon)
    percentage = 1
    mean_evaluation_rewards = np.zeros(int(percentage*horizon/100)+100)
    cum_evaluation_rewards = np.zeros(int(percentage*horizon/100)+100)
    evaluation_duration = np.zeros(int(percentage*horizon/100)+100)
    rewards = np.zeros(horizon*timesteps)
    evaluation_rewards = np.zeros((int(percentage*horizon/100)+100, timesteps))
    evaluated_episodes = 0
    first_time_psne = psne
    if per: b = agent.beta
    if psne: 
        eps_psne = np.linspace(0.5,0.05, terminate_psne)
    

    for episode in tqdm(range(horizon)):

        if psne:
            wandb.log({domain+"_agents_sigma": agent.sigma})
            if episode < head_start_psne and first_time_psne: 
                tradeoff = 0
            else:
                if first_time_psne:
                    print("---------------------------PSNE just started -----------------------") 
                    first_time_psne = False
                tradeoff = eps_psne[episode - head_start_psne] if episode < terminate_psne+ head_start_psne else eps_psne[-1]

            agent.update_psne_tradeoff(tradeoff)
            agent.add_noise_to_nn()

        if episode % int(round(100/percentage)) == 0:
            tmp_mean, tmp_r, tmp_duration, tmp_cum= evaluate(**kwargs, iterations = int(0.05*horizon), agent=agent, timesteps=timesteps, evaluation_episodes = 3, domain = domain, seed = seed, episode = episode, per=per, psne=psne)
            if evaluated_episodes < mean_evaluation_rewards.shape[0]:
                mean_evaluation_rewards[evaluated_episodes], evaluation_rewards[evaluated_episodes], evaluation_duration[evaluated_episodes], cum_evaluation_rewards[evaluated_episodes]  = tmp_mean, tmp_r, tmp_duration, tmp_cum
                evaluated_episodes+=1 
        current_observation, info = env.reset() #seed is not used as the documentation indicates
        if isinstance(current_observation, dict):
            current_observation = np.concatenate([arr for arr in current_observation.values()])
        current_observation = torch.FloatTensor(current_observation)
        total_reward = 0
        loss = 0
        for t in range(timesteps):

            if per:
                dt=10e-6 if t==0 else 0
                fraction = min((episode*t+t) / (horizon*t+dt), 1.0)
                b = b + fraction * (1.0 - b)
                agent.set_beta(beta=b)
        
            action = agent.get_action(current_observation)# User-defined policy function
            action=np.array(action.cpu().squeeze().numpy().reshape((agent.action_size,)))
            observation, reward, done, truncated, info = env.step(action)
            if isinstance(observation, dict):
                observation = np.concatenate([arr for arr in observation.values()])
            loss += agent.step(current_observation, action, reward, torch.FloatTensor(observation), done, episode, timestep=t) #push that in memory
            current_observation = torch.FloatTensor(observation)

            #statistics
            total_reward+=reward
            rewards[episode*timesteps + t] = reward
            #wandb.log({domain+"rewards": reward})
            if done or truncated:
                mean_reward[episode] = total_reward/(t+1)
                #wandb.log({domain+"mean_reward": total_reward/(t+1), "loss": loss/(t+1)})
                break
    env.close()
    eval_env.close()
    wandb.finish()
    return mean_reward, rewards, mean_evaluation_rewards, evaluation_rewards, evaluation_duration, cum_evaluation_rewards

def wandb_session(domain , horizon = None, seed = 0, name = None, code = 0):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"just_checking {domain}",
        name = name,
        # track hyperparameters and run metadata
        config={
        "domain": domain,
        "seed": seed,
        "code":code
        }
    )
    
def evaluate(agent:Agent, timesteps, evaluation_episodes, domain, seed, episode, per, psne, iterations,**kwargs):
    env = make_evaluation_env(dom=domain, iterations=iterations, episode = episode, per=per, psne=psne, **kwargs)
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
    wandb.log({domain+"mean_evaluation_reward": np.mean(mean_reward), 'duration': np.mean(mean_timesteps), 'cum_evaluation_reward': np.mean(mean_sum_reward)})
    env.close()
    return np.mean(mean_reward), average_rewards, np.mean(mean_timesteps), np.mean(mean_sum_reward)

def make_evaluation_env(dom, iterations, episode, per, psne, **kwargs):
    eval_env = gym.make(dom, render_mode = 'rgb_array', **kwargs)
    if episode % iterations != 0: return eval_env 
    eval_env = RecordVideo(eval_env, video_folder = f'./videos/{dom}_train_freq_1/_per_{per}_psne_{psne}', episode_trigger= lambda x:  x % iterations == 0, name_prefix=f'{dom}_true_epside_{episode}')
    return eval_env

def store(path, mean_reward, rewards, mean_evaluation_rewards,evaluation_rewards,  evaluation_duration, cum_evaluation_rewards):
    if not os.path.exists(path+'/stored_data'):
        os.makedirs(path+'/stored_data')
    np.save(path+'/stored_data/mean_reward.npy', mean_reward)
    np.save(path+'/stored_data/rewards.npy', rewards)
    np.save(path+'/stored_data/mean_evaluation_rewards.npy', mean_evaluation_rewards)
    np.save(path+'/stored_data/evaluation_rewards.npy', evaluation_rewards)
    np.save(path+'/stored_data/evaluation_duration.npy', evaluation_duration)
    np.save(path+'/stored_data/cum_evaluation_rewards.npy', cum_evaluation_rewards)

def code_creation(per=True, psne=True):
    if per and psne: return 3
    if not per and psne : return 2
    if not per and not psne: return 1
    return -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Your script description here')
    # Define hyperparameters as command-line arguments
    parser.add_argument('--per', action='store_true', default=False, help='Would you like to use PER')
    parser.add_argument('--psne', action='store_true', default=False, help='Would you like to use PSNE')
    parser.add_argument('--dom', type=str, default='Hopper-v4', help='In what domain would you like to the agent to be tested')
    parser.add_argument('--steps', type=int, default=1000, help='what are the maximum timesteps per episode')
    #parser.add_argument('--max_episode_steps', type=int, default = 100)
    args = parser.parse_args()

    # Access hyperparameters in your script
    per = args.per
    psne = args.psne
    domain = args.dom
    timesteps = args.steps
    max_episode_steps = timesteps
    horizon = int(5_000_000 / timesteps)
    if timesteps < 200:
        horizon = int(horizon/10)

        

    seeds = 3
    
    mean_evaluation_rewards = []

    print(f"{domain}+per_{per}+psne_{psne}")

    for s in range(2,seeds):
        main(seed = s, horizon=horizon, timesteps=timesteps, domain=domain, per=per, psne=psne)