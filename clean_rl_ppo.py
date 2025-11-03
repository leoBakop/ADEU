
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import random
import time
import ale_py
import gymnasium as gym
import numpy as np
import torch
import wandb
from atari import wandb_session

from atari_env import AtariEnv
from ppo_agents import PPO_Agent
from atari_evaluate import evaluate
from tqdm import tqdm

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        gym.register_envs(ale_py)
        

        
        #we are setting the frameskip equal to 1 (so in practice no frameskip). We then apply it using the Wrapper
        env = gym.make(env_id, repeat_action_probability = 0.25, frameskip = 1,  max_episode_steps=10_000,)# render_mode = 'human')
        env = AtariEnv(env, frame_stacking=4, max_no_op_frames=30)
        obs, _ = env.reset()
        env.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32
        )
        
        
       
        return env

    return thunk


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(seed):

    set_seed(seed)

    agent_params = {
        # Algorithm specific arguments
        "env_id": "ALE/Pong-v5",  # the id of the environment
        "total_timesteps": int(20e6),  # total timesteps of the experiments
        "learning_rate": 2.5e-4,  # the learning rate of the optimizer
        "num_envs": 8,  # the number of parallel game environments
        "num_steps": 128,  # the number of steps to run in each environment per policy rollout
        "anneal_lr": True,  # Toggle learning rate annealing for policy and value networks
        "gamma": 0.99,  # the discount factor gamma
        "gae_lambda": 0.95,  # the lambda for the general advantage estimation
        "num_minibatches": 4,  # the number of mini-batches
        "update_epochs": 4,  # the K epochs to update the policy
        "norm_adv": True,  # Toggles advantages normalization
        "clip_coef": 0.1,  # the surrogate clipping coefficient
        "clip_vloss": True,  # Toggles whether or not to use a clipped loss for the value function
        "ent_coef": 0.01,  # coefficient of the entropy
        "vf_coef": 0.5,  # coefficient of the value function
        "max_grad_norm": 0.5,  # the maximum norm for the gradient clipping
        "target_kl": None,  # the target KL divergence threshold
        'head_start':50_000,
    }



    agent_params ['batch_size'] = int(agent_params['num_envs'] * agent_params['num_steps'])
    agent_params["minibatch_size"] = int(agent_params['batch_size'] // agent_params['num_minibatches'])
    agent_params['num_iterations'] = agent_params['total_timesteps'] // agent_params['batch_size']
    run_name = f"PPO, seed {seed}"

    wandb_session(
        domain = agent_params['env_id'],
        rho = None,
        lr = agent_params['learning_rate'],
        name = run_name,
        code = -4,
        debug=False 
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(agent_params['env_id'], i,capture_video=False, run_name=run_name) for i in range(agent_params['num_envs'])],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = PPO_Agent(envs, **agent_params, device=device).to(device)
    

    # ALGO Logic: Storage setup
    obs = torch.zeros((agent_params['num_steps'], agent_params['num_envs']) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((agent_params['num_steps'], agent_params['num_envs']) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((agent_params['num_steps'], agent_params['num_envs'])).to(device)
    rewards = torch.zeros((agent_params['num_steps'], agent_params['num_envs'])).to(device)
    dones = torch.zeros((agent_params['num_steps'], agent_params['num_envs'])).to(device)
    values = torch.zeros((agent_params['num_steps'], agent_params['num_envs'])).to(device)

    cumm_eval_rewards = []

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(agent_params['num_envs']).to(device)
    evaluation_every = int(agent_params['num_iterations']//100)

    rewardss = 0
    loss = 0
    noop = 0
    episodes_till_evaluate = 1

    for iteration in tqdm(range(agent_params['num_iterations'])):
        # Annealing the rate if instructed to do so.
        agent.reduce_lr(iteration)
        episodes_till_evaluate+=1

        if iteration%evaluation_every == 0:
            cumulative_evaluation_reward = evaluate(agent, agent_params['env_id'], timesteps = 10_000, seed = seed, evaluation_episodes=1)
           
            variance, q_mean = agent.get_variance()
            wandb.log({ 'Cumulative episodic reward':cumulative_evaluation_reward,
                        'mean episodic loss during training': loss/episodes_till_evaluate,
                        'mean episodic training reward': rewardss/episodes_till_evaluate,
                        'frequency of noop (episodic)': noop/episodes_till_evaluate,
                        'mean variance at every timestep' : variance,
                        'variance over greediness': variance/q_mean if q_mean!=0 else -1
                        })
            episodes_till_evaluate = 0
            cumm_eval_rewards.append(cumulative_evaluation_reward)
            rewardss = 0
            loss = 0
            noop = 0

        for step in range(agent_params['num_steps']):

            global_step += agent_params['num_envs']
            obs[step] = next_obs 
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            rewardss += reward
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            step+=1
            

        loss += agent.train(obs, actions, rewards, logprobs, dones, values, next_obs, next_done)
            

    envs.close()





if __name__ == "__main__":

    seeds = 3
    for s in range(seeds):
        main(s)
    
