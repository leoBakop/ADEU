import gymnasium as gym
import numpy as np
from q_learning_agent import Q_Learning_Agent, IR_Q_Learning_Agent, UCB_Q_Learning, ADEU_Q_Learning
from tabular_evaluate import evaluate
from tqdm import tqdm
import argparse
import wandb
import os
import pickle
from tabular_main import code_creation, set_seed, wandb_session, reverse_hash
import random


def generate_custom_map(dim, initial_epsilon = 0.5):
    desc = []
    conflict_lines = []
    epsilon = initial_epsilon
    initial_line = "SF" + "".join(["H" for _ in range(dim - 2)])
    desc.append(initial_line)
    #constructing the vertical path
    for i in range(4, dim): 
        if np.random.rand() < epsilon:
            conflict_lines.append(i-3)
            desc.append("HF" + "FFFFFFFFFF" + "".join(["H" for _ in range(dim - 12)]))   #outputs one H, 10+1 F
            epsilon = epsilon*2 if epsilon*1.5  <0.5 else 0.5 
        else:
            epsilon = initial_epsilon
            desc.append("HF" + "".join(["H" for _ in range(dim - 2)]))  
    line_before_before= ("HF" + "".join(["F" for _ in range(dim - 2)]))  
    line_before= ("HF" + "".join(["F" for _ in range(dim - 2)]))     
    #cosntructing the horizontal path
    desc.append(line_before_before)
    desc.append(line_before)
    final_line = "".join(["H" for _ in range(dim - 1)])+"G"
    desc.append(final_line)
    return desc, conflict_lines


def load_background_knowledge(path):
    with open(path, "rb") as file:
        loaded_experience = pickle.load(file)
    return loaded_experience

def filter_states(background_knowledge, conflicts, samples = 1500*2):
    """ this method returns the tuples in which an agent should be trained on.
        since the background knowledge contains only the correcttrajectory---and some initial training data
        I will just sample some random.     
    """
    sampled = random.sample(background_knowledge, k=samples)
    return sampled

def pretraining(agent, background_knowledge, code_of_the_agent, conflicts, iterations = 10):
    """ 
        this method pretrains the agent on the background knowledge.
        this process is repeated multiple times
    """

    for i in tqdm(range(iterations)):
        random.shuffle(background_knowledge)
        for current_observation, action, reward, new_observation, done in background_knowledge:
            if code_of_the_agent == 5: #in the case of a ucb agent
                mask = np.random.binomial(1, agent.p, agent.num_of_agents)
                agent.step([current_observation, action, reward, new_observation, done, mask])
            else:
                agent.step([current_observation, action, reward, new_observation, done])
    print('the agent pretrained succesfully')
    return agent

def run(env:gym.Env, agent:Q_Learning_Agent|ADEU_Q_Learning, code, domain, horizon = 1_000, timesteps = 1000, render = False, label = 0, desc = None, dim = 50, tofl = 0):
    conflicts = 0
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
            evaluate(agent, 
                    timesteps, 
                    domain = domain, 
                    evaluation_episodes = 3, 
                    label = label, 
                    episode = episode, 
                    iterations = int(0.2*horizon), 
                    desc = desc, 
                    normalizer=normalizer, 
                    keep_trajectories=False,
                    save_experience=False)

            wandb.log({"std error during training": mean_std})
            wandb.log({"mean conflicts per episode": conflicts/(100/percentage)})
            conflicts = 0
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
                #in our case this migh also be a forbidden array.
                #so we have to check it out
                if col != 1 and row < dim-2:
                    conflicts+= 1
                    
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







def main(seed, code, label, heads):
    set_seed(seed=seed)
    dim = 1500
    desc, conflicts = generate_custom_map(dim=dim, initial_epsilon=0.1)
    beta = 0
    name = f'{label}, seed:{seed}'
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
    
    domain ="SafetyFrozenLake"
    wandb_session(domain = domain, seed = seed, debug = False, code = code, name = name, beta = beta, heads = heads)
    #since we have already define everything lets try to load the background knowledge
    background_knowledge = load_background_knowledge(path = './experiments/tabular/Safety_CustomFrozenLake/background_experienc.pkl')
    #now lets train the agent in a some speciffic states
    sampled_experience = filter_states(background_knowledge, conflicts, samples = dim**2)
    #then I will train the agent in the `sampled experience`
    agent = pretraining(agent, sampled_experience, code_of_the_agent=code)

    #we free both the background knowledge and the sampled experience
    del background_knowledge
    del sampled_experience
    #after that we only need to run the standard training algorithm
    #but in our new training loop we will also measure and the conflicts
    state_count = run(env=env, 
                      domain = domain, 
                      agent=agent, 
                      render=False, 
                      horizon = int(1e5), 
                      desc = desc, 
                      dim= dim, 
                      label=label, 
                      code = code, 
                      tofl = 1)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Your script description here')
    # Define hyperparameters as command-line arguments
    parser.add_argument('--im', action='store_true', default=False, help='Would you like to use the standrad RND')
    parser.add_argument('--ucb', action='store_true', default=False, help='Would you like to use the standrad RND')
    parser.add_argument('--adeu', action='store_true', default=False, help='Would you like to use the standrad RND')
    parser.add_argument('--heads', type=int, default=-1, help='Would you like to use the standrad RND')

    args = parser.parse_args()

    # Access hyperparameters in your script
    use_im = args.im
    use_ade = args.adeu
    use_ucb = args.ucb
    heads = args.heads

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
        main(seed = s, code = code, label = label, heads = heads )