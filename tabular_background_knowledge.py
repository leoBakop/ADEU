import gymnasium as gym
import numpy as np
from q_learning_agent import Q_Learning_Agent, IR_Q_Learning_Agent, UCB_Q_Learning, ADEU_Q_Learning, sADEU_safety,sADEU_safety_uncertainty
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
    for i in range(3, dim): 
        if i == 3 or i==5: 
            epsilon = 1 #just to make sure that the first lines will have a conflict
        if np.random.rand() < epsilon:
            conflict_lines.append(i-2)
            desc.append("HF" + "FFF" + "".join(["H" for _ in range(dim - 5)]))   #outputs one H, 10+1 F
            epsilon = 0.3
        else:
            epsilon = initial_epsilon
            desc.append("HF" + "".join(["H" for _ in range(dim - 2)]))  
        if (len(conflict_lines) > 100):epsilon = 0
    #line_before_before= ("HF" + "".join(["F" for _ in range(dim - 2)]))  
    line_before= ("HF" + "".join(["F" for _ in range(dim - 2)]))     
    #cosntructing the horizontal path
    #desc.append(line_before_before)
    desc.append(line_before)
    final_line = "".join(["H" for _ in range(dim - 1)])+"G"
    desc.append(final_line)
    print(f'{len(conflict_lines)} constraints were created')
    return desc, conflict_lines


def load_background_knowledge(path):
    with open(path, "rb") as file:
        loaded_experience = pickle.load(file)
    return loaded_experience

def find_speciffic_experience(background_knowledge, row, dim):
    random.shuffle(background_knowledge)
    target_state = dim*row + 1

    for exp in background_knowledge:
        if exp[0] == target_state:
            return exp
    print('experience was not found')
    return None

def filter_states(background_knowledge, conflicts, dim, samples = 1500*2):
    """ this method returns the tuples in which an agent should be trained on.
        since the background knowledge contains only the correcttrajectory---and some initial training data
        I will just sample some random.     
    """
    print('sampling the appropriate experience')
    if samples > 0:
        sampled = random.sample(background_knowledge, k=samples)
    else:
        sampled = []
    for c in tqdm(conflicts):
        exp = find_speciffic_experience(background_knowledge, c, dim)
        if exp is not None:
            sampled.append(exp)
    return sampled

def pretraining(agent, background_knowledge, conflicts, code_of_the_agent, dims, iterations = 10):
    """ 
        this method pretrains the agent on the background knowledge.
        this process is repeated multiple times
    """
    if code == 1:
        agent.beta = 100
    for i in tqdm(range(iterations)):
        random.shuffle(background_knowledge)
        for current_observation, action, reward, new_observation, done in background_knowledge:
            action = 1 
            reward *= 2_000 #it was 20
            if current_observation == 123001:
                print('debug')
            if code_of_the_agent == 5: #in the case of a ucb agent
                mask = np.random.binomial(1, agent.p, agent.num_of_agents)
                agent.step([current_observation, action, reward, new_observation, done, mask])
            else:
                agent.step([current_observation, action, reward, new_observation, done])
    print('the agent pretrained succesfully')
    if code == 1:
        agent.beta = 0.01
    return agent

def backgroun_knowledge(conflict_lines, dims, agent, code_of_the_agent):
    high_reward = 5_000
    if code == 1:
        agent.beta = 100
    for c in conflict_lines:
        conflict_state = (c*dims)+1
        if code_of_the_agent == 5: #in the case of a ucb agent
            mask = np.random.binomial(1, agent.p, agent.num_of_agents)
            agent.step([conflict_state, 1, conflict_state * high_reward, conflict_state+dims, False, mask])
            #now in the rare case that tghe aget will visit the conflict, lets motivate it to return to the previous state
            #agent.step([conflict_state, 1, high_reward, conflict_state+dims, False, mask])
        elif code_of_the_agent == 11 or code_of_the_agent == 12:
            agent.step([conflict_state, 1, conflict_state * high_reward, conflict_state+dims, -1, False])
        else:
            agent.step([conflict_state, 1, conflict_state * high_reward, conflict_state+dims, False])
            #now in the rare case that tghe aget will visit the conflict, lets motivate it to return to the previous state
            #agent.step([conflict_state, 1, high_reward, conflict_state+dims, False])

    #also inform the agent about the 
            

    if code == 1:
        agent.beta = 0.01
    return agent

def run(env:gym.Env, agent:Q_Learning_Agent|ADEU_Q_Learning, code, cretaed_conflicts, domain, horizon = 1_000, timesteps = 1000, render = False, label = 0, desc = None, dim = 50, tofl = 0):
    conflicts = 0
    number_of_different_states = 1 # in this variable I will add (until th evaluation) the numeber of different states visited during an episode
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
                    env,
                    timesteps, 
                    domain = domain, 
                    evaluation_episodes = 3, 
                    label = label, 
                    episode = episode, 
                    iterations = int(horizon + 100), 
                    desc = desc, 
                    normalizer=normalizer, 
                    keep_trajectories=False,
                    save_experience=False)
            c = conflicts/(100/percentage)
            wandb.log({"std error during training": mean_std})
            wandb.log({"mean conflicts per episode": c})
            wandb.log({'conflicts over numebr of unique states visisted per episode': conflicts/number_of_different_states}) #since both contflicts and number of different staes should have been normizlided with the same factor. I do normilize nothing
            number_of_different_states = 0
            conflicts = 0
            mean_std = 0
        agent.set_eps(epsilon[episode if episode < epsilon.size else -1])
        obs, _ = env.reset()
        row, col = reverse_hash(obs,dim)
        states_count[row, col]+=1
        for t in range(timesteps):

            if render: env.render()
            
            row_obs, col_obs = reverse_hash(obs, dim)
            #if row_obs in cretaed_conflicts:
            #    print('debug')
            
            if code == 5: 
                action, std = agent.get_action(obs)
                mean_std += std*percentage/(timesteps*100)
            else: action = agent.get_action(obs)
            
            if tofl != 0 and action == 3 and col == 1:
                action = np.random.choice(np.arange(3), size = 1)[0] #it was arrange(2)
            if tofl != 0 and action == 0 and col > 1:   
                action = np.random.choice(np.arange(1,4), size = 1)[0]
            
            new_obs, reward, done, truncated, info = env.step(action)

            row, col = reverse_hash(new_obs,dim)
            
            states_count[row, col]+=1
            cost = 0
            if desc[row][col] == 'H':
                reward = -10
                done = True
            elif desc[row][col] == 'G':
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
                if col != 1 and row in cretaed_conflicts:
                    cost = 2
                    conflicts+= 1
                    reward = 0
                    #reward = (dim-1)*new_obs/normalizer
                    
            if code == 5:
                mask = np.random.binomial(1, agent.p, agent.num_of_agents)
                agent.step([obs, action, reward, new_obs, done or truncated, mask])
            elif code == 11 or code == 12:
                agent.step([obs, action, reward, new_obs, cost, done or truncated])
            else:
                agent.step([obs, action, reward, new_obs, done or truncated])
            
            obs = new_obs
            
            if (done):
                break
        number_of_different_states += t
        different_states_visited_per_epsiode = []
        if code == 1: 
            agent.rollout = False
    env.close()
    wandb.finish()
    
    return states_count







def main(seed, code, label, heads):
    set_seed(seed=seed)
    dim = 1500
    desc, conflicts = generate_custom_map(dim=dim, initial_epsilon=0.05)
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
        beta = 0.01 # it was 0.01, after the pretraining orocess is over this should change back to 0.01
        agent_params['beta'] = beta
        agent_params['rollout_frequency'] = 0.0
        agent = ADEU_Q_Learning(**agent_params)
    elif code == 11:
        beta = 1
        agent_params['beta'] = beta
        agent_params['delta'] = 0.01
        agent_params['initial_expected_cost'] = 0.1
        agent = sADEU_safety(**agent_params)
    elif code == 12:
        beta = 1
        agent_params['beta'] = beta
        agent_params['delta'] = 0.01
        agent_params['initial_expected_cost'] = 0.01
        agent = sADEU_safety_uncertainty(**agent_params)

    else: 
        agent = Q_Learning_Agent(**agent_params)
    
    domain ="SafetyFrozenLake"
    #since we have already define everything lets try to load the background knowledge
    #background_knowledge = load_background_knowledge(path = './experiments/tabular/Safety_CustomFrozenLake/background_experienc.pkl')
    
    #now lets train the agent in a some speciffic states
    
    """ sampled_experience = filter_states(background_knowledge, conflicts, samples = 0, dim = dim)
    with open('sampled_experience.pkl', 'wb') as f:
        pickle.dump(sampled_experience, f)
    del background_knowledge 
    with open('sampled_experience.pkl', 'rb') as f:
        sampled_experience = pickle.load(f) """
    #then I will train the agent in the `sampled experience`
    """agent = pretraining(agent = agent, 
                        background_knowledge=sampled_experience, 
                        conflicts=conflicts, 
                        dims=dim, 
                        code_of_the_agent=code, 
                        iterations = 1000)"""
    agent = backgroun_knowledge(conflict_lines=conflicts, dims = dim, agent=agent, code_of_the_agent=code)
    

    #we free both the background knowledge and the sampled experience
    
    #del sampled_experience
    wandb_session(domain = domain, seed = seed, debug = False, code = code, name = name, beta = beta, heads = heads)

    #after that we only need to run the standard training algorithm
    #but in our new training loop we will also measure and the conflicts
    state_count = run(env=env, 
                      cretaed_conflicts = conflicts,
                      domain = domain, 
                      timesteps = int(3*dim),
                      agent=agent, 
                      render=False, 
                      horizon = int(1e4), 
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
    parser.add_argument('--sadeu', action='store_true', default=False, help='Would you like to use the standrad RND')
    parser.add_argument('--sadeu_unc', action='store_true', default=False, help='Would you like to use the standrad RND')
    args = parser.parse_args()

    # Access hyperparameters in your script
    use_im = args.im
    use_ade = args.adeu
    use_ucb = args.ucb
    heads = args.heads
    use_sadeu = args.sadeu
    use_sadeu_unc = args.sadeu_unc

    arguments = {
        'adeu': use_ade,
        'sadeu' : use_sadeu,
        'sadeu_unc' : use_sadeu_unc,
        'intrisic motivation':use_im,
        'ucb':use_ucb,
        'Baseline': (not use_ade) and (not use_im) and (not use_ucb)
    }
    label = [a for a in arguments.keys() if arguments[a]][0]
    
    seed = 5
    code = code_creation(adaptive_exploration=use_ade, use_rnd = use_im, ucb=use_ucb, use_sadeu = use_sadeu, use_sadeu_unc = use_sadeu_unc )
    for s in range(seed):
        main(seed = s, code = code, label = label, heads = heads )