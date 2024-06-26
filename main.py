import sumo_env
from sumo_env import SumoEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from agent import Agent
import pickle
import os
import datetime
from copy import copy
import xml.etree.ElementTree as ET


def main():
    #declaration of the testing algorithms
    per = False
    
    


 

    ##for the storing/loaded section
    first_time = True #only this should be changed
    evaluation = 15 #how many times this experiment has run
    modify_seed_value(xml_file_path='important/autobahn.sumocfg',new_seed_value=evaluation)
    adaptive_exploration = True
    seed = evaluation
    I = 1

    inside_layer_1 = 256
    inside_layer_2= 128
    batch_size = 128
    cut_training = 565
    testing_period = 50
    
    horizon = 575 + testing_period + 1
    end_training = horizon - testing_period
    episode_size = 800 #in order to acieve execution time in 200 seconds as karalakou did


   
    if first_time: 
        path = f'experiments/naf/c_1_speed_boost_55_45/per_{per}/upd_{I}/evaluation_{evaluation}'
        path_network = None
        
    else:
        path = f'experiments/naf/election/neutral_envx55_45/per_{per}/evaluation_{evaluation}' #where I would like for the final results to be stored 
        path_network = [f'experiments/naf/c_1_neutral_env/second_False_per_False_pse_False/clear_evaluation/evaluation_{evaluation}',
                        f'experiments/naf/c_1_speed_boost_55_45/second_False_per_False_pse_False/evaluation_{evaluation}']  #if i would like also to load a network
    #create this folder if it is not already exists
    s_string = copy(path_network)
    if not os.path.exists(path+'/stored_data'):
        os.makedirs(path+'/stored_data')
    

    #initialiazation if the storing values
    eps = 1
    c = 1
    p = None
    ################################


    #some initializations
    env = SumoEnv(sleep = .0049) #0.0047
    
    
    observation = env.reset()
    current_observation = torch.FloatTensor(observation)
    
    
    #reading from memory
    if not first_time and type(path_network) is not list:
        checkpoint = end_training -1
    elif not first_time and type(path_network) is list:
        checkpoint = end_training - 15
        
        
    print(f"epsilon is {eps}")
    if not first_time:
        if type(path_network) is not list: path_network = path_network + '/stored_data/network_parameters.pth'
        else: 
            for i,p in enumerate(path_network):
                path_network[i] = p + '/stored_data/network_parameters.pth'
    else:
        path_network = None

    agent = Agent( len(observation),
                    hidden_layer_1=inside_layer_1,
                    hidden_layer_2=inside_layer_2,
                    use_priorized_exp_replay= per,
                    batch_size=batch_size,
                    update_target_every= I,
                    max_speed=1.5,
                    gamma= .99,
                    c=c,
                    min_mem_size= batch_size,
                    learning_rate= 1e-3, #robotics is with e-3 cheetah with e-4
                    tau=.001, #it was 0.005
                    action_size=2,
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
                    train_frequency= 5 , #2 dor all in general except cheetha now I am trying for hopper #it was 8 for halg cheetah,
                    adaptive_exploration=adaptive_exploration,
                    decay=int(0.5*horizon*episode_size)
                    )
   
    
    #training loop
    reward_per_episode, speed, coll_per_episode, sum_of_rewards  =  run(env, 
        agent, 
        current_observation, 
        horizon, 
        episode_size,
        first_time = first_time,
        start = checkpoint if not first_time else 0, 
        eps = eps,
        c=c,

        per = per,
        path = path,
        beta_0 = .5,
        begining = 1,
        goal = .98,
        end_training = end_training,
        s_string = s_string)
    
    show(reward_per_episode, speed, coll_per_episode, horizon, path + '/stored_data', sum_of_rewards)
    
    return


def run(env: SumoEnv,  agent: Agent, first_obs, horizon, episode_size,path = None, start = 0, first_time = True, eps=1, c =1, per = False, end_training = 500, goal = .2, begining =1, beta_0 =0.5,s_string =None):
    loss_inside_episode = 0

    current_observation = first_obs
    reward_per_episode = np.zeros(horizon)
    sum_of_rewards = np.zeros(horizon*episode_size)
    speed_per_episode = np.zeros(horizon)
    coll_per_episode = np.zeros(horizon)

    first_call = True

    #for the eps-greedy algorithm/ general exploration
    goal = goal + 0.01
    
    if per : b = beta_0
    
    if not first_time:
        if not type(s_string) is list:
            r = np.load(s_string+'/stored_data/reward.npy')
            speed = np.load(s_string+'/stored_data/speed.npy')
            coll = np.load(s_string+'/stored_data/coll.npy')
            
        else:
            r = np.zeros(horizon-1)
            speed = np.zeros(horizon-1)
            coll = np.zeros(horizon-1)
            for l in s_string:
                r_tmp = np.load(l+'/stored_data/reward.npy')
                speed_tmp = np.load(l+'/stored_data/speed.npy')
                coll_tmp = np.load(l+'/stored_data/coll.npy')
                r+= r_tmp/len(s_string)
                speed+= speed_tmp/len(s_string)
                coll += coll_tmp/len(s_string)
        reward_per_episode[0:start] = r[:start]
        speed_per_episode[0:start] = speed[:start]
        coll_per_episode[0:start] =coll[:start]

    for episode in range(start,horizon):
        #storing some information

        
        
        if(episode == end_training): 
            print("---------------agent was set to testing mode----------------------")
            env.set_sleep(.005)
        if(episode == horizon - 1): save(path, agent=agent, episode = episode,
                 reward_per_episode=reward_per_episode, speed_per_episode=speed_per_episode, coll_per_episode=coll_per_episode, sum_of_rewards=sum_of_rewards)
        
            
        reward_inside_episode = 0
        loss_inside_episode = 0
        speed_inside_episode = 0
        coll_inside_episode = 0
        collisions_timestep_counter = 0 # metric that aggregates the collisions that happend in 10 timesteps | assuming that hapend between the same vehicle
        
        #trainning loop
        for t in range(episode_size):
            if per:
                dt=10e-6 if t==0 else 0
                fraction = min((episode*t+t) / (horizon*t+dt), 1.0)
                b = b + fraction * (1.0 - b)
                agent.set_beta(beta=b)
                        
            if episode >= end_training: action_continuous=agent.get_clear_action(state = current_observation) #returns a random action with .01 else returns the clear action
            else: action_continuous=agent.get_action(current_observation)
            first_call = not first_call if first_call else first_call

             
            action_continuous=np.array(action_continuous.cpu().squeeze().numpy().reshape((agent.action_size,)))
            action_continuous_str = sumo_env.to_string(action_continuous)
            #execute
            new_observation, reward, done, collisions = env.step(action_continuous_str)
            #storing to memory and performing the action
            done = True if collisions > 1 else False
            loss_inside_episode+=agent.step(current_observation, action_continuous, reward, torch.FloatTensor(new_observation), done, episode, timestep = t) #push that in memory
            current_observation = torch.FloatTensor(new_observation)
            #statistics 
            sum_of_rewards[episode*episode_size + t] = reward
            reward_inside_episode+=reward
            speed_inside_episode+=new_observation[1]
            #collisions update
            if collisions == 1 and collisions_timestep_counter < 10:
                collisions_timestep_counter += 1
            elif (collisions_timestep_counter == 10):
                coll_inside_episode += 1
                collisions_timestep_counter = 0
            else : 
                coll_inside_episode += 1 if collisions_timestep_counter > 0 else 0 
                collisions_timestep_counter = 0

           
        #logistics
        reward_per_episode[episode] = reward_inside_episode/episode_size
        speed_per_episode[episode] = speed_inside_episode/episode_size
        #coll_per_episode[episode] = coll_inside_episode/episode_size
        coll_per_episode[episode] = coll_inside_episode
        #now prints,are refering only in the evaluation section
        print(f"episode {episode}, with average reward: {np.round(reward_per_episode[episode], decimals=3)} and average speed: { np.round(speed_per_episode[episode], decimals = 3)}. Average collisions are: {np.round(coll_per_episode[episode], decimals=3)}")
        
        
                
    return reward_per_episode, speed_per_episode, coll_per_episode, sum_of_rewards 

def show(reward_per_episode, speed, collisions, horizon, path, sum_of_rewards):
    #graphs section 
    plt.figure(1)
    plt.xlabel("Episode") 
    plt.ylabel("reward in t") 
    plt.plot(np.arange(1,horizon+1),reward_per_episode, label="reward per episode")
    plt.grid()
    plt.legend()
    plt.savefig(path) # I would have to save fig in order to store it. That is working without depending on .show
    plt.show()


    plt.figure(2)
    plt.xlabel("Episode") 
    plt.ylabel("speed in t") 
    plt.plot(np.arange(1,horizon+1),speed, label="speed in every episode")
    plt.grid()
    plt.legend()
    plt.show()



    plt.figure(3)
    plt.xlabel("Episode") 
    plt.ylabel("collisions in every round") 
    plt.plot(np.arange(1,horizon+1),collisions, label="coll in every episode")
    plt.grid()
    plt.legend()
    plt.show()

    print(f"mean reward is {np.mean(reward_per_episode[575:625])}")
    print(f"mean speed is {np.mean(speed[575:625])}")
    print(f"mean collisions are {np.mean(collisions[575:625])}")
    return

def save(path:str, agent:Agent, reward_per_episode, speed_per_episode, coll_per_episode, episode = None, sum_of_rewards= None):
    
        

    #fix this in order to save every array in different file 
    np.save(path+'/stored_data/reward.npy', reward_per_episode[:episode])
    np.save(path+'/stored_data/speed.npy', speed_per_episode[:episode])
    np.save(path+'/stored_data/coll.npy', coll_per_episode[:episode])
    np.save(path+'/stored_data/sum_of_rewards.npy', sum_of_rewards[:episode*800])

    # Specify the file path where you want to save the parameters
    # Save the parameters
    
    torch.save(agent.target_model.state_dict(), path+'/stored_data/network_parameters.pth')
    with open(path+'/stored_data/experience.pkl', 'wb') as file:
        pickle.dump(agent.replay_memory.memory, file)

def modify_seed_value(xml_file_path, new_seed_value):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Find the 'seed' element and modify its value
    for seed_element in root.iter('seed'):
        seed_element.set('value', str(new_seed_value))

    # Write the modified XML back to the file
    tree.write(xml_file_path)

if __name__ == '__main__':
    time_start = datetime.datetime.now()
    main()
    time_end = datetime.datetime.now()
    duration = time_end - time_start
    duration = duration.seconds /3600
    print(f"duration in hours is {duration}")

