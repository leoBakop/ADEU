import gym
from gym import Env, spaces
import torch
import numpy as np
import subprocess
import os
import sys
import time
import torch
cur_dir = os.path.dirname(os.path.abspath(__file__))
# print("current dic: ", cur_dir)
sys.path.append(cur_dir)
from shared_memory.prodcon_ipc.shared_memory import SharedMem



dt = 0.25
front = 3
rear = 2

def get_acceleration(action, continuous = False):
        
        if continuous:
            return action

        ux = 0
        uy = 0
        if action == 1:
            ux = 2
            uy = 0
        elif action == 2:
            ux = 1
            uy = 1
        elif action == 3:
            ux = 0
            uy = 2
        elif action == 4:
            ux = -1
            uy = 1
        elif action == 5:
            ux = -2
            uy = 0
        elif action == 6:
            ux = -1
            uy = -1
        elif action == 7:
            ux = 0
            uy = -2
        elif action == 8:
            ux = 1
            uy = -1
        return ux, uy

def get_speed_reward(speed, desired_speed):
    reward = abs(desired_speed - speed) / desired_speed
    return reward

edge_width = 10.2
edge_length = 2000.0 / 4
vx_max = 36.0
vy_max = 10.0

veh_width = veh_width_cav = 1.8
veh_length = veh_length_cav = 3.2
def get_force_reward(dvx, dvy, vx_avg, vy, vy_neigh, dx, dy):
    vx_collide = (dx * dvx) < 0
    vy_collide = (dy > veh_width and dvy < 0) or (dy < -veh_width and dvy > 0)
    vy_in_range = abs(dy) < veh_width
    return repulsive_force(dx, dy, abs(dvx) * vx_collide, abs(dvy) * vy_collide, abs(dvy) * vy_in_range, vx_avg, vy, vy_neigh)


c_longitudinal_decel = -2.5 #maximum longitudinal deceleration for all vehicles
c_lateral = 1.5 #maximum lateral acceleration for all vehicles
a_s = (veh_length + veh_length_cav)/2 + 0.8
b_s = (veh_width + veh_width_cav)/2 + 0.8
react_time_x = 0.3
react_time_y = 0.4
magnitude = 0.5
pow_x = 2
pow_y = 4
pow_t = 2
pow_t2 = 2
pow_x_2 = 2
pow_y_2 = 4
def repulsive_force(r_x, r_y, k_x, k_y, k_y_i, v_x, vy1, vy2):
    r_x_d = abs(r_x)
    r_y_d = abs(r_y)
    

    a = (k_x * k_x) / (abs(c_longitudinal_decel)) + react_time_x * v_x + a_s
    

    b = b_s +  (k_y * k_y / (c_lateral)) + react_time_y * (abs(vy1) + abs(vy2))

    if r_x_d > a or r_y_d > b:
        #print(force)
        return 0
    #b = (e_b*k_y + e_b+0.2) * min(veh_length/r_x_d,1)

    px = np.power(r_x_d / a_s, pow_x)

    py = np.power(r_y_d / b_s, pow_y)

    px2 = np.power(r_x_d / a, pow_x_2)
    py2 = np.power(r_y_d / b, pow_y_2)


    force = magnitude / np.power(px + py + 1, pow_t) + magnitude / np.power(px2 + py2 + 1, pow_t2)

    
    return force


def check_overtake(new_state, old_state):
    """In this method i will check if the agent tries to overtake a vehicle
    using the following algorithm

    Using two consecutively states/observations

    if the vehicle that was in front me in the old state 
    is now behind me, I am trying to overtake

    I am going to check this, using the id of the vehicle
    """

    if old_state is None:
        return False
    new_neigh_behind_id = new_state[-1][0]

    old_neigh_front_id = old_state[-1][1]

    if new_neigh_behind_id == old_neigh_front_id:
        print("overtake:",new_neigh_behind_id)
        return True

    return False

def get_reward(speed, desired_speed, collisions, observation):
    reward = 0.0
    wx = .55  #0.65 
    wy = .45 #1.0
    speed_reward = get_speed_reward(speed, desired_speed)
    vx = observation[1]
    vy = observation[2]
    force_reward = 0
    for i in range(front + rear):
        vx_neigh = observation[3 + i * 4 + 3] # observation[..] always returns Vx of neighbour i
        vy_neigh = observation[3 + i * 4 + 4]
        vx_avg = (speed + vx_neigh) / 2
        dvx = vx_neigh - vx
        dvy = vy_neigh - vy
        dx = observation[3 + i * 4 + 1]
        dy = observation[3 + i * 4 + 2] - observation[0]
        force_reward += get_force_reward(dvx, dvy, vx_avg, vy, vy_neigh, dx, dy)

    reward = 0.1 / (0.1 + wx * speed_reward + wy * force_reward) -  2.5 * collisions
    return reward


def to_string (np_array):
    retval = ""
    for j,i in enumerate(np_array):
        retval += str(i)+" " if j == 0 else str(i)
    return retval    

def normalize_state(state):
  state[0] = state[0] / edge_width
  state[1] = state[1] / vx_max
  state[2] = state[2] / vy_max
  state[3] = state[3] / vx_max

  state[4] = state[4] / edge_length
  state[5] = state[5] / edge_width
  state[6] = state[6] / vx_max
  state[7] = state[7] / vy_max

  state[8] = state[8] / edge_length
  state[9] = state[9] / edge_width
  state[10] = state[10] / vx_max
  state[11] = state[11] / vy_max
  
  state[12] = state[12] / edge_length
  state[13] = state[13] / edge_width
  state[14] = state[14] / vx_max
  state[15] = state[15] / vy_max
  
  state[16] = state[16] / edge_length
  state[17] = state[17] / edge_width
  state[18] = state[18] / vx_max
  state[19] = state[19] / vy_max
  
  state[20] = state[20] / edge_length
  state[21] = state[21] / edge_width
  state[22] = state[22] / vx_max
  state[23] = state[23] / vy_max

  return state




class SumoEnv(Env):
    def __init__(self, sleep = 0.03):
        time.sleep(3)
        super(SumoEnv, self).__init__()
        self.sleep = sleep
        self.state_threshold = 1.0
        high = np.array(
            [
                self.state_threshold, # y position
                self.state_threshold, # x speed
                self.state_threshold, # y speed
                self.state_threshold, # desired speed
                # observation for self ends here
                self.state_threshold, # neighbour x position difference with respect to self
                self.state_threshold, # neighbour y position
                self.state_threshold, # neighbour x speed
                self.state_threshold, # neighbour y speed
                # neighbour 1 observation ends here (front)
                self.state_threshold, # neighbour x position difference with respect to self
                self.state_threshold, # neighbour y position
                self.state_threshold, # neighbour x speed
                self.state_threshold, # neighbour y speed
                # neighbour 2 observation ends here (front)
                self.state_threshold, # neighbour x position difference with respect to self
                self.state_threshold, # neighbour y position
                self.state_threshold, # neighbour x speed
                self.state_threshold, # neighbour y speed
                # neighbour 3 observation ends here (front)
                self.state_threshold, # neighbour x position difference with respect to self
                self.state_threshold, # neighbour y position
                self.state_threshold, # neighbour x speed
                self.state_threshold, # neighbour y speed   
                # neighbour 4 observation ends here (rear)
                self.state_threshold, # neighbour x position difference with respect to self
                self.state_threshold, # neighbour y position
                self.state_threshold, # neighbour x speed
                self.state_threshold  # neighbour y speed
                # neighbour 5 observation ends here (rear)
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        self.sh_mem = SharedMem()
        self.ep_return = 0
        
        cpp_args = ['gnome-terminal', '--', cur_dir + '/important/sumo-gui', '-c', cur_dir + '/important/autobahn.sumocfg', '--start', 'true'] #original
        #cpp_args = ['gnome-terminal', '--', cur_dir + '/important/sumo-gui', '-c', '/important/autobahn.sumocfg', '--start', 'true']
        #cpp_args=["gnome-terminal", "--", "bash", "-c", cur_dir + f"/important/sumo-gui -c {cur_dir}/important/autobahn.sumocfg --start true; exec bash"]
        self.cpp_proc = subprocess.Popen(args=cpp_args, preexec_fn=os.setpgrp)
        return
    def set_sleep(self, sleep):
        self.sleep = sleep

    def get_sleep(self):
        return self.sleep   
    def reset(self):
        # reset
        observation_string = self.sh_mem.load_from_memory()
        
        observation = [float(x) for x in observation_string.split(';')][:-1]
        # print("observation pre normalize_state: ", observation)
        observation = normalize_state(observation)
        # print("observation aft normalize_state: ", observation)
        self.ep_return = 0
        return observation

    def step(self, action):
        # termination of an episode
        done = False

        # assert self.action_space.contains(action), "Invalid action"

        self.sh_mem.load_to_memory(action)
        time.sleep(self.sleep) #new new code imported by leo
        observation_string = self.sh_mem.load_from_memory()
        observation = [float(x) for x in observation_string.split(';')]
        collisions = observation[-1]
        observation = observation[:-1]

        # print("Observation received: ", observation, ", reward: ", reward)
        self.ep_return += 1
        reward = get_reward(observation[1], observation[3], collisions, observation) 
        # print("observation pre normalize_state: ", observation)
        observation = normalize_state(observation)
        # print("observation aft normalize_state: ", observation)

        return observation, reward, done, collisions   

    
    def emulate_step(self, observation, action):
        # x_t
        y_t = observation[0]
        vx_t = observation[1]
        vy_t = observation[2]
        v_des = observation[3]
        #continuous = isinstance(action, int)
        #action is a Tensor now
        

        ax_t, ay_t = get_acceleration(action, continuous=False)

        # x_next = x_t + (vx_t * dt) + (0.5*a_x*(pow(dt,2)))
        y_next = (y_t * edge_width + (vy_t * vy_max * dt) + (0.5 * ay_t * pow(dt,2))) / edge_width
        vx_next = (vx_t * vx_max + (ax_t * dt)) / vx_max
        vy_next = (vy_t * vy_max + (ay_t * dt)) / vy_max
        emulated_observation = [y_next, vx_next, vy_next, v_des]

        # for neighbour in range(front + rear):
        #     dx_t = observation[4 + neighbour]
        #     y_other = observation[5 + neighbour]
        #     vx_other = observation[6 + neighbour]
        #     vy_other = observation[7 + neighbour]
        #     ax_other = 0.0
        #     ay_other = 0.0
        #     # maybe ax_other = ax_t ?

        #     #TODO clip values between (0,1) or (-1,1)
        #     dx_next = dx_t + ((vx_other - vx_t)* dt) + (0.5*(ax_other-ax_t)*(pow(dt,2)))
        #     y_other_next = y_other + (vy_other * dt) + (0.5 * ay_other * pow(dt,2))
        #     vx_other_next = vx_other + (ax_other * dt)
        #     vy_other_next = vy_other + (ay_other * dt)
        #     emulated_observation.extend([dx_next, y_other_next, vx_other_next, vy_other_next])
        
        return emulated_observation

    def close(self):
        # close sumo
        return

    def render(self):
        return
        
if __name__ == '__main__':
    
    env = SumoEnv()
    obs = env.reset()
    i = 0
    while True:
        # action = input("Enter action (0, 1, 2, 3, 4, 5, 6, 7, 8): ")
        i+=1
        #action = str(randrange(9))
        sign = np.random.randint(0,2)
        sign = (-1) ** sign
        action  = 2*sign*np.round([np.random.rand(), np.random.rand()], decimals = 4)
        time.sleep(.2)
        deb = to_string(action)
        retval = env.step(deb)
        
        
