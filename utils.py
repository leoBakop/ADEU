import torch
import numpy as np
#methods that will simulate the sumo environment
#this code was originally written in sumo_env.py and
#just copied in order to achiece compatibllity
edge_width = 10.2
edge_length = 2000.0 / 4
vx_max = 36.0
vy_max = 10.0
def emulate_step(observation, action):
    # x_t
    y_t = observation[0]
    vx_t = observation[1]
    vy_t = observation[2]
    v_des = observation[3]
    #continuous = isinstance(action, int)
    #action is a Tensor now

    
    ax_t, ay_t = action[0], action[1]
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

dt = 0.25
front = 3
rear = 2

def get_speed_reward(speed, desired_speed):
    reward = abs(desired_speed - speed) / desired_speed
    return reward
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


def get_reward(speed, desired_speed, collisions, observation):
    reward = 0.0
    c1 = .55 #it was 0.65
    speed_reward = get_speed_reward(speed, desired_speed)
    return c1*speed_reward + collisions

# Define your custom dynamics function
def dynamics(state, control, update = [0]*20):
    # Implement your custom dynamics here
    retval= emulate_step(state, control)
    offset = len(retval)
    #update the parameters of the agent
    for j,i in enumerate(retval):
        state[j] = i
    #update the parameters of the other vehicles
    for j,i in enumerate(update):
        state[j+offset] += i
    return state


def refit(states, new_states):
    difference = np.zeros((new_states.shape[0], new_states.shape[1]-4))
    states, new_states = np.array(states[:,4:].detach().cpu()), np.array(new_states[:,4:].detach().cpu()) 
    for i, (old,new) in enumerate(zip(states, new_states)):
        difference[i]=(np.abs(new-old))

    return np.mean(difference, axis = 0)


# Define your custom cost functionz
def cost(state, control, update):
    state = dynamics(state, control, update)
    collisions = state[-1]
    state = state[:-1]

    reward = get_reward(state[1], state[3], collisions, state) 
    
    return max(reward,0)  #-10 is a magic number in order to make sure that cost =-reward - 10 is always negative


def append_zero_to_each_tensor(tensor_list):
    # Iterate over each tensor in the list
    ret_val = []
    for i in range(len(tensor_list)):
        # Create a zero tensor of appropriate size
        zero_tensor = torch.zeros(1)

        # Concatenate the zero tensor to the current tensor
        ret_val.append(torch.cat((tensor_list[i].detach().cpu(), zero_tensor)))

    return ret_val