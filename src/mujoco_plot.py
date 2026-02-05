import numpy as np
import os
import matplotlib.pyplot as plt


def moving_average(data, window_size):
    if window_size == 0: return data
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def extend(arr, times):
    if times == 0: return arr
    arr = arr.squeeze()
    last_value = arr[-1]
    
    # Repeat the last value 'times' times
    extension = np.tile(last_value, times)
    
    # Concatenate the extension to the original array
    extended_arr = np.concatenate((arr, extension))
    
    return extended_arr






def main(domain, metric, exploration):
    times = 0
    horizon = 50
    total_episodes = 5_005
    iterations = 3
    window_size = 5
    

    h = (horizon-window_size+1) if window_size != 0 else horizon
    h = h + times
    time = np.linspace(0, total_episodes, h)




    avg_rew_naf = []
    avg_rew_naf_psne = []
    avg_rew_naf_per_psne = []
    rewards = [[]]*3 #0: naf, 1: naf_psne, 2: naf_per_psne
    cov = [[]]*3 #0: naf, 1: naf_psne, 2: naf_per_psne

    samples = list([i for i in range(iterations)  ])

    for i in samples:
        file_path = f"experiments/TD3/{domain}/{exploration}/seed_{i}/stored_data/"
        reward = np.load(file_path+metric) 
        avg_rew_naf.append(reward)
    
    rewards[0] = extend(np.mean(np.array(avg_rew_naf), axis=0), times)
    cov[0] = extend(np.std(np.array(avg_rew_naf), axis=0), times)


    for i,(r,c) in enumerate(zip(rewards, cov)):
        rewards[i] = moving_average(r,window_size)
        cov[i] = moving_average(c,window_size)



    print(f"#########################{domain}+{exploration} #######################")
    print(f"#Mean reward {np.round(np.mean(rewards[0]), decimals = 0)}############")


if __name__ == '__main__':
    domain = "InvertedDoublePendulum-v4"
    metric = "cum_evaluation_rewards.npy"
    exploration = ['Baseline', 'ade', 'rnd', 'noisy_nets', 'Baseline+ade']
    for e in exploration:
        main(domain = domain, metric = metric, exploration=e)