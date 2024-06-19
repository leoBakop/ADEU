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


if __name__ == '__main__':
    times = 0
    horizon = 50
    total_episodes = 5_005
    iterations = 3
    window_size = 5
    domain = "MountainCarContinuous-v0"
    metric = "cum_evaluation_rewards.npy"

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
        file_path = f"experiments/{domain}/per_False_psne_False/seed_{i}/stored_data/"
        reward = np.load(file_path+metric) [:horizon]
        avg_rew_naf.append(reward)
    
    rewards[0] = extend(np.mean(np.array(avg_rew_naf), axis=0), times)
    cov[0] = extend(np.std(np.array(avg_rew_naf), axis=0), times)


    for i in samples:
        file_path = f"experiments/{domain}/per_False_psne_True/seed_{i}/stored_data/"
        reward = np.load(file_path+metric) [:horizon]
        avg_rew_naf_psne.append(reward)

    rewards[1] = extend(np.mean(np.array(avg_rew_naf_psne),  axis=0).reshape(1,-1), times)
    cov[1] = extend(np.std(np.array(avg_rew_naf_psne), axis=0).reshape(1,-1), times)


    for i in samples:
        file_path = f"experiments/{domain}/per_True_psne_True/seed_{i}/stored_data/"
        reward = np.load(file_path+metric) [:horizon]
        avg_rew_naf_per_psne.append(reward)

    rewards[2] = extend(np.mean(np.array(avg_rew_naf_per_psne),  axis=0).reshape(1,-1), times)
    cov[2] = extend(np.std(np.array(avg_rew_naf_per_psne), axis=0).reshape(1,-1), times)

    for i,(r,c) in enumerate(zip(rewards, cov)):
        rewards[i] = moving_average(r,window_size)
        cov[i] = moving_average(c,window_size)
    
    plt.plot(time, rewards[0], color='red', linewidth=2, label='NAF')
    plt.fill_between(time, rewards[0] - cov[0], rewards[0] + cov[0], color='tomato', alpha=0.4)

    plt.plot(time, rewards[1], color='darkgreen', linewidth=2, label='NAF+PSNE')
    plt.fill_between(time, rewards[1] - cov[1], rewards[1] + cov[1], color='green', alpha=0.4)

    plt.plot(time, rewards[2], color='navy', linewidth=2, label='NAF+PER+PSNE')
    plt.fill_between(time, rewards[2] - cov[2], rewards[2] + cov[2], color='cornflowerblue', alpha=0.4)

    # Add labels and legend
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title(f'Cumulative reward at each episode in {domain} ')
    plt.legend()
    plt.grid(True)

    directory_path = f"./fig/{domain}/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    plt.savefig(directory_path+f'{domain+metric}.png')
    # Show the plot
    plt.show()


    print(f"#########################{domain}#######################")
    print(f"#Maximum Reward for NAF {np.max(rewards[0])}############")
    print(f"#Maximum Reward for NAF+PER {np.max(rewards[1])}########")
    print(f"#Maximum Reward for NAF+PER+PSNE {np.max(rewards[2])}###")

    