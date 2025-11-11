import matplotlib.pyplot as plt
import numpy as np
import os
from process_local_data import exponential_smoothing

MAXIMUM_SEEDS = 10
MAXIMUM_TIMESTEPS = int(1e6)


def main():
    algorithms = ['dqn', 'bdqn', 'ucb', 'ez_adeu']
    mean_values = {}
    std_errors = {}
    path = 'experiments/DEEPSEA/'
    for alg in algorithms:
        mean_values[alg], std_errors[alg] = load_results(alg = alg, path=path)
    
    plot_results(mean_values=mean_values, std_errors = std_errors, path = path)
    return 


def load_results(alg, path):
    path +=alg
    mean_values = []
    std_erros = []
    if not os.path.exists(path):
        print(f'error in finding {path}')
        return
    for dim in range(3,100):
        if not os.path.exists(f'{path}/{dim}'):
            continue
        path_dim = f'{path}/{dim}'
        for s in range(MAXIMUM_SEEDS):
            result_among_seeds = []
            
            path_dim_seed = path_dim + f'/seed_{s}/'
            if os.path.exists(path_dim_seed):
                data = {}
                with open(path_dim_seed + "hyperparameters.txt", "r") as f:
                    for line in f:
                        if ":" in line:
                            key, value = line.strip().split(":", 1)
                            data[key.strip()] = value.strip()

                
                result_among_seeds.append(int(data["solved_in"]))

                

            while len(result_among_seeds) < MAXIMUM_SEEDS:
                result_among_seeds.append(MAXIMUM_TIMESTEPS)

            result_among_seeds = exponential_smoothing(
                np.array(result_among_seeds),
                np.arange(len(result_among_seeds)),
                smoothing_param=1
            )
            mean_timesteps = np.mean(np.array(result_among_seeds))
            std_timesteps = np.std(np.array(result_among_seeds), ddof=0)

        mean_values.append(mean_timesteps)
        std_erros.append(std_timesteps)

        
    return mean_values, std_erros


def plot_results(mean_values, std_errors, path):
    plt.figure(figsize=(10, 5))
    plt.title(r'Timesteps required to solve the DeepSea domain in an $N\times N$ grid')
    for key in mean_values.keys():
        x_axis = np.arange(3, 3*len(mean_values[key]) + 1, 3)
        mean = np.array(mean_values[key])
        std = np.array(std_errors[key])
        upper_bound = np.clip(mean + std, a_min = 0,  a_max = MAXIMUM_TIMESTEPS)
        lower_bound = np.clip(mean - std, a_min = 0, a_max = MAXIMUM_TIMESTEPS)
        plt.plot(x_axis, mean, label=key)
        plt.fill_between(x_axis, lower_bound, upper_bound, alpha=0.1)
    plt.ylabel('Timesteps')
    plt.xlabel('N')
    plt.legend()
    plt.savefig(path+'/results.png')
    plt.show()
    


if __name__ == '__main__':
    main()