import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CODES_MAP={'code: 0':'baseline',
           'code: 2': 'IM',
           'code: 3': 'Noisy Nets',
           'code: 5': 'UCB',
           'code: 21': 'ADEU+im',
           'code: 6': 'ADEU+ensemble',
           'code: 7': 'ADEU+ensemble+rho_0.1',
           'code: 1': 'ADEU',
           'code: 11': 'sADEU',
           'code: 12': 'sADEU + unc.'
           }

CODES_TO_PATHS = {
    'code: 0': 'Baseline',
    'code: 1': 'ade',
    'code: 2': 'rnd',
    'code: 3': 'noisy_nets',
    'code: 5': 'UCB',
    'code: 6': 'Adeu+ensemble',
    'code: 21': 'ade mod_rnd',
    'code: 7': 'Adeu+ensemble'
}


def mean_values(codes, path, seeds):
    values = {}
    rewards_per_code = {}
    for i,c in enumerate(codes):
        #/home/intelligence-lab-pc4/leonidas/adaptive_exploration/code/experiments/TD3/new_mujoco_main/Hopper-v5/Baseline/seed_0/stored_data
        #/experiments/TD3/new_mujoco_main/Hopper-v5/Baseline/seed_0/stored_data/cum_evaluation_rewards.npy
        rewards = []
        buffer = [np.load(path+f'{CODES_TO_PATHS[c]}/seed_{s}/stored_data/cum_evaluation_rewards.npy') for s in range(seeds)]
        for r in buffer:
            #drop the last all zero added for safety reasons
                # Find last nonzero index
            last_nonzero_idx = np.max(np.nonzero(r))

            # The first all-zero index is one after that
            first_all_zero_idx = last_nonzero_idx + 1 if last_nonzero_idx + 1 < len(r) else None
            rewards.append(r[:first_all_zero_idx])

        
        rewards = np.array(rewards)
        rewards = np.mean(rewards, axis=0)

        rewards = exponential_smoothing(y_values=rewards, x_values=np.arange(rewards.shape[0]), smoothing_param=0.5)
        rewards = np.array(rewards)
        mean_reward = round(np.mean(rewards), 2)
        max_reward = round(np.max(rewards), 2)
        values[CODES_MAP[codes[i]]] = [mean_reward,max_reward]
        rewards_per_code[c] = rewards
    return values, rewards_per_code


def print_results(results):
    best_results = find_best_performance(results, percentage=0.9)
    print('Algorithm |  Mean Reward |  MAX reward ')
    for key, value in results.items():
        if key in best_results.keys() :
            print(f'\033[1;31m{key}| {value[0]} | {value[1]}\033[0m')
        else:
            print(f'{key}| {value[0]} | {value[1]}')


def find_best_performance(results, percentage = 0.9):
    best_key = None
    best_value = -np.inf
    retval = {}
    #find the best value
    for key, value in results.items():
        if value[0] > best_value:
            best_value = value[0]
            best_key = key
    retval[best_key] = best_value
    #find th values that are at least percentage persent as 
    #good as the best
    limit = percentage*best_value
    for key, value in results.items():
        if value[0] > limit and key !=best_key:
            retval[key] = value[0]


    return retval
def exponential_smoothing(y_values, x_values, smoothing_param=0, viewport_scale=1):
    """
    Apply exponential smoothing to a series of y-values with optional x-value spacing.
    
    Args:
        y_values: List/array of y-values to smooth
        x_values: List/array of corresponding x-values (for variable spacing)
        smoothing_param: Smoothing parameter (0 = no smoothing, higher = more smoothing)
        viewport_scale: Scaling factor for x-axis differences (default 1)
    
    Returns:
        List of smoothed y-values
    """
    if y_values.shape[0] == 0:
        return []
    
    smoothing_weight = min(math.sqrt(smoothing_param or 0), 0.999)
    last_y = 0
    debias_weight = 0
    smoothed_values = []
    range_of_x = max(x_values) - min(x_values) if len(x_values) > 1 else 1
    
    for i in range(len(y_values)):
        # Calculate adjusted smoothing weight based on x-spacing
        prev_x_index = i - 1 if i > 0 else 0
        change_in_x = ((x_values[i] - x_values[prev_x_index]) / range_of_x) * viewport_scale
        smoothing_weight_adj = math.pow(smoothing_weight, change_in_x)
        
        # Apply exponential smoothing
        last_y = last_y * smoothing_weight_adj + y_values[i]
        debias_weight = debias_weight * smoothing_weight_adj + 1
        
        smoothed_values.append(last_y / debias_weight)
    
    return smoothed_values


def ignore_outliers(df, iqr_scale=1.5):
    # Ensure it's a copy to avoid chained assignment issues
    series = df.copy()

    q1 = series.quantile(0.3)
    q3 = series.quantile(0.7)
    iqr = q3 - q1
    lower = q1 - iqr_scale * iqr
    upper = q3 + iqr_scale * iqr


    # Loop through series and replace outliers with previous valid value
    for i in range(1, len(series)):
        val = series.iloc[i]
        if val < lower or val > upper:
            # Find previous valid value
            j = i - 1
            while j >= 0:
                prev_val = series.iloc[j]
                if lower <= prev_val <= upper:
                    series.iloc[i] = prev_val
                    break
                j -= 1
            # If no valid previous value found, keep the original

    return series


def create_graph(df, codes, domain):

    plt.figure(figsize=(10, 5))

    for i,c in enumerate(codes):
        rewards = df[c+' - cumm_evaluation_reward']
        min_rewards = df[c+' - cumm_evaluation_reward__MIN']
        max_rewards = df[c+' - cumm_evaluation_reward__MAX']

        min_rewards = 1.25 * ignore_outliers(min_rewards) #1.5 except humanoid and swimmer and walker  in which I set it to 1.25
        max_rewards = 0.65 * ignore_outliers(max_rewards) #0.5 except humanoid and swimmer and walker (max 0.65) in which I set it to 0.85


        rewards = exponential_smoothing(y_values=rewards, x_values=df['Step'], smoothing_param=0.5)
        min_rewards = exponential_smoothing(y_values=min_rewards, x_values=df['Step'], smoothing_param=0.5)
        max_rewards = exponential_smoothing(y_values=max_rewards, x_values=df['Step'], smoothing_param = 0.5)


        # Plotting
        #x = np.arange(n_points)
        plt.plot(np.arange(len(rewards)), rewards, label=CODES_MAP[c])
        plt.fill_between( np.arange(len(rewards)), min_rewards, max_rewards, alpha=0.2)
        plt.title(f"Cumulative episodic reward in the {domain} domain")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+f'/{domain}_cumulative_rewards_plot_with_std.png')
    plt.show()


def create_graph_2(rewards_per_code, codes, domain, seeds):
    '''
    This method outputs the same result as the create_graph(). The main (and only) difference is 
    that this also outputs the variance instead of the minimum and maximum value
    '''
    plt.figure(figsize=(10, 5))

    for i,c in enumerate(codes):
        #rewards = df[c+' - cumm_evaluation_reward']
        variance, rewards = get_variance(c, seeds, domain)
        
        #min_rewards = 1.25 * ignore_outliers(min_rewards) #1.5 except humanoid and swimmer and walker  in which I set it to 1.25
        #max_rewards = 0.65 * ignore_outliers(max_rewards) #0.5 except humanoid and swimmer and walker (max 0.65) in which I set it to 0.85
        variance = 0.6 * ignore_outliers(pd.Series(variance))
        rewards = ignore_outliers(pd.Series(rewards))
        #rewards = pd.Series(rewards)

        rewards = exponential_smoothing(y_values=rewards, x_values=np.arange(len(rewards)), smoothing_param=0.5)
        variance = exponential_smoothing(y_values=variance, x_values=np.arange(len(variance)), smoothing_param=0.5)
        #min_rewards = exponential_smoothing(y_values=min_rewards, x_values=df['Step'], smoothing_param=0.5)
        #max_rewards = exponential_smoothing(y_values=max_rewards, x_values=df['Step'], smoothing_param = 0.5)
        rewards = np.array(rewards)

        # Plotting
        #x = np.arange(n_points)
        plt.plot(np.arange(len(variance)), rewards, label=CODES_MAP[c])
        plt.fill_between( np.arange(len(variance)), rewards - variance, rewards + variance, alpha=0.2)
        plt.title(f"Cumulative episodic reward in the {domain} domain")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+f'/{domain}_cumulative_rewards_plot.png')
    plt.show()


def get_variance(code, seeds, domain):

    
    variances = []
    rewards = []
    for s in range(seeds):
        path = f'experiments/TD3/{domain}/{CODES_TO_PATHS[code]}/seed_{s}/stored_data/cum_evaluation_rewards.npy'
        variances.append(np.load(path))
        rewards.append(np.load(path))

    rewards = np.array(rewards)
    rewards = np.mean(rewards, axis= 0)
    variances = np.array(variances)
    variances = np.std(variances, axis=0, ddof=1) 

    # Find last nonzero index
    last_nonzero_idx = np.max(np.nonzero(rewards))

    # The first all-zero index is one after that
    first_all_zero_idx = last_nonzero_idx + 1 if last_nonzero_idx + 1 < len(rewards) else None
    return variances[:first_all_zero_idx], rewards[:first_all_zero_idx]


if __name__=='__main__':
    # 'domain' variable refers to the aaai_wandb... path
    results_in_new_mujoco_main = True
    domain = 'Hopper-v5'
    seeds = 2
    path = '/home/intelligence-lab-pc4/leonidas/adaptive_exploration/code/'
    path += f'experiments/TD3/{domain}/' if not results_in_new_mujoco_main else f'/experiments/TD3/new_mujoco_main/{domain}/'


    #codes = ['code: 0', 'code: 2', 'code: 3', 'code: 5',  'code: 21' ,'code: 6', 'code: 7']
    codes = ['code: 0', 'code: 2', 'code: 3', 'code: 5', 'code: 21' , 'code: 7']
    values, rewards_per_code = mean_values(codes, path, seeds)
    print_results(results = values)
    print('______________________________')

    
    create_graph_2(rewards_per_code, codes, domain, seeds = seeds)