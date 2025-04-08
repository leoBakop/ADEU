import math
import pandas as pd
import numpy as np


CODES_MAP={'code: 0':'baseline',
           'code: 2': 'IM',
           'code: 3': 'Noisy Nets',
           'code: 5': 'UCB',
           'code: 21': 'ADEU+im',
           'code: 6': 'ADEU+ensemble'}


def mean_values(df, codes):
    values = {}
    extended_codes = list([c+' - cumm_evaluation_reward' for c in codes])
    for i,c in enumerate(extended_codes):
        rewards = df[c]
        rewards = exponential_smoothing(y_values=rewards, x_values=df['Step'], smoothing_param=0.99)
        rewards = np.array(rewards)
        mean_reward = round(np.mean(rewards), 0)
        max_reward = round(np.max(rewards), 0)
        values[CODES_MAP[codes[i]]] = [mean_reward,max_reward]
    return values


def print_results(results):
    print('Algorithm|  Mean Reward|   MAX reward ')
    for key, value in results.items():
        print(f'{key}| {value[0]} | {value[1]}' )



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

if __name__=='__main__':
    file = 'wandb_results/swimmer_cumulative_rewards.csv'
    898
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)
    codes = ['code: 0', 'code: 21', 'code: 2', 'code: 3', 'code: 5', 'code: 6' ]

    values = mean_values(df, codes)
    print_results(results = values)
