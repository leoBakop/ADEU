import math
import pandas as pd
import numpy as np


CODES_MAP={'code: 0':'baseline',
           'code: 2': 'IM',
           'code: 3': 'Noisy Nets',
           'code: 5': 'UCB',
           'code: 21': 'ADEU+im',
           'code: 6': 'ADEU+ensemble',
           
           'code: 1': 'ADEU'}


def mean_values(df, codes):
    values = {}
    extended_codes = list([c+' - cumm_evaluation_reward' for c in codes])
    for i,c in enumerate(extended_codes):
        rewards = df[c]
        rewards = exponential_smoothing(y_values=rewards, x_values=df['Step'], smoothing_param=0.0)
        rewards = np.array(rewards)
        mean_reward = round(np.mean(rewards), 0)
        max_reward = round(np.max(rewards), 0)
        values[CODES_MAP[codes[i]]] = [mean_reward,max_reward]
    return values


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

    

if __name__=='__main__':
    file = 'wandb_results/new_custom_frozen_lake_cumulative_rewards.csv'
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)
    codes = ['code: 0', 'code: 2', 'code: 3', 'code: 5',  'code: 21' ,'code: 6']
    codes = ['code: 0', 'code: 2', 'code: 5',  'code: 1' ]
    values = mean_values(df, codes)
    print_results(results = values)
    print('______________________________')

