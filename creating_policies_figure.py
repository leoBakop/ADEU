import numpy as np
import matplotlib.pyplot as plt
from process_wandb_data import exponential_smoothing
def load(path):   
    exploration_strategy = np.load(path+'exploration_strategy.npy')
    policy = np.load(path+'policy.npy')
    variance = np.load(path+'variance.npy')
    return policy, exploration_strategy, variance





def smooth_interpolate_exact(x):
    """
    Takes discrete values [0,1,2,3] and returns a smooth, increasing function
    that exactly passes through these points at integer values.
    """
    from scipy.interpolate import interp1d
    
    # Define the exact points we want to hit
    x_points = [0, 1, 2, 3]
    y_points = [0, 1, 2, 3]
    
    # Create a cubic spline interpolation
    f = interp1d(x_points, y_points, kind='cubic', fill_value='extrapolate')
    
    return f(x)

if __name__ == '__main__':
    path = 'experiments/tabular/CustomFrozenLake/adeu_figure/episode900/' # frozen lake 900
    #path = 'experiments/TD3/InvertedPendulum-v5/adeu_figure/episode800/' # inverted pendulum 700 or 800
    policy, exploration_strategy, variance = load(path)
    
    # Example arrays (replace with your actual data)
    x_axis = np.arange(len(policy))  # or use range(len(policy))
    policy = exponential_smoothing(policy,x_axis, smoothing_param=0)
    exploration_strategy = exponential_smoothing(exploration_strategy,x_axis, smoothing_param=0)
    #variance = exponential_smoothing(variance,x_axis, smoothing_param=0.1)
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed


    # First subplot (top)
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    plt.plot(x_axis, policy, label=r"Policy $\pi(s)$", color="blue")
    plt.plot(x_axis, exploration_strategy, label=r"Exploration Strategy $N(\pi(s), g(f(s)))$", color="red", linestyle="--")
    plt.ylabel("Action")
    plt.xlabel("Timesteps within a single episode")
    plt.title("Policy and Exploration actions selected during the 900th episode")
    plt.legend()
    plt.grid(True)

    split_idx = 420
    x_before = x_axis[:split_idx]
    x_after = x_axis[split_idx:]
    variance_before = variance[:split_idx]
    variance_after = variance[split_idx:]
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot

    # Plot the two segments with different colors
    plt.plot(x_before, variance_before, color="blue", linewidth= 3)
    plt.plot(x_after, variance_after, color="blue", linewidth= 3)

    # Second subplot (bottom)
    #plt.plot(x_axis, variance, label="Variance", color="green", linestyle=":")
    plt.title('Uncertainty g(f(s)) during the 900th episode')
    plt.xlabel("Timesteps within a single episode")
    plt.ylabel("g(f(s))")
    plt.grid(True)

    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig('experiments/tabular/CustomFrozenLake/adeu_figure/the_figre.png')
    plt.show()