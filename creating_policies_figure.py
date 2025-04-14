import numpy as np
import matplotlib.pyplot as plt

def load(path):   
    exploration_strategy = np.load(path+'exploration_strategy.npy')
    policy = np.load(path+'policy.npy')
    variance = np.load(path+'variance.npy')
    return policy, exploration_strategy, variance


if __name__ == '__main__':
    path = 'experiments/TD3/InvertedPendulum-v5/adeu_figure/episode100/'
    policy, exploration_strategy, variance = load(path)

    # Example arrays (replace with your actual data)
    x_axis = np.arange(len(policy))  # or use range(len(policy))

    # Plot all arrays on the same figure
    plt.figure()
    plt.plot(x_axis, policy, label="Policy", color="blue")
    plt.plot(x_axis, exploration_strategy, label="Exploration Strategy", color="red", linestyle="--")
    plt.plot(x_axis, variance, label="Variance", color="green", linestyle=":")

    # Customize the plot
    plt.xlabel("Steps (Index)")
    plt.ylabel("Value")
    plt.title("Policy, Exploration, and Variance Over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()