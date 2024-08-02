import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
file_path = 'experiments/naf/Walker2d-v4/Bolz/seed_0/stored_data/variance_of_47500th_episode.npy'
#file_path = 'experiments/naf/Walker2d-v4/ade/seed_0/stored_data/variance_of_47500th_episode.npy'
data = np.load(file_path)[:120]

# Check the shape and type of the data
print(f'Data shape: {data.shape}')
print(f'Data type: {data.dtype}')

# Moving average function
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Smooth the data
window_size = 15
smoothed_data = moving_average(data, window_size)

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(range(window_size - 1, len(data)), smoothed_data, label='Smoothed Data', color='red')
plt.ylim(0, 2.6)
plt.title(r'The value of $P^{-1}$ during a training episode (47.500th episode in the Walker-2d domain)')
plt.xlabel('timesteps')
plt.ylabel(r'$P^{-1}$')
plt.grid(True)
plt.show()
