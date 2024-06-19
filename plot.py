import numpy as np

import matplotlib.pyplot as plt


def moving_average(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size
if __name__ == '__main__':
    horizon = 575+50
    avg_speed = np.zeros(horizon)
    avg_reward = np.zeros(horizon)
    avg_coll = np.zeros(horizon)

    avg_speed_2 = np.zeros(horizon)
    avg_reward_2 = np.zeros(horizon)
    avg_coll_2 = np.zeros(horizon)

    samples = list([i for i in range(10)  ])

    for i in samples:
        file_path = f"experiments/naf/c_1_speed_boost_55_45/second_False_per_True_pse_True/fourth_implementation/upd_1/evaluation_{i}/stored_data/"
        reward = np.load(file_path+"reward.npy")[:625]
        speed = np.load(file_path+"speed.npy")[:625]
        coll = np.load(file_path+"coll.npy")[:625]
        avg_reward+=reward/len(samples)
        avg_speed+=speed/len(samples)
        avg_coll+=coll/len(samples)
    avg_speed = 36*avg_speed - 20
    print("----------------------PER + PSNE")
    print(f"avg reward is {np.mean(avg_reward[575:625])}")
    print(f"avg speed is {np.mean(avg_speed[575:625])}")
    print(f"avg coll is {np.mean(avg_coll[575:625])}")


    samples = list([i for i in range(10)] )
    for i in samples:
        file_path = f"experiments/naf/c_1_speed_boost_55_45/second_False_per_True_pse_False/upd_1/evaluation_{i}/stored_data/"
        reward = np.load(file_path+"reward.npy")[:625]
        speed = np.load(file_path+"speed.npy")[:625]
        coll = np.load(file_path+"coll.npy")[:625]
        avg_reward_2+=reward/len(samples)
        avg_speed_2+=speed/len(samples)
        avg_coll_2+=coll/len(samples)
    avg_speed_2 = 36*avg_speed_2 - 20

    print("-----------------PER----------------")
    print(f"avg reward is {np.mean(avg_reward_2[575:625])}")
    print(f"avg speed is {np.mean(avg_speed_2[575:625])}")
    print(f"avg coll is {np.mean(avg_coll_2[575:625])}")

    window_size = 10
    avg_reward = moving_average(avg_reward, window_size)
    avg_speed = moving_average(avg_speed, window_size)
    avg_coll = moving_average(avg_coll, window_size)

    avg_reward_2 = moving_average(avg_reward_2, window_size)
    avg_speed_2 = moving_average(avg_speed_2, window_size)
    avg_coll_2 = moving_average(avg_coll_2, window_size)
    
    plt.figure(1)
    plt.title("With vs. Without PSNE")
    plt.xlabel("Episode") 
    plt.ylabel("Cumulative Reward")
    plt.plot(np.arange(len(avg_reward)),avg_reward, label = "PER + PSNE")
    plt.plot(np.arange( len(avg_reward_2)),avg_reward_2, label = "PER")
    
    plt.grid()
    plt.legend()
    plt.savefig("experiments/figures/PER_VS_PER_PSNE/reward")
    plt.show()

    plt.figure(2)
    plt.title("With vs. Without PSNE")
    plt.xlabel("Episode") 
    plt.ylabel("Speed Deviation")
    plt.plot(np.arange(len(avg_speed)),avg_speed, label = "PER + PSNE")
    plt.plot(np.arange(len(avg_speed)),avg_speed_2, label = "PER")
    
    plt.grid()
    plt.legend()
    plt.savefig("experiments/figures/PER_VS_PER_PSNE/speed")
    plt.show()

    plt.figure(3)
    plt.title("With vs. Without PSNE")
    plt.xlabel("Episode") 
    plt.ylabel("Number of Collisions")
    plt.axhline(y=0.5, color='red', linestyle='--', label="DDPG's performance")
    plt.plot(np.arange(len(avg_speed)),avg_coll, label = "PER + PSNE")
    plt.plot(np.arange(len(avg_speed)),avg_coll_2, label = "PER")
    
    plt.grid()
    plt.legend()
    plt.savefig("experiments/figures/PER_VS_PER_PSNE/coll")
    plt.show()