import h5py
import matplotlib.pyplot as plt
import numpy as np

strategy_file = "strategy.h5"
with h5py.File(strategy_file, "r") as f:
    q_values = f["q_values"][:]
    reward_tots = f["reward_tots"][:]



group_size = 100
reward_groups = np.vstack(np.array_split(reward_tots, len(reward_tots) / group_size))
average_rewards = np.mean(reward_groups, axis=1)
average_rewards = np.concatenate((np.array([-100]), average_rewards))
group_indeces = np.arange(0, reward_tots.shape[0] + 1, group_size)



plt.plot(reward_tots, color="red", label="episode reward")
plt.plot(group_indeces, average_rewards, color="blue", label="average reward")

plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()

plt.show()

    