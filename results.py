# %% Setup
import h5py
import matplotlib.pyplot as plt
import numpy as np

# %% 1a

strategy_file = "strategy_1a.h5"
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

plt.title("1a")
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()

plt.show()

# %% 1b

strategy_file = "strategy_1b.h5"
with h5py.File(strategy_file, "r") as f:
    q_values = f["q_values"][:]
    reward_tots = f["reward_tots"][:]

group_size = 100
reward_groups = np.vstack(np.array_split(reward_tots, len(reward_tots) / group_size))
average_rewards = np.mean(reward_groups, axis=1)
average_rewards = np.concatenate((np.array([-100]), average_rewards))
group_indeces = np.arange(0, reward_tots.shape[0] + 1, group_size)

step = 1
selected_episode_numbers = np.arange(0, len(reward_tots), step)
plt.plot(selected_episode_numbers, reward_tots[::step], color="red", label="episode reward")
plt.plot(group_indeces, average_rewards, color="blue", label="average reward")

plt.title("1b")
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()

plt.show()

    
# %% 1c

strategy_file = "strategy_1c.h5"
with h5py.File(strategy_file, "r") as f:
    q_values = f["q_values"][:]
    reward_tots = f["reward_tots"][:]

group_size = 100
reward_groups = np.vstack(np.array_split(reward_tots, len(reward_tots) / group_size))
average_rewards = np.mean(reward_groups, axis=1)
average_rewards = np.concatenate((np.array([-100]), average_rewards))
group_indeces = np.arange(0, reward_tots.shape[0] + 1, group_size)

step = 1
selected_episode_numbers = np.arange(0, len(reward_tots), step)
plt.plot(selected_episode_numbers, reward_tots[::step], color="red", label="episode reward")
plt.plot(group_indeces, average_rewards, color="blue", label="average reward")

plt.title("1c")
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()

plt.show()

# %% 2a

strategy_file = "strategy_2a.h5"
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

plt.title("2a")
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()

plt.show()