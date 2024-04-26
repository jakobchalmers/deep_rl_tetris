import numpy as np
import random
import math
import h5py
import torch
import torch.nn as nn
import random

# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.


class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self, alpha, epsilon, episode_count):
        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode = 0
        self.episode_count = episode_count

    def fn_init(self, gameboard):
        self.gameboard = gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions and Q-table and storage for the rewards
        # This function should not return a value, store Q table etc as attributes of self

        self.reward_tots = np.zeros((self.episode_count))

        if self.gameboard.tile_size == 1:
            self.N_rot = 1
        elif self.gameboard.tile_size >= 2:
            self.N_rot = 4

        num_boards = 2 ** (gameboard.N_row * gameboard.N_col)
        num_tiles = len(gameboard.tiles)
        self.q_values = np.zeros(
            (num_boards, num_tiles, gameboard.N_col, self.N_rot)
        )  # (state1, state2, action1, action2)


        # Useful variables:
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.episode_count' the total number of episodes in the training

    def fn_load_strategy(self, strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-table (to Q-table of self) from the input parameter strategy_file (used to test how the agent plays)

        with h5py.File(strategy_file, "r") as f:
            self.q_values = f["q_values"][:]

    def fn_read_state(self):

        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as an integer entry in the Q-table
        # This function should not return a value, store the state as an attribute of self

        gameboard_size = self.gameboard.N_row * self.gameboard.N_col
        bin2dec_arr = 2 ** (gameboard_size - np.arange(1, gameboard_size + 1))
        flat_board = self.gameboard.board.flatten()
        flat_board[flat_board < 0] = 0
        board_state = (flat_board * bin2dec_arr).sum()

        self.state = np.array([board_state, self.gameboard.cur_tile_type], dtype=int)

        # Useful variables:
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def fn_select_action(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the Q-table or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        invalid_q = - np.inf

        board_state = self.state[0]
        tile_state = self.state[1]
        for x_action in range(0, self.gameboard.N_col):
            for rot_action in range(0, self.N_rot):

                unallowed = self.gameboard.fn_move(x_action, rot_action)
                if unallowed:
                    self.q_values[board_state, tile_state, x_action, rot_action] = invalid_q

        state_values = self.q_values[board_state, tile_state, :, :].copy()
        flat_state_values = state_values.flatten()

        indicator = np.random.rand(1)[0]
        if indicator <= self.epsilon:
            allowed_actions = np.where(flat_state_values > invalid_q)[0]

            action = np.random.choice(allowed_actions, size=1)[0]
        else:
            max_value = np.max(flat_state_values)
            greedy_actions = np.where(flat_state_values == max_value)[0]
            action = np.random.choice(greedy_actions, size=1)[0]

        x_action = action // state_values.shape[1]
        rot_action = action % state_values.shape[1]

        self.gameboard.fn_move(x_action, rot_action)
        self.action = np.array([x_action, rot_action])

        # Useful variables:
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self, old_state, reward):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q table using state and action stored as attributes in self and using function arguments for the old state and the reward
        # This function should not return a value, the Q table is stored as an attribute of self

        q_value = self.q_values[old_state[0], old_state[1], self.action[0], self.action[1]].copy()

        max_q_value = (1 - self.gameboard.gameover) * np.max(self.q_values[self.state[0], self.state[1], :, :].flatten())

        self.q_values[old_state[0], old_state[1], self.action[0], self.action[1]] = q_value + self.alpha * (reward + max_q_value - q_value)

        # Useful variables:
        # 'self.alpha' learning rate

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1
            if self.episode % 100 == 0:
                print(
                    "episode "
                    + str(self.episode)
                    + "/"
                    + str(self.episode_count)
                    + " (reward: ",
                    str(
                        np.sum(
                            self.reward_tots[range(self.episode - 100, self.episode)]
                        )
                    ),
                    ")",
                )
            if self.episode % 1000 == 0:
                saveEpisodes = [
                    1000,
                    2000,
                    5000,
                    10000,
                    20000,
                    50000,
                    100000,
                    200000,
                    500000,
                    1000000,
                ]
                if self.episode in saveEpisodes:
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
                    with h5py.File("strategy.h5", "w") as f:
                        f.create_dataset("q_values", data=self.q_values)
                        f.create_dataset("reward_tots", data=self.reward_tots)

            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:

            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()
            old_state = np.copy(self.state)

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += reward


            # Read the new state
            self.fn_read_state()
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state, reward)


class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(
        self,
        alpha,
        epsilon,
        epsilon_scale,
        replay_buffer_size,
        batch_size,
        sync_target_episode_count,
        episode_count,
    ):
        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_scale = epsilon_scale
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.sync_target_episode_count = sync_target_episode_count
        self.episode = 0
        self.episode_count = episode_count

    def fn_init(self, gameboard):
        self.gameboard = gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks (one for calculating actions and one target network), experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self

        # Useful variables:
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

        class QNetwork(nn.Module):
            def __init__(
                self, state_dim, hidden_features_1, hidden_features_2, num_action
            ):
                super().__init__()

                self.layer1 = nn.Linear(state_dim, hidden_features_1)
                self.activation1 = nn.ReLU()

                self.layer2 = nn.Linear(hidden_features_1, hidden_features_2)
                self.activation2 = nn.ReLU()

                self.layer3 = nn.Linear(hidden_features_2, num_action)

            def forward(self, x):
                x = self.layer1(x)
                x = self.activation1(x)

                x = self.layer2(x)
                x = self.activation2(x)

                x = self.layer3(x)  # q-values

                return x

        state_dim = gameboard.N_row * gameboard.N_col + len(gameboard.tiles)
        self.N_rotations = 4
        self.num_actions = gameboard.N_col * self.N_rotations

        self.online_network = QNetwork(
            state_dim=state_dim,
            hidden_features_1=64,
            hidden_features_2=64,
            num_action=self.num_actions,
        )

        # self.target_network = self.online_network
        self.target_network = QNetwork(
            state_dim=state_dim,
            hidden_features_1=64,
            hidden_features_2=64,
            num_action=self.num_actions,
        )
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.alpha
        )

        self.environment_step = 0
        self.exp_buffer = []

        self.reward_tots = np.zeros(self.episode_count)

    def fn_load_strategy(self, strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self

        # Useful variables:
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

        board_state = torch.from_numpy(self.gameboard.board.flatten())

        tile_state = torch.zeros(len(self.gameboard.tiles))
        tile_state[self.gameboard.cur_tile_type] = 1
        self.state = torch.cat((board_state, tile_state))

    def fn_select_action(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables:
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not
        invalid_q = -torch.inf

        q_values = self.online_network(self.state)
        # Create a tensor to store the validity of each action
        # for x_action in range(0, self.gameboard.N_col):
        #     for rotation_action in range(0, self.N_rotations):
        #         unallowed = self.gameboard.fn_move(x_action, rotation_action)
        #         if unallowed:
        #             q_values[x_action * self.N_rotations + rotation_action] = invalid_q

        self.epsilon_E = max(
            self.epsilon, (1 - self.episode / self.epsilon_scale)
        )  # TODO: check if this is correct

        indicator = np.random.rand()
        if indicator <= self.epsilon_E:
            # allowed_actions = torch.where(q_values != invalid_q)[0]
            # random_index = np.random.randint(0, len(allowed_actions))
            # index = allowed_actions[random_index]
            # if self.episode % 10 == 0:
            #     print("Random action")
            index = np.random.randint(0, len(q_values))
        else:
            max_q_value = torch.max(q_values)
            greedy_actions = torch.where(q_values == max_q_value)[0]
            random_index = np.random.randint(0, len(greedy_actions))
            index = greedy_actions[random_index]

        self.action = index
        tile_x = self.action // self.N_rotations
        tile_orientation = self.action % self.N_rotations
        self.action_invalid = self.gameboard.fn_move(tile_x, tile_orientation)

    def fn_reinforce(self, batch):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables:
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

        self.online_network.train()

        batch = torch.stack(batch)

        old_states = batch[:, 0, :]
        actions = batch[:, 1, 0].to(torch.int64).unsqueeze(1)
        rewards = batch[:, 2, 0]
        states = batch[:, 3, :]
        is_terminal = batch[:, 4, 0].to(torch.bool)

        online_old_q_values = self.online_network(old_states)
        online_old_q_values = online_old_q_values.gather(1, actions).squeeze(1)
        target_q_values = self.target_network(states)
        DISCOUNT_FACTOR = 0.99
        y = rewards + DISCOUNT_FACTOR*(~is_terminal * torch.max(target_q_values, dim=1)[0])

        loss = self.criterion(online_old_q_values, y.detach()) # TODO: check if detach is correct
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1

            if self.episode % 10 == 0:
            # if self.episode % 100 == 0:
                print(
                    "episode "
                    + str(self.episode)
                    + "/"
                    + str(self.episode_count)
                    + " (reward: ",
                    str(
                        np.sum(
                            self.reward_tots[range(self.episode - 100, self.episode)]
                        )
                    ),
                    ")",
                )
            if self.episode % 1000 == 0:
                saveEpisodes = [
                    1000,
                    2000,
                    5000,
                    10000,
                    20000,
                    50000,
                    100000,
                    200000,
                    500000,
                    1000000,
                ]
                if self.episode in saveEpisodes:
                    with h5py.File("rewards.h5", "w") as f:
                        f.create_dataset(f"reward_tots", data=self.reward_tots)
                    torch.save(self.online_network.state_dict(), "Q_network.pth")
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and (
                    (self.episode % self.sync_target_episode_count) == 0
                ):
                    self.target_network.load_state_dict(self.online_network.state_dict())
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer
            old_state = self.state.clone()

            # Drop the tile on the game board
            # reward = self.gameboard.fn_drop()
            INVALID_ACTION_PUNISHMENT = - 90
            # reward += self.action_invalid * INVALID_ACTION_PUNISHMENT
            if self.action_invalid:
                reward = INVALID_ACTION_PUNISHMENT
            else:
                reward = self.gameboard.fn_drop()


            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer
            state_dimension = self.state.shape[0]
            is_terminal = self.gameboard.gameover*torch.ones(state_dimension)
            quadruplet = torch.stack(
                (
                    old_state,
                    torch.ones(state_dimension) * self.action,
                    torch.ones(state_dimension) * reward,
                    self.state,
                    is_terminal,
                )
            )

            self.exp_buffer.append(quadruplet)

            if len(self.exp_buffer) > self.replay_buffer_size:  # >= ??
                self.exp_buffer.pop(0)
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets
                batch = random.sample(self.exp_buffer, self.batch_size)
                self.fn_reinforce(batch)

class THumanAgent:
    def fn_init(self, gameboard):
        self.episode = 0
        self.reward_tots = [0]
        self.gameboard = gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self, pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots = [0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(
                            self.gameboard.tile_x,
                            (self.gameboard.tile_orientation + 1)
                            % len(self.gameboard.tiles[self.gameboard.cur_tile_type]),
                        )
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(
                            self.gameboard.tile_x - 1, self.gameboard.tile_orientation
                        )
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(
                            self.gameboard.tile_x + 1, self.gameboard.tile_orientation
                        )
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode] += self.gameboard.fn_drop()