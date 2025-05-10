import math

import gymnasium as gym
from typing import Optional
import tkinter as tk
import threading
import pickle

# Used for Deep Q Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from itertools import count
import skimage.measure

from environment import Environment
from pigeon import Pigeon

X_SIZE = 1000
Y_SIZE = 1000

#TODO This is running and learning, but the observation space needs to include geomag, towns, cities and villages. But only from the pigeons view.
# I think the best way to do this would be to have full lists of the villages, towns etc. and then have binary 1/0 for if that specific one is in the pigeons view.
# The reward function also needs to then take these values into account.

# Created following the guidance of https://gymnasium.farama.org/introduction/create_custom_env/
# TODO REFERENCE?
class DQN(gym.Env):
    def __init__(self):
        self.window = tk.Tk()
        self.window.resizable(False, False)

        # My original environment class - that is being adapted to a Gymnasium environment
        self.env_orig = Environment()

        # Define the agent and target location
        self.canvas, self.passive_objects, self.active_objects, self.geomag_map = self.env_orig.initialise_environment(self.window, X_SIZE, Y_SIZE)
        self.pigeon = Pigeon("Pigeon1", X_SIZE, Y_SIZE, self.passive_objects, self.active_objects, self.geomag_map)
        self.pigeon.drawPigeon(self.canvas)

        # Sets the pigeon location and the loft, updated in step.
        self._agent_location = [self.pigeon.x, self.pigeon.y]
        loft = [f for f in self.active_objects if f.getClass() == "Loft"][0]
        self._target_location = [loft.x, loft.y]

        # Used to ensure that clicking anywhere on the map will move the pigeon to that location.
        self.window.bind("<Button-1>", self.click_handler)

        # THIS OBSERVATION SPACE IS REQUIRED FOR GYM IN THE INIT, BUT IN THE REST OF THE CODE IT IS COMPLETELY IGNORED.
        self.observation_space = gym.spaces.Dict(
            {
                "all": gym.spaces.Box(0, 30, shape=(1000,1000)),
                "pigeon": gym.spaces.Box(0, 30, shape=(1000,1000)),
                "loft": gym.spaces.Box(0, 30, shape=(1000,1000)),
                "geomag": gym.spaces.Box(-10000, 10000, shape=(100,100))
            }
        )

        self.action_space = gym.spaces.Discrete(360)

        t1 = threading.Thread(target=self.solve)
        t1.start()

        self.window.mainloop()

    # def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
    #     # We need the following line to seed self.np_random
    #     super().reset(seed=seed) # taken from the website, unsure currently if necessary for me
    #
    #     # Run these to reset the environment and the pigeon locations
    #     self.canvas.destroy()
    #     self.canvas, self.passive_objects, self.active_objects, self.geomag_map = self.env_orig.initialise_environment(self.window, X_SIZE, Y_SIZE)
    #     self.pigeon = Pigeon("Pigeon1", X_SIZE, Y_SIZE, self.passive_objects, self.active_objects, self.geomag_map)
    #
    #     # Define these as the new locations.
    #     self._agent_location = [self.pigeon.x, self.pigeon.y]
    #     loft = [f for f in self.active_objects if f.getClass() == "Loft"][0]
    #     self._target_location = [loft.x, loft.y]
    #
    #     # Get the observations to return the newest observations after reset
    #     observations = self.get_observations()
    #
    #     return observations

    # todo testing a new reset function that only resets the pigeons location, no other environment values.
    # uncomment above if need to change.
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed) # taken from the website, unsure currently if necessary for me

        self.pigeon = Pigeon("Pigeon1", X_SIZE, Y_SIZE, self.passive_objects, self.active_objects, self.geomag_map)

        # Define these as the new locations.
        self._agent_location = [self.pigeon.x, self.pigeon.y]

        # Get the observations to return the newest observations after reset
        observations = self.get_observations()

        return observations

    def step(self, action):
        # Sets the velocity values used to determine movement
        hypotenuse = 1000 * (0.0138888889/30) # This ensures constant speed. 0.0138... is the real value of 50mph converted to mps and scaled with 30.
        self.pigeon.xv = math.sin(math.radians(action)) * hypotenuse
        self.pigeon.yv = math.cos(math.radians(action)) * hypotenuse

        # Before the step function actually updates any values, the previous location is saved, as it is used in the reward function
        prev_loc = self.pigeon.dist_from_loft

        # Runs the movement and changes the xvalue, unless the pigeon is dead already
        if self.pigeon.alive:
            self.pigeon.move(self.canvas)

        # Checks that the pigeon is alive, and not in a predator area, this function also works out probability of death and
        self.pigeon.pigeonInDanger(self.active_objects)

        # If alive, updates vision and geomagnetic location.
        if self.pigeon.alive:
            self.pigeon.pigeon_vision = self.pigeon.getVision(self.passive_objects, self.active_objects)
            self.pigeon.current_geomag_loc = self.pigeon.updateCurrentGeomag(self.geomag_map)

            # Updates the performance metrics
            self.pigeon.no_moves += 1
            loft = [x for x in self.active_objects if x.getClass() == "Loft"][0]  # This retrieves the loft instance (There should only be one)
            self.pigeon.dist_from_loft = math.sqrt((self.pigeon.x - loft.x) ** 2 + (self.pigeon.y - loft.y) ** 2)

            # Change the internal values for gym env.
            self._agent_location = [self.pigeon.x, self.pigeon.y]

            # End if pigeon is in same location as loft
            if self.pigeon.x == loft.x and self.pigeon.y == loft.y:
                terminated = True
            else:
                terminated = False
        else:
            terminated = True

        # Get the updated observations and reward for the step.
        observations = self.get_observations()
        reward = self.reward_function(prev_loc)

        return observations, reward, terminated

    # Used to get the current location of the pigeon and the loft (observations) - always returns a list
    def get_observations(self):
        # This returns matrices of the pigeon view, and the loft memory
        pigeon_view, loft_view = self.generate_views()

        # Max Pooling to reduce the size of the input space for DQN
        pigeon_view = skimage.measure.block_reduce(pigeon_view, (3,3), np.max)
        loft_view = skimage.measure.block_reduce(loft_view, (3,3), np.max)

        # Geomagnetic values for the pigeon and the loft.
        pigeon_geomag = self.pigeon.current_geomag_loc
        loft_geomag = self.pigeon.geomag_loft

        # Currently outputting only the difference between the geomagnetic values.
        output = list(self.geomag_diff(pigeon_geomag, loft_geomag)) + list(pigeon_view.flatten()) + list(loft_view.flatten())

        return output

    # This finds matrices of the areas surrounding the pigeon and the loft. As it is necessary for the input to be in an understandable, and consistent size.
    def generate_views(self):
        # Either create the environment matrix, or load it from a pickle, to save processing time.
        # TODO COME BACK HERE, BECAUSE THIS SOLUTION WILL NOT WORK FOR CHANGING ENVIRONMENTS
        try:
            with open("data/enviro_info/enviro_matrix.pkl", "rb") as f:
                environ_matrix = pickle.load(f)
        except (FileNotFoundError, EOFError):
            environ_matrix = self.env_orig.create_matrix(X_SIZE, Y_SIZE)
            with open("data/enviro_info/enviro_matrix.pkl", "wb") as f:
                pickle.dump(environ_matrix, f)

        # This finds the area in the full matrix that just the pigeon can see.
        pigeon_matrix = np.zeros((X_SIZE, Y_SIZE))
        mask = self.env_orig.np_circle_func(self.pigeon.x, self.pigeon.y, self.pigeon.viewing_distance, environ_matrix)
        pigeon_matrix[mask] = environ_matrix[mask.T] # Generate the masked version of the full environment, showing only the circular pigeons view.

        loft_matrix = np.zeros((X_SIZE, Y_SIZE))
        mask = self.env_orig.np_circle_func(self._target_location[0], self._target_location[1],
                                            self.pigeon.memory_radius, environ_matrix)
        loft_matrix[mask] = environ_matrix[mask.T] # Generate the masked version of the full environment, showing only the circular loft memory

        # Get radius distances for the loft and the pigeon.
        viewing_distance = int(self.pigeon.viewing_distance)
        memory_radius = int(self.pigeon.memory_radius)

        view_size_pigeon = viewing_distance * 2
        view_size_loft = memory_radius * 2

        # Initialise matrices for the pigeon view size and loft memory.
        pigeon_view = np.zeros([view_size_pigeon, view_size_pigeon])
        loft_view = np.zeros([view_size_loft, view_size_loft])

        # Round and convert positions to int as is necessary, due to them being float values.
        pige_x = int(round(self.pigeon.x))
        pige_y = int(round(self.pigeon.y))
        loft_x = int(round(self._target_location[0]))
        loft_y = int(round(self._target_location[1]))

        # Helper function, to take the important information from the large matrices, and put it in the "_view" oens.
        def find_view(matrix, center_x, center_y, radius, view_matrix):
            matrix_h, matrix_w = matrix.shape

            # Find either the real values in the main matrix, must be highest or 0, as otherwise it can return incorrect values.
            # e.g. -12 will return from wrong side of matrix
            x_min = max(center_x - radius, 0)
            x_max = min(center_x + radius, matrix_h)
            y_min = max(center_y - radius, 0)
            y_max = min(center_y + radius, matrix_w)

            # Placement in the view matrix.
            vx_min = radius - (center_x - x_min)
            vx_max = vx_min + (x_max - x_min)
            vy_min = radius - (center_y - y_min)
            vy_max = vy_min + (y_max - y_min)

            # Insert cropped region into fixed-size view
            view_matrix[vx_min:vx_max, vy_min:vy_max] = matrix[x_min:x_max, y_min:y_max]

            return view_matrix

        # Find the new view matrices
        pigeon_view = find_view(pigeon_matrix, pige_x, pige_y, viewing_distance, pigeon_view)
        loft_view = find_view(loft_matrix, loft_x, loft_y, memory_radius, loft_view)

        return pigeon_view, loft_view

    # This function is used to return the difference between the geomagnetic values, for the observations of the agent.
    def geomag_diff(self, pige_geo, loft_geo):
        output = []
        for x in range(0, len(pige_geo)):
            # TODO MAYBE TRY SCALING THESE VALUES???? lIKE MINMAX OR SOME SHIT
            output.append(pige_geo[x] - loft_geo[x])

        return output

    # TODO FIX REWARD FUNCTION
    # okay so this reward function works a lot better than the others, but wtf is the lower=better bs???
    def reward_function(self, previous_location):
        if self.pigeon.alive:
            if self.pigeon.dist_from_loft > previous_location:
                reward = 0
            else:
                reward = 1
        else:
            reward = 1

        print(reward)
        return reward

    # This is added here so that at any point, you can click an area on the map and the pigeon will move there. Necessary for development.
    def click_handler(self, event):
        if event.num == 1:
            self.pigeon.x = event.x
            self.pigeon.y = event.y

    # Code taken from: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    # TODO REFERENCE
    def solve(self):
        # Hyperparameters using in DQN:
        BATCH_SIZE = 1028 # BATCH_SIZE is the number of transitions sampled from the replay buffer
        GAMMA = 0.99  # GAMMA is the discount factor
        EPS_START = 0.9  # EPS_START is the starting value of epsilon
        EPS_END = 0.05  # EPS_END is the final value of epsilon
        EPS_DECAY = 1000  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        TAU = 0.005  # TAU is the update rate of the target network
        LR = 0.001  # LR is the learning rate of the ``AdamW`` optimizer

        # Set the device to run on GPU, if applicable
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        state = self.get_observations()
        state_size = len(state)
        action_size = self.action_space.n

        print(state_size)

        iterations = 0

        # Initialise the Networks
        policy_net = DeepQLearningNetwork(state_size, action_size).to(device)
        target_net = DeepQLearningNetwork(state_size, action_size).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)

        def select_action(inp_state):
            nonlocal iterations, policy_net, device
            nonlocal BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * iterations / EPS_DECAY)
            iterations += 1

            # THIS IS NECESSARY TO CHECK AND CHANGE if I edit any values later. see the conditions in line 150
            # print(policy_net(inp_state).max(1).indices.item())
            # print(policy_net(inp_state).max(1).indices)
            # print(policy_net(inp_state).max(1).indices.view([0,1]))
            # print(policy_net(inp_state).max(1))

            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return torch.tensor([[policy_net(inp_state).max(1).indices.item()]], device=device) # This originally said "policy_net(inp_state).max(1).indices.view([1,1])"
            else:
                return torch.tensor([[self.action_space.sample()]], device=device, dtype=torch.long)

        def optimize_model():
            nonlocal memory, policy_net, optimizer, target_net, device
            nonlocal BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU

            if len(memory) < BATCH_SIZE:
                return
            transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()

        # Hyperparameter
        num_episodes = 50

        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            state = self.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)
            iterations = 0
            for t in count():
                action = select_action(state)
                observation, reward, terminated = self.step(action.item())
                reward = torch.tensor([reward], device=device)
                if iterations >= 10000:
                    terminated = True
                    reward = torch.tensor([0], device=device)
                done = terminated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, device=device, dtype=torch.float32).flatten().unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                                1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    break


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Class used to hold the memory for the previous movements of the agent
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# The network layout used for training
class DeepQLearningNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DeepQLearningNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
