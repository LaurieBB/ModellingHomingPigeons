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
from gym_environment import GymEnvironment

X_SIZE = 1000
Y_SIZE = 1000

#Todo This is running and learning but more possible ideas:
# Could have the inputs of towns/villages etc. as a list of all possible and binary values if in view or not. This would make it more recognisable?
# Could also try scaling values like the geomagnetic ones, might make it more interpretable.
# Could also try and change the reward function to make it more complex.

class DQN():
    def __init__(self):
        self.window = tk.Tk()
        self.window.resizable(False, False)

        self.env = GymEnvironment(self.window)

        # Used to ensure that clicking anywhere on the map will move the pigeon to that location.
        self.window.bind("<Button-1>", self.env.click_handler)

        t1 = threading.Thread(target=self.solve)
        t1.start()

        self.window.mainloop()


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

        state = self.env.get_observations()
        state_size = len(state)
        action_size = self.env.action_space.n

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
                return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)

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
            criterion = nn.SmoothL1Loss() # TODO MAYBE TRY CHANGING LOSS FUNCTION
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
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)
            for t in count():
                action = select_action(state)
                observation, reward, terminated, truncated = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)

                done = terminated or truncated

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
