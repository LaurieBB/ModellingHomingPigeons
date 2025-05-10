import math
import gymnasium as gym
from typing import Optional
import tkinter as tk
import threading
import pickle
import skimage.measure
import numpy as np

from environment import Environment
from pigeon import Pigeon

X_SIZE = 1000
Y_SIZE = 1000

# Created following the guidance of https://gymnasium.farama.org/introduction/create_custom_env/
# However, this is all my own work as this is simply guidance on formatting.

# This class converts my own custom environment into one that is gymnasium compatible. This makes it easier to use with PyTorch etc.
class GymEnvironment(gym.Env):
    def __init__(self, window):
        # My original environment class - that is being adapted to a Gymnasium environment
        self.env_orig = Environment()

        # Define the agent and target location
        self.canvas, self.passive_objects, self.active_objects, self.geomag_map = self.env_orig.initialise_environment(window, X_SIZE, Y_SIZE)
        self.pigeon = Pigeon("Pigeon1", X_SIZE, Y_SIZE, self.passive_objects, self.active_objects, self.geomag_map)
        self.pigeon.drawPigeon(self.canvas)

        # Sets the pigeon location and the loft, updated in step.
        self._agent_location = [self.pigeon.x, self.pigeon.y]
        loft = [f for f in self.active_objects if f.getClass() == "Loft"][0]
        self._target_location = [loft.x, loft.y]

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
            loft = [x for x in self.active_objects if x.getClass() == "Loft"][0]  # This retrieves the loft instance (There should only be one)
            self.pigeon.dist_from_loft = math.sqrt((self.pigeon.x - loft.x) ** 2 + (self.pigeon.y - loft.y) ** 2)

            # Change the internal values for gym env.
            self._agent_location = [self.pigeon.x, self.pigeon.y]

            # End if pigeon is in same location as loft, it may seem like a low value, however the pigeon is also
            if self.pigeon.dist_from_loft <= 20:
                terminated = True
                truncated = False
            # If the pigeon has done too many moves
            elif self.pigeon.no_moves >= 10000:
                terminated = False
                truncated = True
            # No issues.
            else:
                terminated = False
                truncated = False
        else:
            truncated = True
            terminated = False

        # Get the updated observations and reward for the step.
        observations = self.get_observations()
        reward = self.reward_function(prev_loc)

        # Set reward for if it is finished, terminated=reached goal, truncated=died/too many moves
        if terminated:
            reward = 10
        if truncated:
            reward = -10

        return observations, reward, terminated, truncated

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