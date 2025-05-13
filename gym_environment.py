import math
import gymnasium as gym
from typing import Optional
import tkinter as tk
import threading
import pickle
import skimage.measure
import numpy as np
from numpy.linalg import norm

from environment import Environment, in_area
from pigeon import Pigeon

X_SIZE = 1000
Y_SIZE = 1000

# Created following the guidance of https://gymnasium.farama.org/introduction/create_custom_env/
# However, this is all my own work as this is simply guidance on formatting.

# This class converts my own custom environment into one that is gymnasium compatible. This makes it easier to use with PyTorch etc.
class GymEnvironment(gym.Env):
    def __init__(self, draw=False, window=None):
        # My original environment class - that is being adapted to a Gymnasium environment
        self.env_orig = Environment()

        # Have to customise depending on if I am running a final test or not
        if draw:
            # Define the agent and target location
            self.canvas, self.passive_objects, self.active_objects, self.geomag_map = self.env_orig.initialise_environment(window, X_SIZE, Y_SIZE)
            self.pigeon = Pigeon("Pigeon1", X_SIZE, Y_SIZE, self.passive_objects, self.active_objects, self.geomag_map)
            self.pigeon.drawPigeon(self.canvas)
            self.window = window
            self.draw = True
        else:
            self.passive_objects, self.active_objects, self.geomag_map = self.env_orig.initialise_environment_no_draw(X_SIZE, Y_SIZE)
            self.pigeon = Pigeon("Pigeon1", X_SIZE, Y_SIZE, self.passive_objects, self.active_objects, self.geomag_map)
            self.draw = False

        # Storing the pigeon start location
        self.pigeon_start_loc = [self.pigeon.x, self.pigeon.y]

        # Necessary to flag the first run only
        self.first_run = True # ONLY USE IF MATRIX OBSERVATIONS IS BEING USED

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

    # reset function, overriding the gymnasium required one, used to reset the agent to a random position in the map
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed) # taken from the website, unsure currently if necessary for me

        self.pigeon = Pigeon("Pigeon1", X_SIZE, Y_SIZE, self.passive_objects, self.active_objects, self.geomag_map)

        # Define these as the new locations.
        self.pigeon_start_loc = [self.pigeon.x, self.pigeon.y]
        self._agent_location = [self.pigeon.x, self.pigeon.y]
        loft = [f for f in self.active_objects if f.getClass() == "Loft"][0]
        self._target_location = [loft.x, loft.y]

        if self.draw:
            # Draw euclidian distance to the loft from pigeon start point
            self.canvas.delete("euc_line")
            self.canvas.create_line(self.pigeon_start_loc[0], self.pigeon_start_loc[1], self._target_location[0], self._target_location[1], dash=(5,1), fill="darkblue", tag="euc_line")
            self.canvas.create_oval(self.pigeon_start_loc[0] - 5, self.pigeon_start_loc[1] - 5, self.pigeon_start_loc[0] + 5, self.pigeon_start_loc[1] + 5, fill="darkblue", outline="black", tag="euc_line")

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
            if self.draw:
                self.pigeon.move(self.canvas)
            else:
                self.pigeon.move_no_draw()

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
                print("MY PIGEON IS HOME!!!")
                terminated = True
                truncated = False
            # If the pigeon has done too many moves
            elif self.pigeon.no_moves >= 5000:
                terminated = False
                truncated = True
            # No issues.
            else:
                terminated = False
                truncated = False
        else:
            terminated = False
            truncated = True

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

        # Below is the list observation space
        # pigeon_view = self.env_orig.obj_in_view(self.pigeon.x, self.pigeon.y, self.pigeon.viewing_distance)
        # loft_view = self.env_orig.obj_in_view(self._target_location[0], self._target_location[1], self.pigeon.memory_radius)
        # pigeon_view.append(self.env_orig.act_obj_in_view(self.pigeon.x, self.pigeon.y, self.pigeon.viewing_distance))

        # output = list(self.geomag_diff(pigeon_geomag, loft_geomag)) + list(pigeon_view.flatten()) + list(loft_view.flatten())
        output = list(pigeon_geomag) + list(pigeon_view.flatten()) + list(loft_geomag) + list(loft_view.flatten())

        return output

    # This finds matrices of the areas surrounding the pigeon and the loft. As it is necessary for the input to be in an understandable, and consistent size.
    def generate_views(self):
        # Either create the environment matrix, or load it from a pickle, to save processing time.
        try:
            if not self.first_run:
                with open("data/enviro_info/enviro_matrix.pkl", "rb") as f:
                    environ_matrix = pickle.load(f)
            else:
                environ_matrix = self.env_orig.create_matrix(X_SIZE, Y_SIZE)
                with open("data/enviro_info/enviro_matrix.pkl", "wb") as f:
                    pickle.dump(environ_matrix, f)
                self.first_run = False
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
            matrix_w, matrix_h = matrix.shape

            # To deal with errors when pigeon outside the boundary
            if matrix_w > center_x > 0 and matrix_h > center_y > 0:
                # Find either the real values in the main matrix, must be highest or 0, as otherwise it can return incorrect values.
                # e.g. -12 will return from wrong side of matrix
                x_min = max(center_x - radius, 0)
                x_max = min(center_x + radius, matrix_w)
                y_min = max(center_y - radius, 0)
                y_max = min(center_y + radius, matrix_h)

                # Placement in the view matrix.
                vx_min = radius - (center_x - x_min)
                vx_max = vx_min + (x_max - x_min)
                vy_min = radius - (center_y - y_min)
                vy_max = vy_min + (y_max - y_min)
            else:
                x_min = 0
                x_max = 0
                y_min = 0
                y_max = 0

                # Placement in the view matrix.
                vx_min = 0
                vx_max = 0
                vy_min = 0
                vy_max = 0

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

    # Reward function to indicate good/bad solutions
    def reward_function(self, prev_loc):
        reward = 0

        # Sinuosity - how close to the euclidian distance to the value the movement is.
        pige_start = np.asarray(self.pigeon_start_loc)
        loft_loc = np.asarray(self._target_location)
        current_loc = np.asarray([self.pigeon.x, self.pigeon.y])

        dist_from_euc_line = round(np.abs(np.cross(loft_loc - pige_start, pige_start - current_loc)) / norm(loft_loc - pige_start)/10, 3)

        if dist_from_euc_line <= 1:
            dist_from_euc_line = 1

        reward += round(1/dist_from_euc_line, 3) # THIS HAS AN ERROR WHEN THE PIGEON MOVES BEHIND THE START POINT, IT HAS UNEVEN VALUES.

        # print("VALUE: ", round(1/dist_from_euc_line, 3))
        # print(dist_from_euc_line)

        euc_line = math.sqrt((pige_start[0]-loft_loc[0])**2 + (pige_start[1]-loft_loc[1])**2)
        dist_from_loft = self.pigeon.dist_from_loft

        # This calculates the ratio, to the nearest point on the euclidian line
        # DONT KNOW IF IT IS ACTUALLY SINUOSITY, BUT WHATEVER.
        sinuosity = round((self.pigeon.dist_moved + dist_from_euc_line) / (euc_line - dist_from_loft), 3) * 0.01

        # print(sinuosity)

        # If moving towards home
        if self.pigeon.dist_from_loft < prev_loc:
            reward += 1

        # If in predator area
        for item in self.env_orig.active_objects:
            if item.getClass() == "Predator":
                if in_area(item.x, item.y, self.pigeon.x, self.pigeon.y, item.radius):
                    reward -= 1

        # If against a wall
        if self.pigeon.x < 1 or self.pigeon.y < 1 or self.pigeon.x > 999 or self.pigeon.y > 999:
            reward -= 3

        # If pigeon can see the loft
        if self.pigeon.dist_from_loft <= self.pigeon.viewing_distance:
            reward += 1

        # Pigeon distance from the loft
        # scale_dist_from_loft = np.log(self.pigeon.dist_from_loft/100)
        # reward -= scale_dist_from_loft

        # print(reward)

        return reward

    # This is added here so that at any point, you can click an area on the map and the pigeon will move there. Necessary for development.
    def click_handler(self, event):
        if event.num == 1:
            self.pigeon.x = event.x
            self.pigeon.y = event.y

    def save_env(self):
        passive_obj = self.env_orig.passive_objects
        active_obj = self.env_orig.active_objects
        geo_mag = self.env_orig.geo_map
        villages = self.env_orig.villages
        towns = self.env_orig.towns
        cities = self.env_orig.cities

        for item in active_obj:
            item.image = None

        save_all = (passive_obj, active_obj, geo_mag, villages, towns, cities)

        with open("model_parameters/environment.pkl", "wb") as f:
            pickle.dump(save_all, f)

    def load_env(self, passive_objects, active_objects, geo_mag, villages, towns, cities):
        self.env_orig.passive_objects = passive_objects
        self.env_orig.active_objects = active_objects
        self.env_orig.geo_map = geo_mag
        self.env_orig.villages = villages
        self.env_orig.towns = towns
        self.env_orig.cities = cities

        self.passive_objects = passive_objects
        self.active_objects = active_objects
        self.geomag_map = self.env_orig.geo_map.Map

        if self.draw:
            self.canvas.delete("all")

            for item in self.passive_objects:
                if item.getClass() == "City":
                    item.drawCity(self.canvas)
                elif item.getClass() == "Town":
                    item.drawTown(self.canvas)
                elif item.getClass() == "Village":
                    item.drawVillage(self.canvas)

            for item in self.active_objects:
                if item.getClass() == "Loft":
                    item.drawLoft(self.canvas, self.window)
                if item.getClass() == "Predator":
                    item.drawPredator(self.canvas)

            self.env_orig.geo_map.drawGrid(self.canvas, X_SIZE, Y_SIZE)

        self.first_run = True # To ensure the matrix redraws itself.
        return self.reset()