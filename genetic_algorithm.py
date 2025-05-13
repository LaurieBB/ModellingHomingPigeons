import copy
import datetime
import math
import pickle
import time

import pygad
import tkinter as tk

import skimage
import torch

from environment import Environment, in_area
from gym_environment import GymEnvironment
from reinforcement_learning import DeepQLearningNetwork
from pigeon import Pigeon
import threading
import numpy as np
from numpy.linalg import norm

# Imports for PyTorch with PyGAD
from pygad import torchga

X_SIZE = 1000
Y_SIZE = 1000
UPDATE_SPEED = 50

# Stores the necessary device for running tensors
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# REFERENCE: https://pygad.readthedocs.io/en/latest/ - Code not taken from here, just reference the documentation
class GeneticAlgorithm:
    def __init__(self, draw=False):
        self.batch_size = 150

        self.draw = draw
        if draw:
            self.window = tk.Tk()
            self.window.resizable(False, False)

        # Take the environment variables
        with open("model_parameters/environment.pkl", "rb") as f:
            environment_objects = pickle.load(f)

        passive_objects, active_objects, geo_mag, villages, towns, cities = environment_objects

        # Set environment values.
        if draw:
            self.env = GymEnvironment(draw=True, window=self.window)
            self.window.bind("<Button-1>", self.click_handler)
        else:
            self.env = GymEnvironment(False)

        self.observation = self.env.load_env(passive_objects, active_objects, geo_mag, villages, towns, cities)

        # This is used in the fitness function to edit a pigeon instance, and access the functions without having to change the real one.
        # self.temp_pigeon = Pigeon("temp", 1000, 1000, self.env.passive_objects, self.env.active_objects,
        #                          self.env.geomag_map)
        self.temp_env = GymEnvironment(draw=False)
        self.temp_env.load_env(passive_objects, active_objects, geo_mag, villages, towns, cities)

    def solve(self):
        # function to save current state of the model
        def save_checkpoint():
            nonlocal model

            # Saving the weights and parameters for later use in testing
            torch.save({
                "ga_model_state_dict": model.state_dict(),
            }, "model_parameters/ga_parameters.pt")

            # Saving the entire model for use in GA
            torch.save(model, "model_parameters/ga_model.pt")

            # Save the environment for testing
            self.env.save_env()

        def fitness_func(ga_instance, solution, solution_idx):
            nonlocal model
            model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                               weights_vector=solution)

            # Use the current solution as the model parameters.
            model.load_state_dict(model_weights_dict)

            # Get the current observation (self.observation) - doesn't change when fitness function is used, only in callback
            tens_observation = torch.tensor(self.observation, device=device, dtype=torch.float32).flatten().unsqueeze(0)
            action_inp = torch.tensor([[model(tens_observation).max(1).indices.item()]], device=device).item()

            fitness = 0
            for x in range(0, self.batch_size):
                reward, observations, terminated, truncated = synthetic_step(action_inp, temp_env=self.temp_env)
                if terminated or truncated:
                    fitness += reward
                    break
                else:
                    fitness += reward
                    tens_observation = torch.tensor(observations, device=device, dtype=torch.float32).flatten().unsqueeze(0)
                    action_inp = torch.tensor([[model(tens_observation).max(1).indices.item()]], device=device).item()

            self.temp_env = copy.deepcopy(self.env)

            return fitness/self.batch_size

        # This is used to move the pigeon after each generation, according to the best angle of movement output by the code.
        def callback(ga_instance):
            nonlocal model, no_iterations
            # This is used to get the current fitness of the best solution.
            solution, solution_fitness, solution_idx = ga_instance.best_solution()

            print("Iterations: ", no_iterations)

            model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                               weights_vector=solution)

            # Use the current solution as the model parameters.
            model.load_state_dict(model_weights_dict)

            tens_observation = torch.tensor(self.observation, device=device, dtype=torch.float32).flatten().unsqueeze(0)
            action = torch.tensor([[model(tens_observation).max(1).indices.item()]], device=device)

            self.observation, reward, terminated, truncated = self.env.step(action.item())

            if terminated or truncated:
                self.observation = self.env.reset()

            no_iterations += 1

            if no_iterations % 2  == 0:
                save_checkpoint()

        model = torch.load("model_parameters/dqn_model.pt", weights_only=False, map_location=device)

        torch_ga = torchga.TorchGA(model=model,
                                   num_solutions=10)

        # Run the GA
        ga_instance = pygad.GA(num_generations=100,
                               fitness_func=fitness_func,
                               num_parents_mating=3,
                               sol_per_pop=10,
                               initial_population=torch_ga.population_weights,
                               parent_selection_type='tournament',
                               crossover_type='single_point',
                               mutation_type='random',
                               keep_elitism=2,
                               K_tournament=3,
                               on_generation=callback)

        no_iterations = 0

        if self.draw:
            t1 = threading.Thread(target=ga_instance.run)
            t1.start()

            self.window.mainloop()

        else:
            ga_instance.run()

    # This is added here so that at any point, you can click an area on the map and the pigeon will move there. Necessary for development.
    def click_handler(self, event):
            if event.num == 1:
                self.env.pigeon.x = event.x
                self.env.pigeon.y = event.y


def synthetic_step(action, temp_env):

    def reward_function(prev_loc):
        nonlocal temp_env

        reward = 0

        # Sinuosity - how close to the euclidian distance to the value the movement is.
        pige_start = np.asarray(temp_env.pigeon_start_loc)
        loft_loc = np.asarray(temp_env._target_location)
        current_loc = np.asarray([temp_env.pigeon.x, temp_env.pigeon.y])

        dist_from_euc_line = round(np.abs(np.cross(loft_loc - pige_start, pige_start - current_loc)) / norm(
            loft_loc - pige_start) / 10, 3)

        if dist_from_euc_line <= 1:
            dist_from_euc_line = 1

        reward += round(1 / dist_from_euc_line,
                        3)  # THIS HAS AN ERROR WHEN THE PIGEON MOVES BEHIND THE START POINT, IT HAS UNEVEN VALUES.

        # print("VALUE: ", round(1/dist_from_euc_line, 3))
        # print(dist_from_euc_line)

        dist_from_loft = math.sqrt((current_loc[0] - loft_loc[0]) ** 2 + (current_loc[1] - loft_loc[1]) ** 2)

        # If moving towards home
        if dist_from_loft < prev_loc:
            reward += 1

        # If in predator area
        for item in temp_env.env_orig.active_objects:
            if item.getClass() == "Predator":
                if in_area(item.x, item.y, temp_env.pigeon.x, temp_env.pigeon.y, item.radius):
                    reward -= 1

        # If against a wall
        if temp_env.pigeon.x < 1 or temp_env.pigeon.y < 1 or temp_env.pigeon.x > 999 or temp_env.pigeon.y > 999:
            reward -= 3

        # If pigeon can see the loft
        if temp_env.pigeon.dist_from_loft <= temp_env.pigeon.viewing_distance:
            reward += 1

        return reward


    # Used to get the current location of the pigeon and the loft (observations) - always returns a list
    def get_observations():
        nonlocal temp_env

        # This returns matrices of the pigeon view, and the loft memory
        pigeon_view, loft_view = generate_views()

        # Max Pooling to reduce the size of the input space for DQN
        pigeon_view = skimage.measure.block_reduce(pigeon_view, (3, 3), np.max)
        loft_view = skimage.measure.block_reduce(loft_view, (3, 3), np.max)

        # Geomagnetic values for the pigeon and the loft.
        pigeon_geomag = temp_env.pigeon.current_geomag_loc
        loft_geomag = temp_env.pigeon.geomag_loft

        # Below is the list observation space
        # pigeon_view = self.env_orig.obj_in_view(self.pigeon.x, self.pigeon.y, self.pigeon.viewing_distance)
        # loft_view = self.env_orig.obj_in_view(self._target_location[0], self._target_location[1], self.pigeon.memory_radius)
        # pigeon_view.append(self.env_orig.act_obj_in_view(self.pigeon.x, self.pigeon.y, self.pigeon.viewing_distance))

        # output = list(self.geomag_diff(pigeon_geomag, loft_geomag)) + list(pigeon_view.flatten()) + list(loft_view.flatten())
        output = list(pigeon_geomag) + list(pigeon_view.flatten()) + list(loft_geomag) + list(loft_view.flatten())

        return output

    # This finds matrices of the areas surrounding the pigeon and the loft. As it is necessary for the input to be in an understandable, and consistent size.
    def generate_views():
        nonlocal temp_env

        # Either create the environment matrix, or load it from a pickle, to save processing time.
        with open("data/enviro_info/enviro_matrix.pkl", "rb") as f:
            environ_matrix = pickle.load(f)

        # This finds the area in the full matrix that just the pigeon can see.
        pigeon_matrix = np.zeros((X_SIZE, Y_SIZE))
        mask = temp_env.env_orig.np_circle_func(temp_env.pigeon.x, temp_env.pigeon.y, temp_env.pigeon.viewing_distance, environ_matrix)
        pigeon_matrix[mask] = environ_matrix[mask.T]  # Generate the masked version of the full environment, showing only the circular pigeons view.

        loft_matrix = np.zeros((X_SIZE, Y_SIZE))
        mask = temp_env.env_orig.np_circle_func(temp_env._target_location[0], temp_env._target_location[1],
                                            temp_env.pigeon.memory_radius, environ_matrix)
        loft_matrix[mask] = environ_matrix[mask.T]  # Generate the masked version of the full environment, showing only the circular loft memory

        # Get radius distances for the loft and the pigeon.
        viewing_distance = int(temp_env.pigeon.viewing_distance)
        memory_radius = int(temp_env.pigeon.memory_radius)

        view_size_pigeon = viewing_distance * 2
        view_size_loft = memory_radius * 2

        # Initialise matrices for the pigeon view size and loft memory.
        pigeon_view = np.zeros([view_size_pigeon, view_size_pigeon])
        loft_view = np.zeros([view_size_loft, view_size_loft])

        # Round and convert positions to int as is necessary, due to them being float values.
        pige_x = int(round(temp_env.pigeon.x))
        pige_y = int(round(temp_env.pigeon.y))
        loft_x = int(round(temp_env._target_location[0]))
        loft_y = int(round(temp_env._target_location[1]))

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

    # Calculate pigeon speed
    hypotenuse = 1000 * (
                0.0138888889 / 30)  # This ensures constant speed. 0.0138... is the real value of 50mph converted to mps and scaled with 30.
    pigeon_xv = math.sin(math.radians(action)) * hypotenuse
    pigeon_yv = math.cos(math.radians(action)) * hypotenuse

    # previous location of the pigeon
    prev_loc = temp_env.pigeon.dist_from_loft

    if temp_env.pigeon.alive:
        # Move temp pigeon
        temp_env.pigeon.x = temp_env.pigeon.x + pigeon_xv
        temp_env.pigeon.y = temp_env.pigeon.y + pigeon_yv

    # Checks that the pigeon is alive, and not in a predator area, this function also works out probability of death and
    temp_env.pigeon.pigeonInDanger(temp_env.active_objects)

    # If alive, updates vision and geomagnetic location.
    if temp_env.pigeon.alive:
        # Updates the performance metrics
        loft = [x for x in temp_env.active_objects if x.getClass() == "Loft"][
            0]  # This retrieves the loft instance (There should only be one)
        temp_env.pigeon.dist_from_loft = math.sqrt(
            (temp_env.pigeon.x - loft.x) ** 2 + (temp_env.pigeon.y - loft.y) ** 2)

        # End if pigeon is in same location as loft, it may seem like a low value, however the pigeon is also
        if temp_env.pigeon.dist_from_loft <= 20:
            print("MY PIGEON IS HOME!!!")
            terminated = True
            truncated = False
        elif temp_env.pigeon.no_moves >= 5000:
            terminated = False
            truncated = True
        # No issues.
        else:
            terminated = False
            truncated = False
    else:
        terminated = False
        truncated = True

    observations = get_observations()
    reward_out = reward_function(prev_loc)

    # Set reward for if it is finished, terminated=reached goal, truncated=died/too many moves
    if terminated:
        reward_out = 10
    if truncated:
        reward_out = -10

    return reward_out, observations, terminated, truncated




