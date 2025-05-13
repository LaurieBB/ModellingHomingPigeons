import datetime
import math
import pickle
import time

import pygad
import tkinter as tk

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

# TODO REFERENCE: https://pygad.readthedocs.io/en/latest/ - Code not taken from here, just reference the documentation
class GeneticAlgorithm:
    def __init__(self, draw=False):
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

        self.observation =  self.env.load_env(passive_objects, active_objects, geo_mag, villages, towns, cities)

        # This is used in the fitness function to edit a pigeon instance, and access the functions without having to change the real one.
        self.temp_pigeon = Pigeon("temp", 1000, 1000, self.env.passive_objects, self.env.active_objects,
                                 self.env.geomag_map)


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

            # This is a step function taken from gym, but applied to a brand new pigeon to test the outcome of a singular step and return a reward, without editing the environment.
            def synthetic_step(action):
                # Calculate pigeon speed
                hypotenuse = 1000 * (0.0138888889 / 30)  # This ensures constant speed. 0.0138... is the real value of 50mph converted to mps and scaled with 30.
                pigeon_xv = math.sin(math.radians(action)) * hypotenuse
                pigeon_yv = math.cos(math.radians(action)) * hypotenuse

                # previous location of the pigeon
                prev_loc = self.env.pigeon.dist_from_loft
                self.temp_pigeon.dist_from_loft = self.env.pigeon.dist_from_loft

                if self.env.pigeon.alive:
                    # Move temp pigeon
                    self.temp_pigeon.x = self.env.pigeon.x + pigeon_xv
                    self.temp_pigeon.y = self.env.pigeon.y + pigeon_yv

                # Checks that the pigeon is alive, and not in a predator area, this function also works out probability of death and
                self.temp_pigeon.pigeonInDanger(self.env.active_objects)

                # If alive, updates vision and geomagnetic location.
                if self.env.pigeon.alive:
                    # Updates the performance metrics
                    loft = [x for x in self.env.active_objects if x.getClass() == "Loft"][0]  # This retrieves the loft instance (There should only be one)
                    self.temp_pigeon.dist_from_loft = math.sqrt((self.temp_pigeon.x - loft.x) ** 2 + (self.temp_pigeon.y - loft.y) ** 2)

                    # End if pigeon is in same location as loft, it may seem like a low value, however the pigeon is also
                    if self.temp_pigeon.dist_from_loft <= 20:
                        print("MY PIGEON IS HOME!!!")
                        terminated = True
                        truncated = False
                    elif self.env.pigeon.no_moves >= 5000:
                        terminated = False
                        truncated = True
                    # No issues.
                    else:
                        terminated = False
                        truncated = False
                else:
                    terminated = False
                    truncated = True

                reward_out = reward_function(prev_loc)

                # Necessary if the pigeon goes too far out of bounds
                if reward_out <= -10:
                    truncated = True
                    terminated = False

                # Set reward for if it is finished, terminated=reached goal, truncated=died/too many moves
                if terminated:
                    reward_out = 10
                if truncated:
                    reward_out = -10

                return reward_out

            # todo change this if change reward function in gym_environment
            def reward_function(prev_loc):
                reward = 0

                # Sinuosity - how close to the euclidian distance to the value the movement is.
                pige_start = np.asarray(self.env.pigeon_start_loc)
                loft_loc = np.asarray(self.env._target_location)
                current_loc = np.asarray([self.temp_pigeon.x, self.temp_pigeon.y])

                dist_from_euc_line = round(np.abs(np.cross(loft_loc - pige_start, pige_start - current_loc)) / norm(
                    loft_loc - pige_start) / 10, 3)

                if dist_from_euc_line <= 1:
                    dist_from_euc_line = 1

                reward += round(1 / dist_from_euc_line, 3)  # THIS HAS AN ERROR WHEN THE PIGEON MOVES BEHIND THE START POINT, IT HAS UNEVEN VALUES.

                # print("VALUE: ", round(1/dist_from_euc_line, 3))
                # print(dist_from_euc_line)

                dist_from_loft = math.sqrt((current_loc[0] - loft_loc[0]) ** 2 + (current_loc[1] - loft_loc[1]) ** 2)

                # If moving towards home
                if dist_from_loft < prev_loc:
                    reward += 1

                # If in predator area
                for item in self.env.env_orig.active_objects:
                    if item.getClass() == "Predator":
                        if in_area(item.x, item.y, self.temp_pigeon.x, self.temp_pigeon.y, item.radius):
                            reward -= 1

                # If against a wall
                if self.temp_pigeon.x < 1 or self.temp_pigeon.y < 1 or self.temp_pigeon.x > 999 or self.temp_pigeon.y > 999:
                    reward -= 3

                # If pigeon can see the loft
                if self.temp_pigeon.dist_from_loft <= self.temp_pigeon.viewing_distance:
                    reward += 1

                return reward

            fitness = synthetic_step(action_inp)

            return fitness

        # This is used to move the pigeon after each generation, according to the best angle of movement output by the code.
        def callback(ga_instance):
            nonlocal model, no_iterations
            # This is used to get the current fitness of the best solution.
            solution, solution_fitness, solution_idx = ga_instance.best_solution()

            model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                               weights_vector=solution)

            # Use the current solution as the model parameters.
            model.load_state_dict(model_weights_dict)

            tens_observation = torch.tensor(self.observation, device=device, dtype=torch.float32).flatten().unsqueeze(0)
            action = torch.tensor([[model(tens_observation).max(1).indices.item()]], device=device)

            self.observation, reward, terminated, truncated = self.env.step(action.item())

            no_iterations += 1

            if no_iterations % 5 == 0:
                save_checkpoint()

        model = torch.load("model_parameters/dqn_model.pt", weights_only=False)

        torch_ga = torchga.TorchGA(model=model,
                                   num_solutions=10)

        # Run the GA
        ga_instance = pygad.GA(num_generations=200,
                               fitness_func=fitness_func,
                               num_parents_mating=3,
                               sol_per_pop=10,
                               initial_population=torch_ga.population_weights,
                               parent_selection_type='tournament',
                               crossover_type='single_point',
                               mutation_type='random',
                               keep_elitism=2, # MAYBE CHANGE BACK TO 1 IF NO DIFFERENCE
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





