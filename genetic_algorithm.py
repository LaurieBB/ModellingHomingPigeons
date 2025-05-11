import datetime
import math
import time

import pygad
import tkinter as tk

from environment import Environment, in_area
from pigeon import Pigeon
import threading
import numpy as np
from numpy.linalg import norm

X_SIZE = 1000
Y_SIZE = 1000
UPDATE_SPEED = 50

# TODO REFERENCE: https://pygad.readthedocs.io/en/latest/ - Code not taken from here, just reference the documentation
class GeneticAlgorithm:
    def __init__(self):
        # Used in update to show a "real world" time passing in the canvas
        self.real_time = datetime.datetime(2025, 1, 1, 0, 0, 0)
        self.window = tk.Tk()
        self.window.resizable(False, False)

        # Initialise environment
        self.env = Environment()
        self.canvas, self.passive_objects, self.active_objects, self.geomag_map = self.env.initialise_environment(self.window,
                                                                                                         X_SIZE, Y_SIZE)

        self.window.bind("<Button-1>", self.click_handler)

        self.pigeon = Pigeon("Pigeon1", X_SIZE, Y_SIZE, self.passive_objects, self.active_objects, self.geomag_map)

        self.pigeon_start_loc = [self.pigeon.x, self.pigeon.y]

        loft = [f for f in self.active_objects if f.getClass() == "Loft"][0]
        self.loft_location = [loft.x, loft.y]

    def solve(self):
        def fitness_func(ga_instance, solution, solution_idx):
            reward = 0
            # Sinuosity - how close to the euclidian distance to the value the movement is.
            pige_start = np.asarray(self.pigeon_start_loc)
            loft_loc = np.asarray(self.loft_location)
            current_loc = np.asarray([self.pigeon.x, self.pigeon.y])

            dist_from_euc_line = np.abs(np.cross(loft_loc - pige_start, pige_start - current_loc)) / norm(
                loft_loc - pige_start)

            reward -= dist_from_euc_line / 100

            # If moving towards home
            if self.pigeon.dist_from_loft < prev_loc:
                reward += 1

            # If in predator area
            for item in self.env.active_objects:
                if item.getClass() == "Predator":
                    if in_area(item.x, item.y, self.pigeon.x, self.pigeon.y, item.radius):
                        reward -= 1

            # If against a wall
            if self.pigeon.x < 1 or self.pigeon.y < 1 or self.pigeon.x > 999 or self.pigeon.y > 999:
                reward -= 1

            # If pigeon can see the loft
            if self.pigeon.dist_from_loft <= self.pigeon.viewing_distance:
                reward += 1

            print(reward)

            return reward
            #TODO THIS CURRENTLY ISN'T LEARNING BECAUSE IT IS FINDING A SINGULAR OPTIMAL SOLUTIONS (ONLY ONE ANGLE)
            # AND NOT REACTING TO DIFFERENT OBSTACLES, OR EVEN MOVING OUT OF ONE SQUARE. IT IS STUCK IN A LOCAL
            # OPTIMA. COME BACK AND TRY AND WORK ON IT.

            #todo Additionally, maybe the fact it is finding only one solution is not adequate for this, as it will just always
            # move in one direction, regardless of obstacles. Maybe this is just a fitness function fix though????

        # This is used to move the pigeon after each generation, according to the best angle of movement output by the code.
        def callback(ga_instance):
            # This is used to get the current fitness of the best solution.
            solution, solution_fitness, solution_idx = ga_instance.best_solution()

            print("     Best fitness: ", solution_fitness)
            print("     Best solution: ", solution)
            print("     Geomag val: ", (1/sum(self.pigeon.geomagDifference()))*10000)

            # Angle is set to the best solution
            angle = solution

            self.pigeon.update(self.canvas, self.passive_objects, self.active_objects, self.geomag_map, angle,
                               UPDATE_SPEED)

            # This updates the time in the top left, to show how much time is passing in the "real world" version of the model. This shows how fast these pigeons travel in reality
            self.real_time += datetime.timedelta(seconds=1)
            self.canvas.delete("time")
            self.canvas.create_text(20, 40, anchor="nw", text=f"Real World Time: {self.real_time.time()}", tag="time")

            # Returns "stop" when either the pigeon dies, or if it finds the solution.
            if self.pigeon.dist_from_loft <= 0 or not self.pigeon.alive:
                return "stop"

            # time.sleep(0.05)

        # Run the GA
        ga_instance = pygad.GA(num_generations=10000,
                               num_parents_mating=10,
                               fitness_func=fitness_func,
                               sol_per_pop=30,
                               num_genes=1,
                               gene_type=int,
                               gene_space=range(0, 360),
                               init_range_low=0,
                               init_range_high=360,
                               random_mutation_min_val=0,
                               random_mutation_max_val=360,
                               mutation_by_replacement=True,
                               mutation_probability=0.1,
                               mutation_percent_genes=100,
                               on_generation=callback)

        # ga_instance.run()

        self.pigeon.drawPigeon(self.canvas)

        t1 = threading.Thread(target=ga_instance.run)
        t1.start()

        self.window.mainloop()

    # This is added here so that at any point, you can click an area on the map and the pigeon will move there. Necessary for development.
    def click_handler(self, event):
            if event.num == 1:
                self.pigeon.x = event.x
                self.pigeon.y = event.y





