import datetime
import math
import time

import pygad
import tkinter as tk

from environment import Environment
from pigeon import Pigeon
import threading

X_SIZE = 1000
Y_SIZE = 1000
UPDATE_SPEED = 50

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

        self.pigeon = Pigeon("Pigeon1", X_SIZE, Y_SIZE, self.passive_objects, self.active_objects, self.geomag_map)

    def solve(self):
        def fitness_func(ga_instance, solution, solution_idx):
            fitness = 0

            #TODO THIS CURRENTLY ISN'T LEARNING BECAUSE IT IS FINDING A SINGULAR OPTIMAL SOLUTIONS (ONLY ONE ANGLE)
            # AND NOT REACTING TO DIFFERENT OBSTACLES, OR EVEN MOVING OUT OF ONE SQUARE. IT IS STUCK IN A LOCAL
            # OPTIMA. COME BACK AND TRY AND WORK ON IT.

            #todo Additionally, maybe the fact it is finding only one solution is not adequate for this, as it will just always
            # move in one direction, regardless of obstacles. Maybe this is just a fitness function fix though????

            # TODO ADD IN THE DIRECT DISTANCE BETWEEN THE PIGEON AND THE LOFT, THIS IS ALLOWED. THEY DO NOT HAVE TO HAVE ONLY THE PIGEONS KNOWLEDGE. 

            # Reward similarity to home map, based on distance and other aspects
            # home_view = self.pigeon.memory_home
            # pigeon_view = self.pigeon.pigeon_vision
            # for item in home_view:
            #     # If the specific town/village/city matches the one in the home view.
            #     if item[3] in pigeon_view:
            #         fitness += 100  # TODO MAY NEED TO CHANGE THESE VALUES
            #
            #         # Get the same value in the pigeon view
            #         pigeon_item = [f for f in pigeon_view if f[3] == item[3]][0]
            #         # Evaluate the difference between the distance from home to the item, compared to the pigeon to the item.
            #         fitness += math.floor((1 / abs(item[2] - pigeon_item[2])) * 100)
            #         print(f"Distance from home item value: {math.floor((1 / abs(item[2] - pigeon_item[2])) * 100)}")

            # Reward similarity to geomagnetic map
            geo_diff = sum(self.pigeon.geomagDifference())
            fitness += (1 / geo_diff) * 10000  # If the distance is higher, it is worse. The fitness will be highest when all values are 0.
            print("Geomag val: ", (1/sum(self.pigeon.geomagDifference()))*10000)


            # Penalise being inside the danger zone

            # Penalise same movement for multiple turns, with no change (e.g. if hit a wall)

            # Penalise death majorly
            if not self.pigeon.alive:
                fitness = 0

            return fitness

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
            if self.pigeon.dist_from_loft == 0 or not self.pigeon.alive:
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




