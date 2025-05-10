import datetime
import random
import time
import tkinter as tk
import threading

from reinforcement_learning import DQN
from genetic_algorithm import GeneticAlgorithm
from pigeon import Pigeon
from environment import Environment

X_SIZE = 1000
Y_SIZE = 1000
UPDATE_SPEED = 50

# TODO GO THROUGH AND REFACTOR SO PASSIVE_OBJECTS, ACTIVE_OBJECTS, GEOMAG_MAP ETC. ARE ALL ACCESS THROUGH AN ENVIRONMENT INSTANCE. NOT PASSED THROUGH FUNCS

# THis is just random movements, as a test and a baseline
def update(canvas, passive_objects, active_objects, geomag_map, real_time, pigeon):
    angle = random.randint(0, 360)
    pigeon.update(canvas, passive_objects, active_objects, geomag_map, angle, UPDATE_SPEED)

    print(pigeon.geomagDifference())

    # This updates the time in the top left, to show how much time is passing in the "real world" version of the model. This shows how fast these pigeons travel in reality
    real_time = real_time + datetime.timedelta(seconds=1)
    canvas.delete("time")
    canvas.create_text(20, 40, anchor="nw", text=f"Real World Time: {real_time.time()}", tag="time")

    canvas.after(UPDATE_SPEED, update, canvas, passive_objects, active_objects, geomag_map, real_time, pigeon)

# Just calls update above to start random movement.
def test():
    window = tk.Tk()
    window.resizable(False,False)

    env = Environment

    canvas, passive_objects, active_objects, geomag_map = env.initialise_environment(window, X_SIZE, Y_SIZE)

    pigeon = Pigeon("Jeff", X_SIZE, Y_SIZE, passive_objects, active_objects, geomag_map)
    pigeon.drawPigeon(canvas)

    # Used in update to show a "real world" time passing in the canvas
    real_time = datetime.datetime(2025,1,1, 0, 0, 0)

    update(canvas, passive_objects, active_objects, geomag_map, real_time, pigeon)

    window.mainloop()

def genetic_algorithm():
    genetic_algorithm = GeneticAlgorithm()
    genetic_algorithm.solve()


DQN()
# genetic_algorithm()
# test()