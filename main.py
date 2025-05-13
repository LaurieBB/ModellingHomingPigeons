import datetime
import random
import time
import tkinter as tk
import threading

from reinforcement_learning import DQN
from genetic_algorithm import GeneticAlgorithm
from pigeon import Pigeon
from environment import Environment
from testing import run_tests

X_SIZE = 1000
Y_SIZE = 1000
UPDATE_SPEED = 50

# "draw" should be a boolean value indicating if you want the tkinter environment printed
def run_ga(draw):
    start = time.time()
    genetic_algorithm = GeneticAlgorithm(draw)
    genetic_algorithm.solve()
    end = time.time()
    print("time to run: ", end - start)

# "draw" should be a boolean value indicating if you want the tkinter environment printed
def run_dqn(draw):
    start = time.time()
    DQN(draw)
    end = time.time()
    print("time to run: ", end - start)

# "draw" should be a boolean value indicating if you want the tkinter environment printed
# "generalisability" is a boolean value indicating which test you want to run, False=Normal test, True=Generalisability test (see report)
def tests(draw, generalisability):
    start = time.time()
    run_tests(draw, generalisability)
    end = time.time()
    print("time to run: ", end - start)


tests(False, False)
# run_dqn(False)
# run_ga(False)




