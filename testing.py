import pickle
import threading
import time
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from torch import optim

from environment import Environment
from gym_environment import GymEnvironment
from reinforcement_learning import DQN, DeepQLearningNetwork
from genetic_algorithm import GeneticAlgorithm
from scipy.stats import ttest_ind

class testDQNGeneralisability:
    def __init__(self, no_runs=10, draw=False):
        self.no_runs = no_runs
        self.draw = draw

        if self.draw:
            self.window = tk.Tk()
            self.window.resizable(False, False)

        # Take the environment variables
        with open("model_parameters/environment.pkl", "rb") as f:
            environment_objects = pickle.load(f)

        passive_objects, active_objects, geo_mag, villages, towns, cities = environment_objects

        self.metrics = {
            'name': "DQN_Generalisability",
            'no_moves': [],
            'dist_moved': [],
            'bool_reached_goal': [],
            'bool_landed': [],
            'bool_died': [],
            'time_taken': []
        }

        # Set the device to run on GPU, if applicable
        self.device = torch.device(
            "cpu"
        )

        # Get the state and action space size from a pickle
        with open("model_parameters/space_size.pkl", "rb") as f:
            state_size, action_size = pickle.load(f)

        # Initialise the networks
        # target_net = DeepQLearningNetwork(state_size, action_size)
        self.policy_net = DeepQLearningNetwork(state_size, action_size)
        # optimizer = optim.AdamW(policy_net.parameters(), amsgrad=True) # LEARNING RATE WAS IN HERE, DOES IT NEED TO BE RE-ADDED?

        # Load in the weights and parameters from the saved file
        checkpoint = torch.load("model_parameters/dqn_parameters.pt", weights_only=True)
        # target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set them to evaluation mode.
        # target_net.eval()
        self.policy_net.eval()

        if self.draw:
            t1 = threading.Thread(target=self.run)
            t1.start()

            self.window.mainloop()
        else:
            self.run()

    def run(self):
        for x in range(0, self.no_runs):
            print("Run: ", x)

            # Set new environment each time, to test generalisability
            if self.draw:
                self.env = GymEnvironment(draw=True, window=self.window)

                # Used to ensure that clicking anywhere on the map will move the pigeon to that location.
                self.window.bind("<Button-1>", self.env.click_handler)
            else:
                self.env = GymEnvironment(draw=False)

            # Initialise the metrics
            self.metrics['no_moves'].append(0)
            self.metrics['dist_moved'].append(0)

            start = time.time()
            terminated = False
            truncated = False
            self.env.reset()
            while not terminated and not truncated:
                self.metrics['no_moves'][-1] += 1
                # This is the real distance, for a set velocity at any point. See pigeon class
                self.metrics['dist_moved'][-1] += 1000 * (0.0138888889 / 30)
                observation = torch.tensor(self.env.get_observations(), device=self.device,
                                           dtype=torch.float32).flatten().unsqueeze(0)
                action = torch.tensor([[self.policy_net(observation).max(1).indices.item()]], device=self.device)
                _, _, terminated, truncated = self.env.step(action.item())

            end = time.time()
            self.metrics['time_taken'].append(end - start)

            # If still alive and reached loft
            if terminated:
                self.metrics['bool_reached_goal'].append(True)
                self.metrics['bool_died'].append(False)
                self.metrics['bool_landed'].append(False)
            else:
                self.metrics['bool_reached_goal'].append(False)
                if not self.env.pigeon.alive:
                    self.metrics['bool_died'].append(True)
                    self.metrics['bool_landed'].append(False)
                else:
                    self.metrics['bool_died'].append(False)
                    self.metrics['bool_landed'].append(True)

        with open("metric_runs/metrics.pkl", "ab+") as f:
            pickle.dump(self.metrics, f)

        if self.draw:
            self.window.destroy()


class testGAGeneralisability:
    def __init__(self, no_runs=10, draw=False):
        self.no_runs = no_runs
        self.draw = draw

        if self.draw:
            self.window = tk.Tk()
            self.window.resizable(False, False)

        # Take the environment variables
        with open("model_parameters/environment.pkl", "rb") as f:
            environment_objects = pickle.load(f)

        passive_objects, active_objects, geo_mag, villages, towns, cities = environment_objects

        self.metrics = {
            'name': "GA_Generalisability",
            'no_moves': [],
            'dist_moved': [],
            'bool_reached_goal': [],
            'bool_landed': [],
            'bool_died': [],
            'time_taken': []
        }

        # Set the device to run on GPU, if applicable
        self.device = torch.device(
            "cpu"
        )

        # Get the state and action space size from a pickle
        with open("model_parameters/space_size.pkl", "rb") as f:
            state_size, action_size = pickle.load(f)

        # Initialise the networks
        # target_net = DeepQLearningNetwork(state_size, action_size)
        self.policy_net = DeepQLearningNetwork(state_size, action_size)
        # optimizer = optim.AdamW(policy_net.parameters(), amsgrad=True) # LEARNING RATE WAS IN HERE, DOES IT NEED TO BE RE-ADDED?

        # Load in the weights and parameters from the saved file
        checkpoint = torch.load("model_parameters/ga_parameters.pt", weights_only=True)
        # target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.policy_net.load_state_dict(checkpoint['ga_model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set them to evaluation mode.
        # target_net.eval()
        self.policy_net.eval()

        if self.draw:
            t1 = threading.Thread(target=self.run)
            t1.start()

            self.window.mainloop()
        else:
            self.run()

    def run(self):
        for x in range(0, self.no_runs):
            print("Run: ", x)

            # Set new environment each time, to test generalisability
            if self.draw:
                self.env = GymEnvironment(draw=True, window=self.window)

                # Used to ensure that clicking anywhere on the map will move the pigeon to that location.
                self.window.bind("<Button-1>", self.env.click_handler)
            else:
                self.env = GymEnvironment(draw=False)

            # Initialise the metrics
            self.metrics['no_moves'].append(0)
            self.metrics['dist_moved'].append(0)

            start = time.time()
            terminated = False
            truncated = False
            self.env.reset()
            while not terminated and not truncated:
                self.metrics['no_moves'][-1] += 1
                # This is the real distance, for a set velocity at any point. See pigeon class
                self.metrics['dist_moved'][-1] += 1000 * (0.0138888889 / 30)
                observation = torch.tensor(self.env.get_observations(), device=self.device,
                                           dtype=torch.float32).flatten().unsqueeze(0)
                action = torch.tensor([[self.policy_net(observation).max(1).indices.item()]], device=self.device)
                _, _, terminated, truncated = self.env.step(action.item())

            end = time.time()
            self.metrics['time_taken'].append(end - start)

            # If still alive and reached loft
            if terminated:
                self.metrics['bool_reached_goal'].append(True)
                self.metrics['bool_died'].append(False)
                self.metrics['bool_landed'].append(False)
            else:
                self.metrics['bool_reached_goal'].append(False)
                if not self.env.pigeon.alive:
                    self.metrics['bool_died'].append(True)
                    self.metrics['bool_landed'].append(False)
                else:
                    self.metrics['bool_died'].append(False)
                    self.metrics['bool_landed'].append(True)

        with open("metric_runs/metrics.pkl", "ab+") as f:
            pickle.dump(self.metrics, f)

        if self.draw:
            self.window.destroy()


class test_DQN:
    def __init__(self, no_runs=10, draw=False):
        self.no_runs = no_runs
        self.draw = draw

        if self.draw:
            self.window = tk.Tk()
            self.window.resizable(False, False)

        # Take the environment variables
        with open("model_parameters/environment.pkl", "rb") as f:
            environment_objects = pickle.load(f)

        passive_objects, active_objects, geo_mag, villages, towns, cities = environment_objects

        if self.draw:
            self.env = GymEnvironment(draw=True, window=self.window)

            # Used to ensure that clicking anywhere on the map will move the pigeon to that location.
            self.window.bind("<Button-1>", self.env.click_handler)
        else:
            self.env = GymEnvironment(draw=False)
        self.env.load_env(passive_objects, active_objects, geo_mag, villages, towns, cities)

        self.metrics = {
            'name': "DQN",
            'no_moves': [],
            'dist_moved': [],
            'bool_reached_goal': [],
            'bool_landed': [],
            'bool_died': [],
            'time_taken': []
        }

        # Set the device to run on GPU, if applicable
        self.device = torch.device(
            "cpu"
        )

        # Get the state and action space size from a pickle
        with open("model_parameters/space_size.pkl", "rb") as f:
            state_size, action_size = pickle.load(f)

        # Initialise the networks
        # target_net = DeepQLearningNetwork(state_size, action_size)
        self.policy_net = DeepQLearningNetwork(state_size, action_size)
        # optimizer = optim.AdamW(policy_net.parameters(), amsgrad=True) # LEARNING RATE WAS IN HERE, DOES IT NEED TO BE RE-ADDED?

        # Load in the weights and parameters from the saved file
        checkpoint = torch.load("model_parameters/dqn_parameters.pt", weights_only=True)
        # target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set them to evaluation mode.
        # target_net.eval()
        self.policy_net.eval()

        if self.draw:
            t1 = threading.Thread(target=self.run)
            t1.start()

            self.window.mainloop()
        else:
            self.run()

    def run(self):
        for x in range(0, self.no_runs):
            print("Run: ", x)

            # Initialise the metrics
            self.metrics['no_moves'].append(0)
            self.metrics['dist_moved'].append(0)

            start = time.time()
            terminated = False
            truncated = False
            self.env.reset()
            while not terminated and not truncated:
                self.metrics['no_moves'][-1] += 1
                # This is the real distance, for a set velocity at any point. See pigeon class
                self.metrics['dist_moved'][-1] += 1000 * (0.0138888889/30)
                observation = torch.tensor(self.env.get_observations(), device=self.device, dtype=torch.float32).flatten().unsqueeze(0)
                action = torch.tensor([[self.policy_net(observation).max(1).indices.item()]], device=self.device)
                _, _, terminated, truncated = self.env.step(action.item())

            end = time.time()
            self.metrics['time_taken'].append(end-start)

            # If still alive and reached loft
            if terminated:
                self.metrics['bool_reached_goal'].append(True)
                self.metrics['bool_died'].append(False)
                self.metrics['bool_landed'].append(False)
            else:
                self.metrics['bool_reached_goal'].append(False)
                if not self.env.pigeon.alive:
                    self.metrics['bool_died'].append(True)
                    self.metrics['bool_landed'].append(False)
                else:
                    self.metrics['bool_died'].append(False)
                    self.metrics['bool_landed'].append(True)

        with open("metric_runs/metrics.pkl", "ab+") as f:
            pickle.dump(self.metrics, f)

        if self.draw:
            self.window.destroy()

class test_GA:
    def __init__(self, no_runs=10, draw=False):
        self.no_runs = no_runs
        self.draw = draw

        if self.draw:
            self.window = tk.Tk()
            self.window.resizable(False, False)

        # Take the environment variables
        with open("model_parameters/environment.pkl", "rb") as f:
            environment_objects = pickle.load(f)

        passive_objects, active_objects, geo_mag, villages, towns, cities = environment_objects

        if self.draw:
            self.env = GymEnvironment(draw=True, window=self.window)

            # Used to ensure that clicking anywhere on the map will move the pigeon to that location.
            self.window.bind("<Button-1>", self.env.click_handler)
        else:
            self.env = GymEnvironment(draw=False)
        self.env.load_env(passive_objects, active_objects, geo_mag, villages, towns, cities)

        self.metrics = {
            'name': "GA",
            'no_moves': [],
            'dist_moved': [],
            'bool_reached_goal': [],
            'bool_landed': [],
            'bool_died': [],
            'time_taken': []
        }

        # Set the device to run on GPU, if applicable
        self.device = torch.device(
            "cpu"
        )

        # Get the state and action space size from a pickle
        with open("model_parameters/space_size.pkl", "rb") as f:
            state_size, action_size = pickle.load(f)

        # Initialise the networks
        # target_net = DeepQLearningNetwork(state_size, action_size)
        self.policy_net = DeepQLearningNetwork(state_size, action_size)
        # optimizer = optim.AdamW(policy_net.parameters(), amsgrad=True) # LEARNING RATE WAS IN HERE, DOES IT NEED TO BE RE-ADDED?

        # Load in the weights and parameters from the saved file
        checkpoint = torch.load("model_parameters/ga_parameters.pt", weights_only=True)
        # target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.policy_net.load_state_dict(checkpoint['ga_model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set them to evaluation mode.
        # target_net.eval()
        self.policy_net.eval()

        if self.draw:
            t1 = threading.Thread(target=self.run)
            t1.start()

            self.window.mainloop()
        else:
            self.run()

    def run(self):
        for x in range(0, self.no_runs):
            print("Run: ", x)
            # Initialise the metrics
            self.metrics['no_moves'].append(0)
            self.metrics['dist_moved'].append(0)

            start = time.time()
            terminated = False
            truncated = False
            self.env.reset()
            while not terminated and not truncated:
                self.metrics['no_moves'][-1] += 1
                # This is the real distance, for a set velocity at any point. See pigeon class
                self.metrics['dist_moved'][-1] += 1000 * (0.0138888889 / 30)
                observation = torch.tensor(self.env.get_observations(), device=self.device,
                                           dtype=torch.float32).flatten().unsqueeze(0)
                action = torch.tensor([[self.policy_net(observation).max(1).indices.item()]], device=self.device)
                _, _, terminated, truncated = self.env.step(action.item())

            end = time.time()
            self.metrics['time_taken'].append(end - start)

            # If still alive and reached loft
            if terminated:
                self.metrics['bool_reached_goal'].append(True)
                self.metrics['bool_died'].append(False)
                self.metrics['bool_landed'].append(False)
            else:
                self.metrics['bool_reached_goal'].append(False)
                if not self.env.pigeon.alive:
                    self.metrics['bool_died'].append(True)
                    self.metrics['bool_landed'].append(False)
                else:
                    self.metrics['bool_died'].append(False)
                    self.metrics['bool_landed'].append(True)

        with open("metric_runs/metrics.pkl", "ab+") as f:
            pickle.dump(self.metrics, f)

        if self.draw:
            self.window.destroy()


# Plot all the graphs for a list full of the dictionaries of metrics values.
def plot_graphs(metrics):
    # Metrics are stored in this form:
        # metrics = {
        #     'name': ""
        #     'no_moves': [],
        #     'dist_moved': [],
        #     'bool_reached_goal': [],
        #     'bool_landed' = [],
        #     'bool_died': [],
        #     'time_taken': []
        # }


    # Box plots to compare average number of moves
    # Code influenced by: https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
    box_colours = ["mistyrose", "lightyellow"]
    fig, axes = plt.subplots(figsize=(10, 6))

    no_moves = []
    for met in metrics:
        no_moves.append(met['no_moves'])

    all_plts = axes.boxplot(no_moves, notch=False, sym='+', orientation='vertical')

    # Add horzontal lines to the gride
    axes.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    axes.set(
        axisbelow = True,
        title='Number of Moves by Model Type',
        xlabel='Model',
        ylabel='Number of Moves',
    )

    for x in range(0, len(metrics)):
        box = all_plts['boxes'][x]

        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])

        # Put colour in th plot
        axes.add_patch(Polygon(box_coords, facecolor=box_colours[x % 2]))

        # Re add the medians as the colour blocked it.
        med = all_plts['medians'][x]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            axes.plot(median_x, median_y)

    model_names = [metrics[x]['name'] for x in range(0, len(metrics))]
    axes.set_xticklabels(model_names)


    # Box plots for distance moved
    box_colours = ["mistyrose", "lightyellow"]
    fig, axes = plt.subplots(figsize=(10, 6))

    dist_moved = []
    for met in metrics:
        dist_moved.append(met['dist_moved'])

    all_plts = axes.boxplot(dist_moved, notch=False, sym='+', orientation='vertical')

    # Add horzontal lines to the gride
    axes.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                    alpha=0.5)

    axes.set(
        axisbelow=True,
        title='Distance Moved by Model Type',
        xlabel='Model',
        ylabel='Distance Moved',
    )

    for x in range(0, len(metrics)):
        box = all_plts['boxes'][x]

        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])

        # Put colour in th plot
        axes.add_patch(Polygon(box_coords, facecolor=box_colours[x % 2]))

        # Re add the medians as the colour blocked it.
        med = all_plts['medians'][x]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            axes.plot(median_x, median_y)

    model_names = [metrics[x]['name'] for x in range(0, len(metrics))]
    axes.set_xticklabels(model_names)


    # Horizontal bar charts to show number of dead pigeons and number that reached goal
    # Code taken from: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    x = np.arange(len(metrics))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))

    ax.set(axisbelow=True,
           title='No Pigeons Reached Goal, No. Pigeons Caught by Predator and No. Pigeons Lost by Model Type',)


    metrics_vals = {}

    # Model 1
    no_reached_goal = 0
    no_died = 0
    no_lost = 0
    for boolean in metrics[0]['bool_reached_goal']:
        if boolean:
            no_reached_goal += 1
    for boolean in metrics[0]['bool_died']:
        if boolean:
            no_died += 1
    for boolean in metrics[0]['bool_landed']:
        if boolean:
            no_lost += 1

    metrics_vals['No. Reached Goal'] = [no_reached_goal]
    metrics_vals['No Caught by Predator'] = [no_died]
    metrics_vals['No. Lost'] = [no_lost]

    # Model 2
    no_reached_goal = 0
    no_died = 0
    no_lost = 0
    for boolean in metrics[1]['bool_reached_goal']:
        if boolean:
            no_reached_goal += 1
    for boolean in metrics[1]['bool_died']:
        if boolean:
            no_died += 1
    for boolean in metrics[1]['bool_landed']:
        if boolean:
            no_lost += 1

        print(1)
        print(no_reached_goal)
        print(no_died)
        print(no_lost)

        print(metrics[1])

    metrics_vals['No. Reached Goal'].append(no_reached_goal)
    metrics_vals['No Caught by Predator'].append(no_died)
    metrics_vals['No. Lost'].append(no_lost)

    for attribute, measurement in metrics_vals.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('No. Pigeons ')
    ax.set_xlabel('Model Type')
    ax.legend(loc='upper right', ncols=3)
    model_names = [metrics[x]['name'] for x in range(0, len(metrics))]
    ax.set_xticks(x + 0.5*width, model_names)

    # Add vertical lines to the grid
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                    alpha=0.5)


    plt.show()


def t_tests(metrics):
    # metrics = {
    #     'name': ""
    #     'no_moves': [],
    #     'dist_moved': [],
    #     'bool_reached_goal': [],
    #     'bool_landed' = [],
    #     'bool_died': [],
    #     'time_taken': []
    # }

    t_test = {}

    for key in metrics[0].keys():
        if key not in ['name', 'bool_died', 'bool_landed', 'bool_reached_goal']:
            result = ttest_ind(metrics[0][key], metrics[1][key]).pvalue
            t_test[key] = result

    for key in t_test.keys():
        print(key, " p value = ", t_test[key])


def run_tests(new_run=False, generalisability=False):
    if new_run:
        open("metric_runs/metrics.pkl", "wb").close()

        if not generalisability:
            test_DQN(10, False)
            test_GA(10, False)
        else:
            testDQNGeneralisability(10, False)
            testGAGeneralisability(10, False)

    # unpickle the metrics
    metrics = []
    with open("metric_runs/metrics.pkl", "rb") as f:
        while 1:
            try:
                loaded = pickle.load(f)
                metrics.append(loaded)
            except EOFError:
                break

    plot_graphs(metrics)
    t_tests(metrics)





