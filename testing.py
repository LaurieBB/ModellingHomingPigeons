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


class Test:
    def test_DQN_generalisability(self, no_runs=10):
        metrics = {
            'name': "DQN-Generalisability",
            'no_moves': [],
            'dist_moved': [],
            'bool_reached_goal': [],
            'bool_died': [],
            'time_taken': []
        }

        # Get the state and action space size from a pickle
        with open("model_parameters/space_size.pkl", "rb") as f:
            state_size, action_size = pickle.load(f)

        # Initialise the networks
        target_net = DeepQLearningNetwork(state_size, action_size)
        policy_net = DeepQLearningNetwork(state_size, action_size)
        optimizer = optim.AdamW(policy_net.parameters(),
                                amsgrad=True)  # LEARNING RATE WAS IN HERE, DOES IT NEED TO BE RE-ADDED?

        # Load in the weights and parameters from the saved file
        checkpoint = torch.load("model_parameters/dqn_parameters.pt", weights_only=True)
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set them to evaluation mode.
        target_net.eval()
        policy_net.eval()

        # Take the environment variables
        with open("model_parameters/environment.pkl", "rb") as f:
            environment_objects = pickle.load(f)

        passive_objects, active_objects, geo_mag, villages, towns, cities = environment_objects

        # Initialise window
        window = tk.Tk()
        window.resizable(False, False)

        # Used to ensure that clicking anywhere on the map will move the pigeon to that location.
        # window.bind("<Button-1>", self.env.env_orig.click_handler)

        # Run the tests and store the metrics
        def run():
            for x in range(0, no_runs):
                # Set new environment each time, to test generalisability
                self.env = GymEnvironment(draw=True, window=window)

                # Initialise the metrics
                metrics['no_moves'].append(0)
                metrics['dist_moved'].append(0)

                start = time.time()
                terminated = False
                truncated = False
                self.env.reset()
                while not terminated and not truncated:
                    metrics['no_moves'].append(1)
                    # This is the real distance, for a set velocity at any point. See pigeon class
                    metrics['dist_moved'].append(1000 * (0.0138888889 / 30))
                    action = policy_net(self.env.get_observations())
                    _, _, terminated, truncated = self.env.step(action)

                end = time.time()
                metrics['time_taken'].append(end - start)

                # If still alive and reached loft
                if terminated:
                    metrics['bool_reached_goal'].append(True)
                    metrics['bool_died'].append(False)
                else:
                    metrics['bool_reached_goal'].append(False)
                    if not self.env.pigeon.alive:
                        metrics['bool_died'].append(True)
                    else:
                        metrics['bool_died'].append(False)

        t1 = threading.Thread(target=run)
        t1.start()

        window.mainloop()

    # TODO ADD SAME FUNCTIONS FOR GA.

def test_DQN(no_runs=10):
    metrics = {
        'name': "DQN",
        'no_moves': [],
        'dist_moved': [],
        'bool_reached_goal': [],
        'bool_died': [],
        'time_taken': []
    }

    # Get the state and action space size from a pickle
    with open("model_parameters/space_size.pkl", "rb") as f:
        state_size, action_size = pickle.load(f)

    # Initialise the networks
    target_net = DeepQLearningNetwork(state_size, action_size)
    policy_net = DeepQLearningNetwork(state_size, action_size)
    optimizer = optim.AdamW(policy_net.parameters(), amsgrad=True) # LEARNING RATE WAS IN HERE, DOES IT NEED TO BE RE-ADDED?

    # Load in the weights and parameters from the saved file
    checkpoint = torch.load("model_parameters/dqn_parameters.pt", weights_only=True)
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Set them to evaluation mode.
    target_net.eval()
    policy_net.eval()

    # Take the environment variables
    with open("model_parameters/environment.pkl", "rb") as f:
        environment_objects = pickle.load(f)

    passive_objects, active_objects, geo_mag, villages, towns, cities = environment_objects

    # Initialise window
    window = tk.Tk()
    window.resizable(False, False)

    # Set environment values.
    env = GymEnvironment(draw=True, window=window)
    env.load_env(passive_objects, active_objects, geo_mag, villages, towns, cities)

    # Used to ensure that clicking anywhere on the map will move the pigeon to that location.
    # window.bind("<Button-1>", self.env.env_orig.click_handler)

    # Run the tests and store the metrics
    def run():
        nonlocal env, window
        # Stores the necessary device for running tensors
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        for x in range(0, no_runs):
            print("Run ", x)
            # Initialise the metrics
            metrics['no_moves'].append(0)
            metrics['dist_moved'].append(0)

            start = time.time()
            terminated = False
            truncated = False
            env.reset()
            while not terminated and not truncated:
                metrics['no_moves'].append(1)
                # This is the real distance, for a set velocity at any point. See pigeon class
                metrics['dist_moved'].append(1000 * (0.0138888889/30))
                observation = torch.tensor(env.get_observations(), device=device, dtype=torch.float32).flatten().unsqueeze(0)
                action = torch.tensor([[policy_net(observation).max(1).indices.item()]], device=device)
                _, _, terminated, truncated = env.step(action.item())

            end = time.time()
            metrics['time_taken'].append(end-start)

            # If still alive and reached loft
            if terminated:
                metrics['bool_reached_goal'].append(True)
                metrics['bool_died'].append(False)
            else:
                metrics['bool_reached_goal'].append(False)
                if not env.pigeon.alive:
                    metrics['bool_died'].append(True)
                else:
                    metrics['bool_died'].append(False)


        loft = [f for f in env.env_orig.active_objects if f.getClass() == "Loft"][0]
        loft.image = None

        window.quit() # TODO ERRORING HERE WHEN RUN MULTIPLE LOOPS, I DO NOT THINK THIS IS BEING CALLED CORRECTLY BUT IDK WHY. NEED TO FIX.

    t1 = threading.Thread(target=run)
    t1.start()

    window.mainloop()

    return metrics

# Plot all the graphs for a list full of the dictionaries of metrics values.
def plot_graphs(metrics):
    # Metrics are stored in this form:
        # metrics = {
        #     'name': ""
        #     'no_moves': [],
        #     'dist_moved': [],
        #     'bool_reached_goal': [],
        #     'bool_died': [],
        #     'time_taken': []
        # }


    # Box plots to compare average number of moves
    # Code influenced by: https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
    box_colours = ["mistyrose", "lightyellow"]
    fig, ax = plt.subplots(1, len(metrics))
    ax.set(
        axisbelow = True,
        title='Number of Moves by Model Type',
        xlabel='Model',
        ylabel='Number of Moves',
    )

    no_moves = []
    for met in metrics:
        no_moves.append(met['no_moves'])

    all_plts = ax.boxplot(no_moves, notch=False, sym='+', orientation='vertical')

    for x in range(0, len(metrics)):
        box = all_plts['boxes'][x]

        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])

        # Put colour in th plot
        ax.add_patch(Polygon(box_coords, facecolor=box_colours[x % 2]))

        # Re add the medians as the colour blocked it.
        med = all_plts['medians'][x]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            ax.plot(median_x, median_y)


    # Box plots for distance moved
    fig, ax = plt.subplots(1, len(metrics))
    for x in range(0, len(metrics)):
        ax.boxplot(metrics[x]['dist_moved'])


    # Horizontal bar charts to show number of dead pigeons and number that reached goal
    x = np.arange(len(metrics))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for z in range(0, len(metrics)):
        no_reached_goal = 0
        no_died = 0
        for y in range(0, len(metrics[z]['bool_reached_goal'])):
            if metrics[z]['bool_reached_goal'][y]:
                no_reached_goal += 1
            if metrics[z]['bool_died'][y]:
                no_died += 1

        offset = width * multiplier
        rects = ax.barh(x + offset, no_reached_goal, width, label="No. Reached Loft")
        rects1 = ax.barh(x + offset, no_died, width, label="No. Eaten by Predator")
        ax.bar_label(rects, padding=3)
        ax.bar_label(rects1, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('No. Pigeons ')
    ax.set_title('Model Type')
    model_names = [metrics[x]['name'] for x in range(0, len(metrics))]
    ax.set_xticks(x + width, model_names)


    # Significance tests, for continuous variables (no_moves, dist_moved) to see if there is a significant difference.

    plt.show()


def run_tests():
    # metrics = []
    # metrics.append(test_DQN(1))
    #
    # with open("pickle_test.pkl", "wb") as f:
    #     pickle.dump(metrics, f)

    with open("pickle_test.pkl", "rb") as f: # TODO REMEMBER TO DELETE THIS.
        metrics = pickle.load(f)

    plot_graphs(metrics)




