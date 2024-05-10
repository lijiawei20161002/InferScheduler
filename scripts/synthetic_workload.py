import matplotlib.pyplot as plt
from scheduler import SchedulerSimulator
import copy
import os
import random
from datetime import datetime, timedelta
import pandas as pd
from scheduler import Request, SchedulerSimulator
from predictor import Predictor

# Delay values for different batch sizes
inference_delays = {
    1: 42.89945313,
    2: 45.02945313,
    4: 50.47695313,
    8: 62.123125,
    16: 84.1871875
}

# Mapping of policies to lovely colors for plotting
color_map = {
    'offline solver': 'lavender',
    'online alg': 'peachpuff',
    'online solver': 'silver',
    'random': 'cornflowerblue',
    'bidding': 'springgreen',
    'fcfs': 'crimson',
    'deadline': 'gold'
}

def plot_results(x_values, y_values, xlabel, ylabel, title, filename):
    """Function to plot and save the results."""
    print(x_values, y_values)
    plt.figure(figsize=(10,8))
    for policy, values in y_values.items():
        if 'wo' in policy:
            plt.plot(x_values, values, color=color_map[policy.split('_')[0]], linestyle='--', label=f"{policy.split('_')[0]}")
        else:
            plt.plot(x_values, values, color=color_map[policy.split('_')[0]], label=f"{policy.split('_')[0]}")
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.legend(fontsize='large')
    plt.grid()
    plt.savefig(filename)
    plt.clf()

if __name__ == "__main__":
    # Range of request sizes to test
    request_sizes = [100, 150, 200, 250, 300]
    scheduling_policies = ['offline solver', 'online solver', 'random', 'bidding', 'fcfs', 'deadline']
    batching_policies = [16]

    goodput_values = {f"{policy}_{batch}": [] for policy in scheduling_policies for batch in batching_policies}
    average_jct_values = copy.deepcopy(goodput_values)

    # Initialize the scheduler simulator
    simulator = SchedulerSimulator({}, inference_delays, 'offline solver', 16)
    requests = simulator.generate_requests(10000, inference_delays)
    simulator.set_requests(requests)
    #simulator.log_requests_to_csv()
    #predictor = Predictor()
    #predictor.train('requests_log.csv')

    for num_requests in request_sizes:
        # Generate synthetic requests
        requests = simulator.generate_requests(num_requests, inference_delays)

        for scheduling_policy in scheduling_policies:
            for batch_policy in batching_policies:
                print(f"Scheduling Policy: {scheduling_policy}, Batch Policy: {batch_policy}, Requests: {num_requests}")

                # Reset simulator
                simulator.reset(copy.deepcopy(requests), inference_delays, scheduling_policy, batch_policy)
                if scheduling_policy == 'offline solver':
                    simulator.call_offline_solver()
                
                # Execute the scheduling policy and collect goodput and average JCT
                goodput, avg_jct = simulator.run_simulation()
                key = f'{scheduling_policy}_{batch_policy}'
                goodput_values[key].append(goodput)
                average_jct_values[key].append(avg_jct)

    # Plot the goodput results
    plot_results(request_sizes, goodput_values, "Number of Requests", "Goodput",
                 "Scheduler Simulator Goodput vs. Number of Requests", "goodput.png")

    # Plot the average JCT results
    plot_results(request_sizes, average_jct_values, "Number of Requests", "Average JCT",
                 "Scheduler Simulator Average JCT vs. Number of Requests", "average_jct.png")
