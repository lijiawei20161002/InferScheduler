import matplotlib.pyplot as plt  
from scheduler import SchedulerSimulator  
import copy
import os
import glob
import random
from datetime import datetime, timedelta
import pandas as pd
from scheduler import Request, SchedulerSimulator
  
inference_delays = {1: 42.89945313, 2: 45.02945313, 4: 50.47695313, 8: 62.123125, 16: 84.1871875}      
color_map = {  
    'offline solver': 'purple',  
    'online alg': 'pink',
    'online solver': 'gray',
    'random': 'blue',  
    'bidding': 'green',  
    'fcfs': 'red',  
    'deadline': 'orange'  
}  
  
def plot_results(x_values, y_values, xlabel, ylabel, title, filename):  
    for policy, values in y_values.items():  
        if 'wo' in policy:
            plt.plot(x_values, values, color=color_map[policy.split('_')[0]], linestyle='--', label=f'{policy}')  
        else:
            plt.plot(x_values, values, color=color_map[policy.split('_')[0]], marker='*', label=f'{policy}')
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    plt.title(title)  
    plt.legend()  
    plt.grid()  
    plt.savefig(filename)  
    plt.clf()  

def parse_trace_file(filename, num_requests=None):
    """Parses the trace CSV file and generates a list of Request objects."""
    df = pd.read_csv(filename)
    if num_requests is not None:
        df = df.iloc[:num_requests]
    requests = {}
    for i, row in df.iterrows():
        id = f"{i+1}"
        timestamp = datetime.strptime(row['TIMESTAMP'][:26], '%Y-%m-%d %H:%M:%S.%f')
        generated_tokens = int(row['GeneratedTokens'])
        deadline = datetime.strptime(row['Deadline'], '%Y-%m-%d %H:%M:%S.%f')
        request = Request(str(id), generated_tokens, timestamp, deadline)
        requests[id] = request
    return requests
  
if __name__ == "__main__":  
    num_iteration_values = list(range(100, 1501, 100))  
    first_time_flag = True
    scheduling_policies = ['offline solver', 'online solver', 'random', 'bidding', 'fcfs', 'deadline']  
    batching_policies = [16]  
    goodput_values = {}  
    average_jct_values = {}  
  
    # Initialize the scheduler simulator  
    trace_file = "data/AzureLLMInferenceTrace_conv.csv"
    requests = parse_trace_file(trace_file, 500)
    simulator = SchedulerSimulator([], inference_delays, 'offline solver', 16, start=requests['1'].arrival_time)
    for scheduling_policy in scheduling_policies:
        for batch_policy in batching_policies:
            print(scheduling_policy)
            simulator.requests = copy.deepcopy(requests)
            if scheduling_policy == 'offline solver':
                simulator.call_offline_solver()
            else:
                simulator.reset(copy.deepcopy(requests), inference_delays, scheduling_policy, batch_policy)
            _, delay = simulator.run_simulation()
            key = f'{scheduling_policy}_{batch_policy}'
            goodput_values[key] = []
            for i in range(len(num_iteration_values)):
                filename = f'{scheduling_policy}_{batch_policy}.log'
                end = num_iteration_values[i]
                goodput = simulator.calculate_goodput_from_log(filename, 0, end)
                print(f'{scheduling_policy}_{batch_policy}.log', goodput)
                goodput_values[key].append(goodput)
    
    # Plot the goodput results  
    plot_results(num_iteration_values, goodput_values, "Number of Iterations", "Goodput",  
                 "Scheduler Simulator Goodput vs. Iterations", "goodput.png")  
  