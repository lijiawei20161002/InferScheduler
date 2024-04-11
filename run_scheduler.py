import matplotlib.pyplot as plt  
from scheduler import SchedulerSimulator  
import copy
import os
import glob
import random
from datetime import datetime, timedelta
import pandas as pd
from scheduler import Request, SchedulerSimulator
  
#inference_delays = {1: 0.04289945313, 2: 0.04502945313, 4: 0.05047695313, 8: 0.062123125, 16: 0.0841871875}
inference_delays = {1: 42.89945313, 2: 45.02945313, 4: 50.47695313, 8: 62.123125, 16: 84.1871875}    
  
def main(simulator, requests, start=0):    
    simulator.requests = requests  
    if scheduling_policy == 'offline solver':
        simulator.call_offline_solver()

    # Run the simulation  
    goodput, average_jct = simulator.run_simulation()  
    objective_metric = simulator.calculate_objective_from_log(scheduling_policy, start)

    # Return the goodput, average_jct and objective_metric
    return goodput, average_jct, objective_metric  
  
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

def parse_trace_file(filename, num_requests):
    """Parses the trace CSV file and generates a list of Request objects."""
    df = pd.read_csv(filename)
    df = df.iloc[:num_requests]
    requests = {}
    for i, row in df.iterrows():
        id = f"{i+1}"
        timestamp = datetime.strptime(row['TIMESTAMP'], '%Y-%m-%d %H:%M:%S.%f')
        generated_tokens = int(row['GeneratedTokens'])
        if random.random() > 0.5:
            deadline = timestamp + timedelta(seconds=int(random.expovariate(2.0 / (inference_delays[16] * generated_tokens))))
        else:
            deadline = timestamp + timedelta(seconds=int(random.expovariate(0.5 / (inference_delays[16] * generated_tokens))))
        request = Request(str(timestamp), generated_tokens, timestamp, deadline)
        requests[id] = request
    return requests
  
if __name__ == "__main__":  
    num_requests_values = list(range(100, 301, 80))  
    first_time_flag = True
    trace_file = "data/AzureLLMInferenceTrace_conv.csv"
  
    # Define colors for each scheduling policy  
    color_map = {  
        'offline solver': 'purple',  
        'online alg': 'pink',
        'online solver': 'gray',
        'random': 'blue',  
        'bidding': 'green',  
        'fcfs': 'red',  
        'deadline': 'orange'  
    }  
  
    scheduling_policies = ['offline solver', 'online solver', 'random', 'bidding', 'fcfs', 'deadline']  
    batching_policies = [16]  
  
    goodput_values = {}  
    average_jct_values = {}  
    objective_metrics = {}
    for policy in scheduling_policies:  
        for batch_policy in batching_policies:  
            key = f'{policy}'  
            goodput_values[key] = []  
            average_jct_values[key] = []  
            objective_metrics[key] = []
  
    # Initialize the scheduler simulator  
    simulator = SchedulerSimulator([], inference_delays, 'offline solver', 16)
    
    cnt = 0
    # Loop through num_requests from 100 to 2000 with a step of 100 for random policy  
    for num_requests in num_requests_values:  
        print(f'num requests: {num_requests}') 
        original_requests = simulator.generate_requests(num_requests, inference_delays)
        #original_requests = 
        for scheduling_policy in scheduling_policies:  
            for batch_policy in batching_policies:  
                print(scheduling_policy, batch_policy)  
                requests = copy.deepcopy(original_requests)
                simulator.reset([], inference_delays, scheduling_policy, batch_policy)
                simulator.switching_cost = 10
                #objective_metric = main(simulator, requests, cnt)
                goodput, average_jct, objective_metric = main(simulator, requests, cnt)  
                key = f'{scheduling_policy}'  
                goodput_values[key].append(goodput) 
                average_jct_values[key].append(average_jct)  
                objective_metrics[key].append(objective_metric)
                #os.system('python3 autoremove.py')
        cnt += 1
    
    # Plot the goodput results  
    plot_results(num_requests_values, goodput_values, "Number of Requests", "Goodput",  
                 "Scheduler Simulator Goodput vs. Number of Requests", "goodput.png")  
  
    # Plot the average JCT results  
    plot_results(num_requests_values, average_jct_values, "Number of Requests", "Average JCT",  
                 "Scheduler Simulator Average JCT vs. Number of Requests", "average_jct.png")  

    # Plot the goodput results  
    plot_results(num_requests_values, objective_metrics, "Number of Requests", "Objective",  
                 "Scheduler Simulator Objective vs. Number of Requests", "objective.png")  
  
