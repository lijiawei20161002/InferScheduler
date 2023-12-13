import matplotlib.pyplot as plt  
from scheduler import SchedulerSimulator  
import copy
import os
import glob
  
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
  
if __name__ == "__main__":  
    num_requests_values = list(range(10, 101, 20))  
    first_time_flag = True
  
    # Define colors for each scheduling policy  
    # color_map = {  
    #     'offline solver': 'purple',  
    #     'online solver': 'pink',
    #     'random': 'blue',  
    #     'bidding': 'green',  
    #     'fcfs': 'red',  
    #     'deadline': 'orange'  
    # }  

    color_map = {  
        'offline solver': 'purple',  
        'online solver': 'pink',
        'random': 'blue',  
        'bidding': 'green',  
        'fcfs': 'red',  
        'deadline': 'orange'  
    }  
  
    scheduling_policies = ['offline solver', 'online solver', 'random', 'bidding', 'fcfs', 'deadline']  

    #scheduling_policies = ['online solver', 'random', 'bidding', 'fcfs', 'deadline']  


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
        # Generate the requests  
        original_requests = simulator.generate_requests(num_requests, inference_delays)
  
        for scheduling_policy in scheduling_policies:  
            for batch_policy in batching_policies:  
                print(scheduling_policy, batch_policy)  
                requests = copy.deepcopy(original_requests)
                simulator.reset([], inference_delays, scheduling_policy, batch_policy)
                simulator.switching_cost = 10
                #objective_metric = main(simulator, requests, cnt)
                goodput, average_jct, objective_metric = main(simulator, requests, cnt)  
                key = f'{scheduling_policy}'  
                #goodput_values[key].append(goodput)  
                #average_jct_values[key].append(average_jct)  
                objective_metrics[key].append(objective_metric)
                #os.system('python3 autoremove.py')

        cnt += 1

    # Plot the goodput results  
    plot_results(num_requests_values, goodput_values, "Number of Requests", "Goodput",  
                 #"Scheduler Simulator Goodput vs. Number of Requests", "goodput.png")  
  
    # Plot the average JCT results  
    plot_results(num_requests_values, average_jct_values, "Number of Requests", "Average JCT",  
                 #"Scheduler Simulator Average JCT vs. Number of Requests", "average_jct.png")  

    # Plot the goodput results  
    plot_results(num_requests_values, objective_metrics, "Number of Requests", "Objective",  
                 "Scheduler Simulator Objective vs. Number of Requests", "objective.png")  
  
