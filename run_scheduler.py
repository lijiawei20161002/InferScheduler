import matplotlib.pyplot as plt  
from scheduler import SchedulerSimulator  
  
inference_delays = {1: 42.89945313, 2: 45.02945313, 4: 50.47695313, 8: 62.123125, 16: 84.1871875}  
  
def main(num_requests, inference_delays, scheduling_policy, batching_policy):  
    # Initialize the scheduler simulator  
    simulator = SchedulerSimulator([], inference_delays, scheduling_policy, batching_policy)  
  
    # Generate the requests  
    requests = simulator.generate_requests(num_requests, inference_delays)  
    simulator.requests = requests  
    simulator.update_timespan()
    if scheduling_policy == 'offline optimal':
        simulator.calculate_offline_optimal()
  
    # Run the simulation  
    goodput, average_jct = simulator.run_simulation()  
  
    # Return the goodput and average_jct  
    return goodput, average_jct  
  
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
    num_requests_values = list(range(100, 1010, 100))  
  
    # Define colors for each scheduling policy  
    color_map = {  
        'offline optimal': 'purple',  
        'repeated offline solver': 'pink',
        'random': 'blue',  
        'bidding': 'green',  
        'fcfs': 'red',  
        'deadline': 'orange'  
    }  
  
    scheduling_policies = ['offline optimal', 'random', 'bidding', 'fcfs', 'deadline']  
    batching_policies = ['dynamic batching']  
  
    goodput_values = {}  
    average_jct_values = {}  
    for policy in scheduling_policies:  
        for batch_policy in batching_policies:  
            key = f'{policy}_{"wo" if batch_policy == 16 else "w"}_dynamic_batching'  
            goodput_values[key] = []  
            average_jct_values[key] = []  
  
    # Loop through num_requests from 100 to 2000 with a step of 100 for random policy  
    for num_requests in num_requests_values:  
        for policy in scheduling_policies:  
            for batch_policy in batching_policies:  
                print(policy, batch_policy)
                goodput, average_jct = main(num_requests, inference_delays, scheduling_policy=policy, batching_policy=batch_policy)  
                key = f'{policy}_{"wo" if batch_policy == 16 else "w"}_dynamic_batching'  
                goodput_values[key].append(goodput)  
                average_jct_values[key].append(average_jct)  
  
    # Plot the goodput results  
    plot_results(num_requests_values, goodput_values, "Number of Requests", "Goodput",  
                 "Scheduler Simulator Goodput vs. Number of Requests", "goodput.png")  
  
    # Plot the average JCT results  
    plot_results(num_requests_values, average_jct_values, "Number of Requests", "Average JCT",  
                 "Scheduler Simulator Average JCT vs. Number of Requests", "average_jct.png")  
