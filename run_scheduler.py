import os  
import sys  
import matplotlib.pyplot as plt  
from scheduler import SchedulerSimulator  
  
inference_delays = {1: 42.89945313, 2: 45.02945313, 4: 50.47695313, 8: 62.123125, 16: 84.1871875}  
  
def main(num_requests, inference_delays, policy):  
    # Initialize the scheduler simulator  
    simulator = SchedulerSimulator([], inference_delays, policy)  
  
    # Generate the requests  
    requests = simulator.generate_requests(num_requests, inference_delays)  
    simulator.requests = requests  
  
    # Run the simulation  
    goodput = simulator.run_simulation()  
  
    # Return the goodput  
    return goodput  
  
if __name__ == "__main__":   
    num_requests_values = list(range(100, 2100, 100))  
    goodput_values_random = []  
    goodput_values_bidding = []  
  
    # Loop through num_requests from 100 to 2000 with a step of 100 for random policy  
    for num_requests in num_requests_values:  
        goodput_random = main(num_requests, inference_delays, policy='random')  
        goodput_values_random.append(goodput_random)  
  
    # Loop through num_requests from 100 to 2000 with a step of 100 for bidding policy  
    for num_requests in num_requests_values:  
        goodput_bidding = main(num_requests, inference_delays, policy='bidding')  
        goodput_values_bidding.append(goodput_bidding)  
  
    # Plot the results  
    plt.plot(num_requests_values, goodput_values_random, marker='o', linestyle='-', linewidth=2, label='Random')  
    plt.plot(num_requests_values, goodput_values_bidding, marker='o', linestyle='-', linewidth=2, label='Bidding')  
    plt.xlabel("Number of Requests")  
    plt.ylabel("Goodput")  
    plt.title("Scheduler Simulator Goodput vs. Number of Requests")  
    plt.legend()  
    plt.grid()  
    plt.savefig("test.png")  
