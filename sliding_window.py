import matplotlib.pyplot as plt
from scheduler import SchedulerSimulator
import copy

inference_delays = {1: 42.89945313, 2: 45.02945313, 4: 50.47695313, 8: 62.123125, 16: 84.1871875}

def main(simulator, requests):    
    simulator.requests = requests  
    # Run the simulation  
    goodput, average_jct = simulator.run_simulation()  
    return goodput

if __name__ == "__main__":  
    num_requests = 200  # Fixed number of requests for simplicity
    planning_windows = [500, 1000, 1500, 2000, 2500]  # Different planning window sizes
    goodput_values = []  # Store goodput values for each planning window size

    # Generate a fixed set of requests
    simulator = SchedulerSimulator([], inference_delays, 'online solver', 16)
    original_requests = simulator.generate_requests(num_requests, inference_delays)

    # Iterate over different planning window sizes
    for window_size in planning_windows:
        print(f'Running simulation with planning window: {window_size}')
        requests = copy.deepcopy(original_requests)
        simulator = SchedulerSimulator(requests, inference_delays, 'online solver', 16, planning_window_size=window_size)
        goodput = main(simulator, requests)
        goodput_values.append(goodput)
    
    # Plot the goodput results
    plt.plot(planning_windows, goodput_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Planning Window Size')
    plt.ylabel('Goodput')
    plt.title('Goodput vs. Planning Window Size for Online Solver')
    plt.grid(True)
    plt.savefig("goodput_vs_planning_window_size.png")
    plt.show()
