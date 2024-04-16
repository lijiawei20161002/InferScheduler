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
    planning_windows = [10,20,30,40,50,60,70,80,90,100]  # Different planning window sizes
    reserves = [0, 2, 4, 6, 8, 10]
    goodput_values = []  # Store goodput values for each planning window size

    # Generate a fixed set of requests
    simulator = SchedulerSimulator([], inference_delays, 'online solver', 16)
    original_requests = simulator.generate_requests(num_requests, inference_delays)

    # Iterate over different planning window sizes
    #for window_size in planning_windows:
    for reserve in reserves:
        #print(f'Running simulation with planning window: {window_size}')
        print(f'Running simulation with reserve batch capacity: {reserve}')
        requests = copy.deepcopy(original_requests)
        #simulator = SchedulerSimulator(requests, inference_delays, 'online solver', 16, planning_window_size=window_size)
        simulator = SchedulerSimulator(requests, inference_delays, 'online solver', 16, reserve=reserve)
        goodput = main(simulator, requests)
        goodput_values.append(goodput)
    
    # Plot the goodput results
    plt.plot(reserves, goodput_values, marker='o', linestyle='-', color='b')
    #plt.xlabel('Planning Window Size')
    plt.xlabel('Reserve Batch Capacity')
    plt.ylabel('Goodput')
    #plt.title('Goodput vs. Planning Window Size for Online Solver')
    plt.title('Goodput vs. Reserve Batch Size for Online Solver')
    plt.grid(True)
    #plt.savefig("goodput_vs_planning_window_size.png")
    plt.savefig("goodput_vs_reserve_batch_capacity.png")
    plt.show()
