from multiprocessing import process
import random
import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB
import matplotlib.pyplot as plt
import math, re, ast, os, glob
from datetime import datetime, timedelta
import copy
from numpy import inf

class Request:
    def __init__(self, id, tokens, arrival_time, deadline):
        self.id = id
        self.tokens = tokens
        self.deadline = deadline
        self.arrival_time = arrival_time
        self.score = tokens / deadline if deadline >0 else inf
        if deadline > arrival_time:
            self.priority = 1 / (deadline - arrival_time)
        else:
            self.priority = 0

    def update_score(self, current_time):
        if current_time < self.deadline:
            self.score = self.tokens / (self.deadline - current_time)

    def update_priority(self, current_time):
        if current_time < self.deadline:
            self.priority = 1 / (self.deadline - current_time)


class SchedulerSimulator:
    def __init__(self, requests, inference_delays, scheduling_policy, batching_policy):  
        self.requests = requests  
        self.inference_delays = inference_delays  
        self.current_time = 0  
        self.total_completion_time = 0 
        self.scheduling_policy = scheduling_policy  
        self.batching_policy = batching_policy
        self.iteration = 0
        self.B = 16
        self.alpha = 0.5
        self.new_request_arrive = False  # Flag to track arrival of new requests
        self.old_request_leave = False  # Flag to track leaving of old requests
        self.previous_selected_requests = []
        self.previous_batch_size = 16
        self.switching_cost = 0
        self.mode = 'incremental'

    def call_offline_solver(self):
        self.requests_order = self.offline_solver()

    def time2iter(self, t):
        return int(t//self.inference_delays[16])
    
    def reset(self, requests, inference_delays, scheduling_policy, batching_policy):
        self.requests = requests  
        self.inference_delays = inference_delays  
        self.current_time = 0  
        self.total_completion_time = 0 
        self.scheduling_policy = scheduling_policy  
        self.batching_policy = batching_policy
        self.iteration = 0
        self.B = 16
        self.alpha = 0.5
        self.new_request_arrive = False  # Flag to track arrival of new requests
        self.old_request_leave = False  # Flag to track leaving of old requests
        self.previous_selected_requests = []
        self.previous_batch_size = 16
        self.mode == 'incremental'
    
    def calculate_delay(self, batch_size):  
        if batch_size in self.inference_delays:  
            return self.inference_delays[batch_size]  
        else:  
            return float('inf')  # Assume infinite delay for batch sizes not in the list 

    def log_info(self, info):
        with open(self.scheduling_policy + '.log', 'a') as log_file:
            log_file.write(info)

    def log_scheduler_decision(self, iteration, current_requests, selected_requests, batch_size):
        """
        Logs the decision made by the scheduler at each iteration, including the current state of requests.

        Args:
            iteration (int): The current iteration of the simulation.
            current_requests (list): List of current requests waiting for processing.
            selected_requests (list): List of selected requests in the current iteration.
            batch_size (int): The batch size used in the current iteration.
        """
        decision_text = f"Iteration: {iteration}, Time: {(iteration-1)*self.inference_delays[16]} to {iteration*self.inference_delays[16]}\n"
        decision_text += f"Current Requests: {[req.__dict__ for req in current_requests]}\n"
        decision_text += f"Selected Requests: {[req.__dict__ for req in selected_requests]}\n"
        decision_text += f"Batch Size: {batch_size}\n"
        decision_text += "---------------------------------\n"

        with open(self.scheduling_policy + '.log', 'a') as log_file:
            log_file.write(decision_text)
        
    def online_solver(self, processing_requests):  
        if len(processing_requests) == 0:
            return [], 16
        # Create a new model
        model = gp.Model("Scheduler")
        # Disable model output
        model.Params.LogToConsole = 0
        # Set time limit
        #model.setParam('TimeLimit', 0.1)
        model.setParam('LogFile', 'online.solver')  # Write a log file
        model.Params.Presolve = -1  # Automatic presolve level

        # Define constants
        N = len(processing_requests)  # Number of requests
        T = max(max([self.time2iter(req.deadline) for req in processing_requests]), sum([req.tokens for req in processing_requests])) + max([req.tokens for req in processing_requests])  # Max iterations
        print("N:", N, "T:", T)

        # Add decision variables
        x = model.addVars(N, T, vtype=GRB.BINARY, name="x")
        if self.switching_cost:
            s = model.addVars(N, T, vtype=GRB.BINARY, name="s")  # Switching variable
            c = model.addVars(N, T, vtype=GRB.INTEGER, name="c")  # Processed tokens variable
        
        # Use previous solution as intial solution
        if self.mode == 'incremental':
            if len(self.previous_selected_requests)>0:
                for i, req in enumerate(processing_requests):
                    if req in self.previous_selected_requests:
                        for t in range(T):
                            x[i, t].start=1

        # Set the objective
        switching_cost = self.switching_cost  # Define the switching cost
        objective = gp.quicksum(
            gp.quicksum((t - self.time2iter(processing_requests[i].deadline)) * x[i, t-self.iteration] 
                        for t in range(self.iteration, T + self.iteration)) 
            for i in range(N)) + gp.quicksum(
                gp.quicksum(c[i, t-1] for t in range(T)) * switching_cost
            for i in range(N))
        model.setObjective(objective, GRB.MINIMIZE)
        # Add constraints
        # Completion constraint
        for i in range(N):
            model.addConstr(gp.quicksum(x[i, t] for t in range(T)) == processing_requests[i].tokens)
            if processing_requests[i].tokens < 0:
                print('here:', processing_requests[i].tokens)

        # No scheduling before arrival constraint
        for i in range(N):
            for t in range(self.time2iter(processing_requests[i].arrival_time)-self.iteration):
                model.addConstr(x[i, t] == 0)

        # Batch size constraint
        for t in range(T):
            model.addConstr(gp.quicksum(x[i, t] for i in range(N)) <= self.B)

        if self.switching_cost > 0:
            # Schedule constraint
            for i in range(N):
                for t in range(2, T):  # Starting from 2 because we need t-1 to be valid
                    model.addConstr(s[i, t] <= x[i, t])
                    model.addConstr(s[i, t] <= 1 - x[i, t-1])
                    model.addConstr(s[i, t] >= x[i, t] - x[i, t-1])
                    model.addConstr(s[i, t] >= 0)
            # Tracking processed token constraint
            for i in range(N):
                for t in range(2, T):  # Starting from 2 because we need t-1 to be valid
                    model.addConstr(c[i, t] == c[i, t-1] + x[i, t])
                model.addConstr(c[i, 0] == x[i, 0])

        # Solve
        model.optimize()
        if model.status == GRB.INFEASIBLE:
            # Compute and print an Irreducible Inconsistent Subsystem (IIS)
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            print("\nThe following constraint(s) cannot be satisfied:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)

            # Optionally, you could also write the IIS to a file
            model.write("infeasible_model.lp")
        
        # Store the requests in the dictionary
        selected_requests = []
        for i in range(N):
            if (hasattr(x[i, 0], 'X') and x[i, 0].X > 0.5) or (hasattr(x[i, 0], 'Xn') and x[i, 0].Xn > 0.5):
                selected_requests.append(processing_requests[i])
        
        # Store the batch size in the dictionary
        #for batch_size in self.inference_delays:
            #if batch_size >= len(selected_requests):
                #break
        batch_size = 16
        return selected_requests, batch_size
    
    def offline_solver(self):  
        requests = list(self.requests.values())
        model = gp.Model("Scheduler")  # Create a new model
        model.Params.LogToConsole = 0  # Disable model output
        #model.setParam('TimeLimit', 0.1)  # Set time limit
        model.Params.Presolve = -1  # Automatic presolve level
        model.params.Threads = 0  # Using 0 gurobi will determine the number of threads automatically
        model.setParam('LogFile', 'offline.solver')  # Write a log file

        # Define constants
        N = len(self.requests)  # Number of requests
        T = max(sum([req.tokens for req in requests]), max([self.time2iter(req.deadline) for req in requests])) + max([req.tokens for req in requests])  # Max iterations
        print("N=", N, " T=", T)

        # Add decision variables
        x = model.addVars(N, T, vtype=GRB.BINARY, name="x")
        s = model.addVars(N, T, vtype=GRB.BINARY, name="s")  # Switching variable
        c = model.addVars(N, T, vtype=GRB.INTEGER, name="c")  # Processed tokens variable
        print("Add variables done!")

        # Set the objective
        objective = gp.quicksum(
                gp.quicksum((t - self.time2iter(requests[i].deadline)) * x[i, t-self.iteration] 
                            for t in range(self.iteration, T + self.iteration)) 
                for i in range(N)) + gp.quicksum(
                    gp.quicksum(s[i, t] for t in range(T)) * self.switching_cost
                for i in range(N))
        model.setObjective(objective, GRB.MINIMIZE)
        # Add constraints
        # Completion constraint
        for i in range(N):
            model.addConstr(gp.quicksum(x[i, t] for t in range(T)) == requests[i].tokens)

        # No scheduling before arrival constraint
        for i in range(N):
            for t in range(self.time2iter(requests[i].arrival_time)+1):
                model.addConstr(x[i, t] == 0)

        # Batch size constraint
            model.addConstr(gp.quicksum(x[i, t] for i in range(N)) <= self.B)

        if self.switching_cost > 0:
            # Schedule constraint
            for i in range(N):
                for t in range(2, T):  # Starting from 2 because we need t-1 to be valid
                    model.addConstr(s[i, t] <= x[i, t])
                    model.addConstr(s[i, t] <= 1 - x[i, t-1])
                    model.addConstr(s[i, t] >= x[i, t] - x[i, t-1])
                    model.addConstr(s[i, t] >= 0)
            # Tracking processed token constraint
            for i in range(N):
                for t in range(2, T):  # Starting from 2 because we need t-1 to be valid
                    model.addConstr(c[i, t] == c[i, t-1] + x[i, t])
                model.addConstr(c[i, 0] == x[i, 0])

        # Solve
        model.optimize()

        if model.status == GRB.INFEASIBLE:
            # Compute and print an Irreducible Inconsistent Subsystem (IIS)
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            print("\nThe following constraint(s) cannot be satisfied:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)

            # Optionally, you could also write the IIS to a file
            model.write("infeasible_model.ilp")

        # Extract the solution (this is just an example to extract the x values)
        solution = {}
        for i in range(N):
            for t in range(T):
                if hasattr(x[i, t], 'X'):
                    solution[i, t] = x[i, t].X
                elif hasattr(x[i,t], 'Xn'):
                    solution[i, t] = x[i, t].Xn
        
        # Store the requests in the dictionary
        requests_order = []
        for iteration in range(T):
            selected_requests = []
            for i in range(N):
                if solution[i, iteration] > 0.5:
                    selected_requests.append(requests[i])
            requests_order.append(selected_requests)
        
        return requests_order

    def scheduler(self, processing_requests):
        selected_requests = []

        # Update scores and sort based on the scheduling policy.
        if self.scheduling_policy == 'online solver':
            if self.new_request_arrive:  # Only call if new requests have arrived
                selected_requests, best_batch_size = self.online_solver(processing_requests)
                self.new_request_arrive = False  # Reset the flag
                self.previous_selected_requests = selected_requests
                self.previous_batch_size = best_batch_size
                return selected_requests, best_batch_size
            elif self.old_request_leave:  # Only call if old requests have left
                selected_requests, best_batch_size = self.online_solver(processing_requests)
                self.old_request_leave = False  # Reset the flag
                self.previous_selected_requests = selected_requests
                self.previous_batch_size = best_batch_size
                return selected_requests, best_batch_size
            elif len(self.previous_selected_requests) == 0 and len(processing_requests)>0:
                selected_requests, best_batch_size = self.online_solver(processing_requests)
                self.previous_selected_requests = selected_requests
                self.previous_batch_size = best_batch_size
                return selected_requests, best_batch_size
            else:
                return self.previous_selected_requests, self.previous_batch_size
        if self.scheduling_policy == 'offline solver':
            selected_requests = self.requests_order[self.iteration]
            batch_size = 16
            #for batch_size in self.inference_delays:
                #if batch_size >= len(selected_requests):
                    #break
            return selected_requests, batch_size
        if self.scheduling_policy == 'bidding':
            for req in processing_requests:
                req.update_score(self.current_time)
            processing_requests = sorted(processing_requests, key=lambda req: req.score, reverse=True)
        if self.scheduling_policy == 'random':
            # Shuffle requests for random selection.
            random.shuffle(processing_requests)
        if self.scheduling_policy == 'deadline':
            for req in processing_requests:
                req.update_priority(self.current_time)
            processing_requests = sorted(processing_requests, key=lambda req: req.priority, reverse=True)
        if self.scheduling_policy == 'fcfs':
            processing_requests = sorted(processing_requests, key=lambda req: req.arrival_time)
        if self.batching_policy in self.inference_delays:
            best_batch_size = self.batching_policy
            selected_requests = processing_requests[:best_batch_size]
        if self.batching_policy == 'dynamic batching':
            # Initialize variables for finding the best batch size.  
            max_score = -1  
            best_batch_size = 1  
        
            # Iterate through all possible batch sizes.  
            for batch_size in range(1, len(self.inference_delays) + 1):  
                # Select current batch of requests for the given batch size.  
                current_requests = processing_requests[:batch_size]  
                # Round the actual batch size to the smallest profiled delay key
                actual_batch_size = min(len(current_requests), batch_size)
                for key in self.inference_delays:
                    if key >= actual_batch_size:
                        actual_batch_size = key
                        break
        
                # Calculate the number of requests that can be processed within their deadlines.  
                num_requests_within_deadline = sum(  
                    req.deadline >= self.current_time + self.calculate_delay(actual_batch_size)*req.tokens  
                    for req in current_requests  
                )  
        
                # Calculate the score (goodput) for the current batch size.  
                score = num_requests_within_deadline / self.calculate_delay(batch_size)  
        
                # Update the best batch size and selected requests if the current score is better.  
                if score > max_score:  
                    max_score = score  
                    best_batch_size = actual_batch_size  
                    selected_requests = current_requests  
    
        return selected_requests, best_batch_size  
  
    def run_one_iteration(self, processing_requests, goodput):  
        selected_requests, batch_size = self.scheduler(processing_requests)  
        self.log_scheduler_decision(self.iteration, processing_requests, selected_requests, batch_size)
        for req in selected_requests:  
            req.tokens -= 1  
        for req in processing_requests:
            if req.tokens <= 0:  
                processing_requests.remove(req)  
                self.old_request_leave = True
                if self.current_time <= req.deadline:  
                    goodput += 1  # Increment goodput if request finishes before deadline  
        #delay = self.calculate_delay(batch_size) 
        delay = self.calculate_delay(16) 
        self.total_completion_time += delay  # Update total_completion_time 
  
        self.current_time += delay  # Update current_time by adding the total_delay_now 
        self.iteration += 1 
        return delay, goodput  
  
    def run_simulation(self): 
        goodput = 0  
        processing_requests = []  
        pending_tokens_over_time = []  
        requests = list(self.requests.values())
        self.log_info(f"---------------------------------\n")
        self.log_info(f"N={len(requests)}\n")
        self.log_info(f"---------------------------------\n")

        while len(requests)>0 or len(processing_requests)>0:    
            arrived_requests = [req for req in requests if req.arrival_time <= self.current_time]  
            if len(arrived_requests)>0:
                processing_requests.extend(arrived_requests)  
                self.new_request_arrive = True
                requests = [req for req in requests if req not in arrived_requests]  
                for req in arrived_requests:
                    del self.requests[req.id]
    
            _, goodput = self.run_one_iteration(processing_requests, goodput)  
            #pending_tokens_over_time.append(self.pending_tokens(processing_requests))  # Record pending tokens
        
        if goodput > 0:
            average_jct = self.total_completion_time / goodput  # Calculate average JCT 
        else:
            average_jct = 0

        #self.plot_pending_tokens(pending_tokens_over_time)
        return goodput, average_jct  

    def generate_requests(self, num_requests, inference_delays):
        mu = 35.403855367569996
        sigma = 31.604314122710903
        last_arrival = 0
        requests = {}

        # Parameters for burstiness and urgency
        burst_period = 10  # Defines how often the arrival rate changes
        urgent_request_period = 5  # Interval for injecting urgent requests
        high_rate = 70  # High arrival rate
        low_rate = 15  # Low arrival rate
        current_rate = high_rate

        for i in range(num_requests):
            id = f"{i+1}"
            
            # Introduce short, urgent requests
            if i % urgent_request_period == 0:
                tokens = np.random.randint(1, 5)  # Urgent requests have 1 to 4 tokens
                deadline_factor = 2.0  # Tighter deadlines for urgent requests
            else:
                tokens = max(1, int(np.random.normal(mu, sigma)))
                deadline_factor = 0.5  # Normal deadlines for regular requests

            # Alternate between high and low arrival rates
            if i % burst_period < 1:
                current_rate = high_rate if current_rate == low_rate else low_rate

            arrival_time = last_arrival + int(random.expovariate(current_rate / (inference_delays[16] * mu)))
            last_arrival = arrival_time
            deadline = arrival_time + int(random.expovariate(deadline_factor / (inference_delays[16] * tokens)))
            request = Request(id, tokens, arrival_time, deadline)
            requests[id] = request

        return requests


    def pending_tokens(self, processing_requests):
        # Filter out inactive requests
        active_requests = [req for req in processing_requests if req.arrival_time <= self.current_time and req.deadline >= self.current_time]
        return sum([req.tokens for req in active_requests])
    
    def plot_pending_tokens(self, pending_tokens_over_time):
        plt.grid()
        plt.plot(pending_tokens_over_time)
        plt.title('Pending Tokens Over Time')
        plt.xlabel('Time (iterations)')
        plt.ylabel('Pending Tokens')
        filename = f"{self.scheduling_policy}_{self.batching_policy}.png"
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
        plt.clf()

    def calculate_objective_from_log(self, scheduling_policy, start=0):
        objective_metric = 0

        with open(f'{scheduling_policy}.log', 'r') as log_file:
            log_data = log_file.read()

        iterations_data = log_data.split('---------------------------------\n')

        previous_selected_requests = []
        startpoint = -1
        for iteration_data in iterations_data:
            if iteration_data.strip() == '':
                continue

            # Extract iteration number
            iteration_match = re.search(r"Iteration: (\d+)", iteration_data)
            iteration = int(iteration_match.group(1)) if iteration_match else None
            if iteration == 0:
                startpoint += 1
            if startpoint < start:
                continue
            if startpoint > start:
                return objective_metric

            # Extract selected requests
            selected_requests_match = re.search(r"Selected Requests: (.+?)\n", iteration_data)
            if selected_requests_match:
                selected_requests_str = selected_requests_match.group(1)
                selected_requests = ast.literal_eval(selected_requests_str)  # Convert string to list of dicts

                for req in selected_requests:
                    arrival_iter = self.time2iter(req['arrival_time'])
                    if iteration >= arrival_iter:
                        # Calculate the objective for each selected request
                        objective_metric += int(iteration - arrival_iter)
                    switching_cost = self.switching_cost
                    for pre in previous_selected_requests:
                        if pre['arrival_time'] == req['arrival_time'] and pre['deadline'] == req['deadline']:
                            switching_cost = 0
                            break
                    objective_metric += switching_cost
                previous_selected_requests = copy.deepcopy(selected_requests)

        return objective_metric

    def remove_files(self, pattern):
        files = glob.glob(pattern)
        for f in files:
            try:
                os.remove(f)
                print(f"Removed: {f}")
            except OSError as e:
                print(f"Error: {f} : {e.strerror}")

    def parse_log_line(self, line):
        """Parse a single line of the log file."""
        # Assuming the line format is consistent with the provided example
        # Extracting the relevant parts using string manipulation
        parts = line.split('\n')
        if len(parts)<6:
            return None
        iteration_info = parts[0].strip()
        current_requests = parts[1].split('Current Requests: ')[0].strip()
        selected_requests = parts[2].split('Selected Requests: ')[0].strip()
        batch_size = parts[3].split('Batch Size: ')[0].strip()
        return iteration_info, current_requests, selected_requests, batch_size

    def compare_log_files(self, file1, file2):
        """Compare two log files and return the differences."""
        differences = []

        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            file1_lines = f1.read().strip().split('---------------------------------')
            file2_lines = f2.read().strip().split('---------------------------------')

            # Compare line by line
            for line1, line2 in zip(file1_lines, file2_lines):
                parsed_line1 = self.parse_log_line(line1)
                parsed_line2 = self.parse_log_line(line2)

                if parsed_line1 != parsed_line2:
                    differences.append((parsed_line1, parsed_line2))

        return differences

