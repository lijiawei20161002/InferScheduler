from multiprocessing import process
import random
import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB
import matplotlib.pyplot as plt
import math


class Request:
    def __init__(self, tokens, arrival_time, deadline):
        self.tokens = tokens
        self.deadline = deadline
        self.arrival_time = arrival_time
        self.score = tokens / deadline
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

    def calculate_offline_optimal(self):
        self.requests_order = self.offline_optimal()

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
    
    def calculate_delay(self, batch_size):  
        if batch_size in self.inference_delays:  
            return self.inference_delays[batch_size]  
        else:  
            return float('inf')  # Assume infinite delay for batch sizes not in the list  
        
    def repeated_offline_scheduler(self, processing_requests):  
        if len(processing_requests) == 0:
            return [], 16
        if len(processing_requests) < 16:
            return processing_requests, 16
        # Create a new model
        model = gp.Model("Scheduler")
        # Disable model output
        #model.Params.LogToConsole = 0
        # Set time limit
        #model.setParam('TimeLimit', 0.1)
        model.setParam('LogFile', 'repeated_offline_scheduler.log')  # Write a log file

        # Define constants
        N = len(processing_requests)  # Number of requests
        T = max(sum([req.tokens for req in processing_requests])*2, (max([req.deadline for req in processing_requests])-self.iteration) * 2)  # Max iterations

        # Add decision variables
        x = model.addVars(N, T, vtype=GRB.BINARY, name="x")

        # Set the objective
        objective = gp.quicksum(
            gp.quicksum(int(t-self.time2iter(processing_requests[i].deadline)) * x[i, t-self.iteration] for t in range(self.iteration, T+self.iteration))
            for i in range(N)
        )
        model.setObjective(objective, GRB.MINIMIZE)
        # Add constraints
        # Completion constraint
        for i in range(N):
            model.addConstr(gp.quicksum(x[i, t] for t in range(T)) == processing_requests[i].tokens)

        # No scheduling before arrival constraint
        for i in range(N):
            for t in range(self.time2iter(processing_requests[i].arrival_time)+1):
                model.addConstr(x[i, t] == 0)

        # Batch size constraint
        for t in range(T):
            model.addConstr(gp.quicksum(x[i, t] for i in range(N)) <= self.B)

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
        
        # Store the requests in the dictionary
        selected_requests = []
        for i in range(N):
            if (hasattr(x[i, 0], 'X') and x[i, 0].X > 0.5) or (hasattr(x[i, 0], 'Xn') and x[i, 0].Xn > 0.5):
                selected_requests.append(processing_requests[i])
        
        # Store the batch size in the dictionary
        for batch_size in self.inference_delays:
            if batch_size >= len(selected_requests):
                break
        return selected_requests, batch_size
    
    def offline_optimal(self):  
        model = gp.Model("Scheduler")  # Create a new model
        #model.Params.LogToConsole = 0  # Disable model output
        #model.setParam('TimeLimit', 0.1)  # Set time limit
        model.Params.Presolve = 2  # Aggressive presolve
        model.params.Threads = 0  # Using 0 gurobi will determine the number of threads automatically
        model.setParam('LogFile', 'offline_optimal.log')  # Write a log file

        # Define constants
        N = len(self.requests)  # Number of requests
        T = self.time2iter(max([req.deadline for req in self.requests]))*2  # Max iterations
        print("N=", N, " T=", T)

        # Add decision variables
        x = model.addVars(N, T, vtype=GRB.BINARY, name="x")
        print("Add variables done!")

        # Set the objective
        objective = gp.quicksum(
            gp.quicksum(int(t-self.time2iter(self.requests[i].deadline)) * x[i, t] for t in range(self.time2iter(self.requests[i].deadline), T))
            for i in range(N)
        )
        model.setObjective(objective, GRB.MINIMIZE)
        # Add constraints
        # Completion constraint
        for i in range(N):
            model.addConstr(gp.quicksum(x[i, t] for t in range(T)) == self.requests[i].tokens)

        # No scheduling before arrival constraint
        for i in range(N):
            for t in range(self.time2iter(self.requests[i].arrival_time)+1):
                model.addConstr(x[i, t] == 0)

        # Batch size constraint
        for t in range(self.iteration, T):
            model.addConstr(gp.quicksum(x[i, t] for i in range(N)) <= self.B)

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
                    selected_requests.append(self.requests[i])
            requests_order.append(selected_requests)
        
        return requests_order

    def scheduler(self, processing_requests):
        selected_requests = []

        # Update scores and sort based on the scheduling policy.
        if self.scheduling_policy == 'repeated offline solver':
            if self.new_request_arrive:  # Only call if new requests have arrived
                selected_requests, best_batch_size = self.repeated_offline_scheduler(processing_requests)
                self.new_request_arrive = False  # Reset the flag
                self.previous_selected_requests = selected_requests
                self.previous_batch_size = best_batch_size
                return selected_requests, best_batch_size
            elif self.old_request_leave:  # Only call if old requests have left
                selected_requests, best_batch_size = self.repeated_offline_scheduler(processing_requests)
                self.old_request_leave = False  # Reset the flag
                self.previous_selected_requests = selected_requests
                self.previous_batch_size = best_batch_size
                return selected_requests, best_batch_size
            elif len(self.previous_selected_requests) == 0 and len(processing_requests)>0:
                selected_requests, best_batch_size = self.repeated_offline_scheduler(processing_requests)
                print(processing_requests[0].tokens, len(processing_requests), len(selected_requests))
                self.previous_selected_requests = selected_requests
                self.previous_batch_size = best_batch_size
                return selected_requests, best_batch_size
            else:
                return self.previous_selected_requests, self.previous_batch_size
        if self.scheduling_policy == 'offline optimal':
            selected_requests = self.requests_order[self.iteration]
            for batch_size in self.inference_delays:
                if batch_size >= len(selected_requests):
                    break
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
        for req in selected_requests:  
            req.tokens -= 1  
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

        while len(self.requests)>0 or len(processing_requests)>0:    
            arrived_requests = [req for req in self.requests if req.arrival_time <= self.current_time]  
            if len(arrived_requests)>0:
                processing_requests.extend(arrived_requests)  
                self.new_request_arrive = True
                self.requests = [req for req in self.requests if req not in arrived_requests]  
                # print('Total Requests:', len(self.requests), 'Processing Requests:', len(processing_requests))
    
            _, goodput = self.run_one_iteration(processing_requests, goodput)  
            pending_tokens_over_time.append(self.pending_tokens(processing_requests))  # Record pending tokens
        
        if goodput > 0:
            average_jct = self.total_completion_time / goodput  # Calculate average JCT 
        else:
            average_jct = 0

        #self.plot_pending_tokens(pending_tokens_over_time)
        return goodput, average_jct  
  
    def generate_requests(self, num_requests, inference_delays):  
        mu = 35.403855367569996  
        sigma = 31.604314122710903   
        requests = [  
            Request(  
                tokens := max(1, int(np.random.normal(mu, sigma))),  
                arrival_time := round(random.uniform(0, num_requests*mu*inference_delays[16])),
                deadline= round(arrival_time + int(random.expovariate(1/(inference_delays[16] * tokens * 2))))
            )  
            for _ in range(num_requests)  
        ]  
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
