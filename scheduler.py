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
import json
from datetime import datetime
from predictor import Predictor
import pandas as pd

class Request:
    def __init__(self, id, tokens, arrival_time, deadline):
        self.id = id
        self.tokens = tokens
        self.deadline = deadline
        self.arrival_time = arrival_time
        self.score = self.calculate_score()
        self.priority = self.calculate_priority()
        self.switching_cost = self.tokens/2

    def calculate_score(self): 
        time_to_deadline = (self.deadline - self.arrival_time).total_seconds()
        return self.tokens / time_to_deadline if time_to_deadline > 0 else float('inf')

    def update_score(self, current_time):
        if current_time < self.deadline:
            time_to_deadline = (self.deadline - current_time).total_seconds()
            self.score = self.tokens / time_to_deadline

    def calculate_priority(self):
        time_to_deadline = (self.deadline - self.arrival_time).total_seconds()
        if time_to_deadline > 0:
            return 1 / time_to_deadline
        else:
            return 0

    def update_priority(self, current_time):
        if current_time < self.deadline:
            time_to_deadline = (self.deadline - current_time).total_seconds()
            self.priority = 1 / time_to_deadline

class SchedulerSimulator:
    def __init__(self, requests, inference_delays, scheduling_policy, batching_policy, start=datetime.now(), planning_window_size=2000, reserve=0, predictor=Predictor(), timespan=inf):  
        self.requests = requests  
        self.inference_delays = inference_delays  
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
        self.switching_cost = 4
        #self.planning_window_size = planning_window_size
        self.planning_window_size = int(float(10*1000)/inference_delays[16])
        self.mode = 'incremental'
        options = {
            "WLSACCESSID": "eafc1f5c-8281-47e0-beeb-9f05ca1d6344",
            "WLSSECRET": "bef85e1a-fcab-4b34-b60a-687bbcb3a9df",
            "LICENSEID": 2496914,
        }
        self.env = gp.Env(params=options)
        self.reserve = reserve
        self.time_labels=[0, 1, 10, 20, 30, 60, 1000]
        self.token_labels=[0, 10, 100, 200, 500, 10000]
        self.start = start
        self.current_time = start  
        self.predictor = predictor
        self.timespan = timespan

    def set_timespan(self, timespan):
        self.timespan = timespan
        new_requests = {key: req for key, req in self.requests.items() if req.arrival_time < self.start + timedelta(milliseconds=timespan * self.inference_delays[16])}
        self.requests = new_requests

    def set_planning_window(self, size):
        self.planning_window_size = size

    def call_offline_solver(self):
        self.requests_order = self.offline_solver()

    def time2iter(self, t):
        return int((t-self.start).total_seconds()*1000//self.inference_delays[16])
        
    def reset(self, requests, inference_delays, scheduling_policy, batching_policy):
        self.requests = requests  
        self.inference_delays = inference_delays    
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
        self.current_time = self.start
    
    def calculate_delay(self, batch_size):  
        if batch_size in self.inference_delays:  
            return self.inference_delays[batch_size]  
        else:  
            return float('inf')  # Assume infinite delay for batch sizes not in the list 

    def log_info(self, info):
        with open(f'{self.scheduling_policy}_{self.batching_policy}.log', 'a') as log_file:
            log_file.write(info)
            log_file.flush()
            os.fsync(log_file.fileno())

    def format_requests_for_logging(self, requests):
        formatted_requests = []
        for req in requests:
            # Convert each request into a dictionary with datetime fields serialized to strings
            formatted_request = {
                "id": req.id,
                "tokens": req.tokens,
                "arrival_time": req.arrival_time.isoformat() if isinstance(req.arrival_time, datetime) else req.arrival_time,
                "deadline": req.deadline.isoformat() if isinstance(req.deadline, datetime) else req.deadline,
                "score": req.score,
                "priority": req.priority,
                "switching_cost": req.switching_cost
            }
            formatted_requests.append(formatted_request)
        return json.dumps(formatted_requests)

    def log_scheduler_decision(self, iteration, current_requests, selected_requests):
        """
        Logs the decision made by the scheduler at each iteration, including the current state of requests.

        Args:
            iteration (int): The current iteration of the simulation.
            current_requests (list): List of current requests waiting for processing.
            selected_requests (list): List of selected requests in the current iteration.
            batch_size (int): The batch size used in the current iteration.
        """
        formatted_current_requests = self.format_requests_for_logging(current_requests)
        formatted_selected_requests = self.format_requests_for_logging(selected_requests)
        decision_text = f'{{"Iteration": {iteration}, "Time": "{self.start+timedelta(milliseconds=(iteration-1)*self.inference_delays[16])} to {self.start+timedelta(milliseconds=iteration*self.inference_delays[16])}", '
        decision_text += f'"Current Requests": {formatted_current_requests}, '
        decision_text += f'"Selected Requests": {formatted_selected_requests}'
        decision_text += '}\n'
        decision_text += "---------------------------------\n"

        with open(f'{self.scheduling_policy}_{self.batching_policy}.log', 'a+') as log_file:
            log_file.write(decision_text)
            log_file.flush()
            os.fsync(log_file.fileno())

    def approximate_probability(self, request, x_i):
        if self.time2iter(request.deadline) == 0:
            return 9999  # Avoid division by zero
        else:
            return (request.tokens-x_i) / self.time2iter(request.deadline)
        
    def create_dataframe_from_requests(self, current_requests):
        data = {
            'TIMESTAMP': [req.arrival_time for req in current_requests],
            'GeneratedTokens': [req.tokens for req in current_requests],
            'Deadline': [req.deadline for req in current_requests]
        }
        return pd.DataFrame(data)
    
    def predict_request_distribution(self, processing_requests):
        df = self.create_dataframe_from_requests(processing_requests)
        features = self.predictor.create_feature(df)
        predictions = self.predictor.predict([features])[0].reshape(-1, 5)
        return predictions
    
    def create_virtual_requests(self, predictions):
        virtual_requests = []
        for time_bucket, tokens_bucket in np.ndindex(predictions.shape):
            count = round(predictions[time_bucket, tokens_bucket])
            for _ in range(count):
                tokens = self.token_labels[tokens_bucket]
                deadline = self.current_time + timedelta(seconds=int(self.time_labels[time_bucket]))
                virtual_requests.append(Request("virtual", tokens, self.current_time, deadline))
        return virtual_requests
        
    def online_alg(self, processing_requests):
        # Create a new model
        model = gp.Model("OnlineScheduler")

        # Parameters
        N = len(processing_requests)  # Number of requests
        T = 1000  # Maximum time step considered

        # Decision variables
        x = model.addVars(N, vtype=GRB.BINARY, name="x")  # Request inclusion in the batch
        y = model.addVars(N, T, vtype=GRB.INTEGER, name="y")  # Tokens processed by request by timestep
        s = model.addVars(T, vtype=GRB.CONTINUOUS, name="s")  # Slope variables
        z = model.addVar(vtype=GRB.CONTINUOUS, name="z")  # Largest slope variable

        # Objective function: maximize the number of selected requests minus the largest slope
        model.setObjective(gp.quicksum(x[i] for i in range(N)) - z, GRB.MAXIMIZE)

        # Constraints
        # Batch size constraint
        model.addConstr(gp.quicksum(x[i] for i in range(N)) <= self.B, "BatchSize")

        # Token processing constraints for each request by its deadline
        for i in range(N):
            for t in range(processing_requests[i].deadline, T):
                model.addConstr(
                    y[i, t] >= processing_requests[i].tokens - (processing_requests[i].deadline - t),
                    f"TokenReq_{i}_{t}"
                )

        # Slope calculation constraints
        for t in range(1, T):
            model.addConstr(s[t] == gp.quicksum(y[i, t] for i in range(N)), f"Slope_{t}")

        # Largest slope constraint
        for t in range(1, T):
            model.addConstr(z >= s[t], f"LargestSlope_{t}")

        # Solve the model
        model.optimize()

        # Check if a solution was found
        if model.status == GRB.OPTIMAL:
            selected_requests = [processing_requests[i] for i in range(N) if x[i].X > 0.5]
            largest_slope = z.X
            return selected_requests, largest_slope
        else:
            print("No optimal solution found.")
            return [], None
        
    def online_solver(self, processing_requests):  
        if len(processing_requests) == 0:
            return [], 16
        # Create a new model
        model = gp.Model("Scheduler", env=self.env)
        # Disable model output
        model.Params.LogToConsole = 0
        # Set time limit
        #model.setParam('TimeLimit', 0.1)
        model.setParam('LogFile', 'online.solver')  # Write a log file
        model.Params.Presolve = -1  # Automatic presolve level

        # Predict future requests and create virtual requests based on predictions
        predictions = self.predict_request_distribution(processing_requests)
        virtual_requests = self.create_virtual_requests(predictions)
        all_requests = processing_requests + virtual_requests

        # Define constants
        N = len(all_requests)  
        T = self.planning_window_size

        # Define decision variables for request selection, switching, and batch size
        x = model.addVars(N, T, vtype=GRB.BINARY, name="x")
        #x = model.addVars(N, vtype=GRB.BINARY, name="x")
        #s = model.addVars(N, vtype=GRB.BINARY, name="s")  # Switching variable
        finished = model.addVars(N, vtype=GRB.BINARY, name="finished")  # Request finished before deadline
        #b = self.B
        b = model.addVar(vtype=GRB.INTEGER, lb=0, name="b")  # Batch size variable
        
        # Use previous solution as intial solution
        if self.mode == 'incremental':
            if len(self.previous_selected_requests)>0:
                for i, req in enumerate(processing_requests):
                    if req in self.previous_selected_requests:
                        x[i, 0].start=1
                        #s[i].start=1

        # Set the objective
        objective = gp.quicksum(finished[i] for i in range(N)) + gp.quicksum(gp.quicksum(x[i, t] 
                            for t in range(T)) for i in range(N))
        model.setObjective(objective, GRB.MAXIMIZE)

        # Completion constraint: Request is considered finished if the tokens are processed before the deadline
        for i in range(N):
            model.addConstr(
                gp.quicksum(x[i, t] for t in range(min(T, self.time2iter(all_requests[i].deadline)))) >= all_requests[i].tokens * finished[i],
                f"Completion_{i}",
            )

        # Batch size constraint
        model.addConstr(gp.quicksum(x[i, 0] for i in range(N)) <= self.B)
        for t in range(1, T):
            model.addConstr(gp.quicksum(x[i, t] for i in range(N)) <= self.B-self.reserve)

        # Forbidding selection of virtual requests at step 0
        num_real_requests = len(processing_requests)
        for i in range(num_real_requests, N):  # Only apply to virtual requests
            model.addConstr(x[i, 0] == 0, f"Forbid_virtual_{i}_at_step_0")

        # Schedule constraint
        #for i in range(N):
        #    model.addConstr(s[i] >= 0)
        #    model.addConstr(s[i] >= x[i, 0] - (1 if processing_requests[i] in self.previous_selected_requests else 0))

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
        for i in range(len(processing_requests)):
            if (hasattr(x[i, 0], 'X') and x[i, 0].X > 0.5) or (hasattr(x[i, 0], 'Xn') and x[i, 0].Xn > 0.5):
                selected_requests.append(processing_requests[i])
        
        return selected_requests, b

    def offline_solver(self):  
        requests = list(self.requests.values())
        model = gp.Model("Scheduler", env=self.env)  # Create a new model
        model.Params.LogToConsole = 0  # Disable model output
        model.Params.Presolve = -1  # Automatic presolve level
        model.params.Threads = 0  # Using 0 gurobi will determine the number of threads automatically
        model.setParam('LogFile', 'offline.solver')  # Write a log file

        # Define constants
        N = len(self.requests)  # Number of requests
        T = min(self.timespan, max(sum([req.tokens for req in requests]), max([self.time2iter(req.deadline) for req in requests])) + 2*sum([req.tokens for req in requests])) + 100  # Max iterations
        
        # Add decision variables
        x = model.addVars(N, T, vtype=GRB.BINARY, name="x")  # Request processing at time t
        finished = model.addVars(N, vtype=GRB.BINARY, name="finished")  # Request finished before deadline

        # Set the objective to maximize the number of requests finished before the deadline
        # and finish as early as possible
        objective = gp.quicksum(finished[i] for i in range(N)) - gp.quicksum(gp.quicksum((t - self.time2iter(requests[i].arrival_time)) * x[i, t] 
                            for t in range(T)) for i in range(N))
        model.setObjective(objective, GRB.MAXIMIZE)

        # Completion constraint: Request is considered finished if the tokens are processed before the deadline
        for i in range(N):
            model.addConstr(
                gp.quicksum(x[i, t] for t in range(min(T, self.time2iter(requests[i].deadline)))) >= requests[i].tokens * finished[i],
                f"Completion_{i}",
            )
        for i in range(N):
            model.addConstr(
                gp.quicksum(x[i, t] for t in range(T)) == requests[i].tokens,
                f"Completion_{i}",
            )

        # No scheduling before arrival constraint
        for i in range(N):
            for t in range(self.time2iter(requests[i].arrival_time)+1):
                model.addConstr(x[i, t] == 0, f"Arrival_{i}_{t}")

        # Batch size constraint
        for t in range(T):
            model.addConstr(gp.quicksum(x[i, t] for i in range(N)) <= self.B)

        # Solve the model
        model.optimize()

        if model.status == GRB.INFEASIBLE:
            # Compute and print an Irreducible Inconsistent Subsystem (IIS)
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            print("\nThe following constraint(s) cannot be satisfied:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
            model.write("infeasible_model.ilp")

        # Extract the solution
        solution = {}
        for i in range(N):
            for t in range(T):
                if hasattr(x[i, t], 'X'):
                    solution[i, t] = x[i, t].X

        # Store the requests in the dictionary
        requests_order = []
        for iteration in range(T):
            selected_requests = []
            for i in range(N):
                if solution[i, iteration] > 0.5:
                    selected_requests.append(requests[i])
            requests_order.append(selected_requests)
        
        return requests_order
    
    def offline_solver_switching_cost(self):  
        requests = list(self.requests.values())
        model = gp.Model("Scheduler", env=self.env)  # Create a new model
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
        #c = model.addVars(N, T, vtype=GRB.INTEGER, name="c")  # Processed tokens variable
        print("Add variables done!")

        # Set the objective
        objective = gp.quicksum(
                gp.quicksum((t - self.time2iter(requests[i].arrival_time)) * x[i, t] 
                            for t in range(self.iteration, T + self.iteration)) 
                for i in range(N)) + gp.quicksum(
                    gp.quicksum(s[i, t] for t in range(T)) * requests[i].switching_cost
                    #gp.quicksum(s[i, t]*c[i, t] for t in range(T)) * self.switching_cost
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

        # Schedule constraint
        for i in range(N):
            for t in range(2, T):  # Starting from 2 because we need t-1 to be valid
                model.addConstr(s[i, t] <= x[i, t])
                model.addConstr(s[i, t] <= 1 - x[i, t-1])
                model.addConstr(s[i, t] >= x[i, t] - x[i, t-1])
                model.addConstr(s[i, t] >= 0)
        # Tracking processed token constraint
        #for i in range(N):
            #for t in range(2, T):  # Starting from 2 because we need t-1 to be valid
                #model.addConstr(c[i, t] == c[i, t-1] + x[i, t])
            #model.addConstr(c[i, 0] == x[i, 0])

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
                #print(solution[i, iteration])
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
        if self.scheduling_policy == 'online alg':
            if self.new_request_arrive:  # Only call if new requests have arrived
                selected_requests, best_batch_size = self.online_alg(processing_requests)
                self.new_request_arrive = False  # Reset the flag
                self.previous_selected_requests = selected_requests
                self.previous_batch_size = best_batch_size
                return selected_requests, best_batch_size
            elif self.old_request_leave:  # Only call if old requests have left
                selected_requests, best_batch_size = self.online_alg(processing_requests)
                self.old_request_leave = False  # Reset the flag
                self.previous_selected_requests = selected_requests
                self.previous_batch_size = best_batch_size
                return selected_requests, best_batch_size
            elif len(self.previous_selected_requests) == 0 and len(processing_requests)>0:
                selected_requests, best_batch_size = self.online_alg(processing_requests)
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
                    req.deadline >= self.current_time + timedelta(milliseconds=self.calculate_delay(actual_batch_size)*req.tokens)  
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
        self.log_scheduler_decision(self.iteration, processing_requests, selected_requests)
        for req in selected_requests:  
            req.tokens -= 1  
        requests_to_remove = [req for req in processing_requests if req.tokens <= 0]
        for req in requests_to_remove:
            processing_requests.remove(req)  
            self.old_request_leave = True
            if self.current_time <= req.deadline:  
                goodput += 1  # Increment goodput if request finishes before deadline  
        #delay = self.calculate_delay(batch_size) 
        delay = self.calculate_delay(16) 
        self.total_completion_time += delay  # Update total_completion_time 
  
        self.current_time += timedelta(milliseconds=delay)  # Update current_time by adding the total_delay_now 
        self.iteration += 1 
        return delay, goodput  
  
    def run_simulation(self): 
        goodput = 0  
        processing_requests = []  
        pending_tokens_over_time = []  
        requests = self.requests
        self.log_info(f"---------------------------------\n")
        self.log_info(f"N={len(requests)}\n")
        self.log_info(f"---------------------------------\n")

        while (self.iteration < self.timespan) and (len(requests)>0 or len(processing_requests)>0):   
            arrived_requests = [req for req in requests.values() if req.arrival_time < self.current_time]  
            if len(arrived_requests)>0:
                processing_requests.extend(arrived_requests)  
                self.new_request_arrive = True
                for req in arrived_requests:
                    del requests[req.id]
                    #del self.requests[req.id]
    
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
        last_arrival = datetime.now()
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
            if i % burst_period < 2:
                current_rate = high_rate if current_rate == low_rate else low_rate

            arrival_time = last_arrival + timedelta(milliseconds=int(random.expovariate(current_rate / (inference_delays[16] * mu))))
            last_arrival = arrival_time
            deadline = arrival_time + timedelta(milliseconds=int(random.expovariate(deadline_factor / (inference_delays[16] * (tokens)))+1000))
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

    def datetime_parser(self, dct):
        """Parse ISO datetime strings back into datetime objects."""
        for key, value in dct.items():
            if isinstance(value, str):
                try:
                    if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+", value):
                        dct[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass
        return dct

    def calculate_goodput_from_log(self, filename, start_iteration, end_iteration):
        goodput = 0
        # Open and read the log file for the given scheduling policy
        with open(filename, 'r') as log_file:
            for line in log_file:
                try:
                    entry = json.loads(line)
                    iteration = entry.get("Iteration")
                    time_range = entry.get("Time")
                    selected_requests = entry.get("Selected Requests")
                    if iteration is not None and start_iteration <= iteration <= end_iteration:
                        # Time parsing to get the end time
                        if time_range:
                            start_time, end_time = time_range.split(" to ")
                            end_time = datetime.fromisoformat(end_time.strip())

                        # Process selected requests
                        if selected_requests is not None:
                            for req in selected_requests:
                                if req['tokens'] == 1 and datetime.fromisoformat(req['deadline']) >= end_time:
                                    goodput += 1
                except json.JSONDecodeError as e:
                    print(line)
                    print(f"Failed to parse JSON: {str(e)}")
        return goodput

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

