import random  
import numpy as np  
import gurobipy as gp  
from gurobipy import GRB 
import matplotlib.pyplot as plt
  
  
class Request:  
    def __init__(self, tokens, arrival_time, deadline):  
        self.tokens = tokens  
        self.deadline = deadline  
        self.arrival_time = arrival_time  
        self.score = tokens / deadline  
        if deadline > arrival_time:
            self.remain = 1 / (deadline - arrival_time)  
        else:
            self.remain = 0
  
    def update_score(self, current_time):  
        if current_time < self.deadline:  
            self.score = self.tokens / (self.deadline - current_time)  
  
    def update_remaintime(self, current_time):  
        if current_time < self.deadline:  
            self.remain = 1 / (self.deadline - current_time)  
  
  
class SchedulerSimulator:  
    def __init__(self, requests, inference_delays, scheduling_policy, batching_policy):  
        self.requests = requests  
        self.inference_delays = inference_delays  
        self.current_time = 0  
        self.total_completion_time = 0 
        self.scheduling_policy = scheduling_policy  
        self.batching_policy = batching_policy
        self.iteration = 0
  
    def calculate_delay(self, batch_size):  
        if batch_size in self.inference_delays:  
            return self.inference_delays[batch_size]  
        else:  
            return float('inf')  # Assume infinite delay for batch sizes not in the list  
        
    def offline_optimal_scheduler(self, processing_requests):  
        if not processing_requests:  
            return [], 1  
        
        best_selected_requests = []
        best_batch_size = 1
        max_goodput = -1
        
        for batch_size in self.inference_delays.keys():
            model = gp.Model("OfflineOptimal")  
            model.setParam('OutputFlag', 0)  # Turn off Gurobi output

            num_requests = len(processing_requests)  
            max_iterations = int(max(req.tokens for req in processing_requests) // self.inference_delays[batch_size]) + 1
            
            x = model.addVars(num_requests, max_iterations, vtype=GRB.BINARY, name="x")  
            C = model.addVars(num_requests, vtype=GRB.BINARY, name="C")  
        
            model.setObjective(sum(C[i] for i in range(num_requests)), GRB.MAXIMIZE)  
        
            # Constraints   
            for i in range(num_requests):  
                req = processing_requests[i]  
                delay_key = min((key for key in self.inference_delays if key >= req.tokens), default=batch_size)
                iterations_required = int(req.tokens // self.inference_delays[delay_key]) + 1
                for j in range(max_iterations):  
                    if j < iterations_required:  
                        model.addConstr(x[i, j] == 1)  
                    else:  
                        model.addConstr(x[i, j] == 0)  
                model.addConstr(sum(x[i, j] for j in range(max_iterations)) >= req.tokens * C[i])  
        
            # Add time limit for the Gurobi solver (in seconds)  
            time_limit = 0.01  # Set your desired time limit here  
            model.setParam(GRB.Param.TimeLimit, time_limit) 
        
            model.update()  
            model.optimize()  
        
            if model.status == GRB.OPTIMAL:  
                selected_requests = [processing_requests[i] for i in range(num_requests) if C[i].X == 1]
                goodput = sum(1 for req in selected_requests if req.deadline >= self.current_time + self.calculate_delay(batch_size)*req.tokens)
                if goodput > max_goodput:
                    best_selected_requests = selected_requests
                    best_batch_size = batch_size
                    max_goodput = goodput
        
        return best_selected_requests, best_batch_size
    
    def scheduler(self, processing_requests):  
        selected_requests = []  
    
        # Update scores and sort based on the scheduling policy.  
        if self.scheduling_policy == 'offline optimal':  
            selected_requests, best_batch_size = self.offline_optimal_scheduler(processing_requests)  
            return selected_requests, best_batch_size 
        if self.scheduling_policy == 'bidding':  
            for req in processing_requests:  
                req.update_score(self.current_time)  
            processing_requests = sorted(processing_requests, key=lambda req: req.score, reverse=True)  
        if self.scheduling_policy == 'random':  
            # Shuffle requests for random selection.  
            random.shuffle(processing_requests)  
        if self.scheduling_policy == 'deadline':  
            for req in processing_requests:  
                req.update_remaintime(self.current_time)  
            processing_requests = sorted(processing_requests, key=lambda req: req.remain, reverse=True)  
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
                    best_batch_size = batch_size  
                    selected_requests = current_requests  
    
        return selected_requests, best_batch_size  
  
    def run_one_iteration(self, processing_requests, goodput):  
        print(self.iteration)
        self.iteration += 1
        selected_requests, batch_size = self.scheduler(processing_requests)  
  
        for req in selected_requests:  
            req.tokens -= 1  
            if req.tokens == 0:  
                processing_requests.remove(req)  
                if self.current_time < req.deadline:  
                    goodput += 1  # Increment goodput if request finishes before deadline  
        delay = self.calculate_delay(batch_size)  
        self.total_completion_time += delay  # Update total_completion_time 
  
        self.current_time += delay  # Update current_time by adding the total_delay_now  
        return delay, goodput  
  
    def run_simulation(self): 
        goodput = 0  
        processing_requests = []  
        
        interval = 100
        cnt = 0
        while self.requests or processing_requests:    
            arrived_requests = [req for req in self.requests if req.arrival_time <= self.current_time]  
            processing_requests.extend(arrived_requests)  
            self.requests = [req for req in self.requests if req not in arrived_requests]  
            #print('Total Requests:', len(self.requests), 'Processing Requests:', len(processing_requests))
  
            _, goodput = self.run_one_iteration(processing_requests, goodput)  

            #if cnt < interval:
            #    cnt += 1
            #else:
            #    cnt = 0
            #    self.plot(processing_requests, goodput)
        
        average_jct = self.total_completion_time / goodput  # Calculate average JCT 

  
        return goodput, average_jct  
  
    def generate_requests(self, num_requests, inference_delays):  
        mu = 35.403855367569996  
        sigma = 31.604314122710903   
        requests = [  
            Request(  
                tokens := max(1, int(np.random.normal(mu, sigma))),  
                arrival_time := round(random.uniform(0, num_requests*mu*inference_delays[16])),
                deadline= arrival_time + int(random.expovariate(1/(inference_delays[16] * tokens * 2)))
            )  
            for _ in range(num_requests)  
        ]  
        return requests  

    def plot(self, processing_requests, goodput):
        fig, ax = plt.subplots()
        
        # Filter out inactive requests
        active_requests = [req for req in processing_requests if req.arrival_time <= self.current_time and req.deadline >= self.current_time]

        # Add data to the plot
        x = [req.arrival_time for req in active_requests]
        y = [req.tokens for req in active_requests]
        sc = ax.scatter(x, y, c='blue', label='Active Requests')

        ax.set_xlabel('Arrival Time')
        ax.set_ylabel('Tokens')
        ax.set_title(f'Current Time: {self.current_time}, Goodput: {goodput}')
        ax.legend()

        plt.show()
