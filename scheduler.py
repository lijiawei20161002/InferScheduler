import argparse  
import random  
import numpy as np
import itertools
  
  
class Request:  
    def __init__(self, tokens, arrival_time, deadline):  
        self.tokens = tokens  
        self.deadline = deadline  
        self.arrival_time = arrival_time  
        self.score = tokens / deadline  
  
    def update_score(self, current_time):  
        self.score = self.tokens / (self.deadline - current_time)  
  
  
class SchedulerSimulator:  
    def __init__(self, requests, inference_delays, policy):  
        self.requests = requests  
        self.inference_delays = inference_delays  
        self.current_time = 0  
        self.policy = policy
  
    def calculate_delay(self, batch_size):  
        if batch_size in self.inference_delays:  
            return self.inference_delays[batch_size]  
        else:  
            return float('inf')  # Assume infinite delay for batch sizes not in the list  
  
    def scheduler(self, processing_requests):  
        if self.policy == 'bidding':
            for req in processing_requests:  
                req.update_score(self.current_time)  
            selected_requests = sorted(processing_requests, key=lambda req: req.score, reverse=True)[:16]  
            batch_size = max(len(selected_requests), 16)  # Ensure batch size is no smaller than the number of selected requests  
        if self.policy == 'random':
            selected_requests = random.sample(processing_requests, min(len(processing_requests), 16))  
            batch_size = max(len(selected_requests), 16)  # Ensure batch size is no smaller than the number of selected requests  
        if self.policy == 'fcfs':
            selected_requests = sorted(processing_requests, key=lambda req: req.arrival_time, reverse=True)[:16]  
            batch_size = max(len(selected_requests), 16)  # Ensure batch size is no smaller than the number of selected requests  
        return selected_requests, batch_size  
  
    def run_one_iteration(self, processing_requests, goodput):    
        selected_requests, batch_size = self.scheduler(processing_requests)  
  
        for req in selected_requests:  
            req.tokens -= 1  
            if req.tokens == 0:  
                processing_requests.remove(req)  
                if self.current_time <= req.deadline:  
                    goodput += 1  # Increment goodput if request finishes before deadline  
        delay = self.calculate_delay(batch_size)   
  
        self.current_time += delay  # Update current_time by adding the total_delay_now  
        return delay, goodput  
  
    def run_simulation(self): 
        goodput = 0  
        processing_requests = []  
  
        while self.requests or processing_requests:    
            arrived_requests = [req for req in self.requests if req.arrival_time <= self.current_time]  
            processing_requests.extend(arrived_requests)  
            self.requests = [req for req in self.requests if req not in arrived_requests]  
  
            _, goodput = self.run_one_iteration(processing_requests, goodput)  
  
        return goodput  
  
    def generate_requests(self, num_requests, inference_delays):  
        mu = 35.403855367569996  
        sigma = 31.604314122710903  
        lambda_param = 1/1000  # Adjust this value to control the distribution of deadlines  
        requests = [  
            Request(  
                tokens := max(1, int(np.random.normal(mu, sigma))),  
                arrival_time := round(random.uniform(0, num_requests*mu)),
                deadline= arrival_time + inference_delays[16] * tokens * 20 + int(random.expovariate(lambda_param)),   
            )  
            for _ in range(num_requests)  
        ]  
        return requests  
  
  
if __name__ == "__main__":  
    random.seed(42)  
  
    inference_delays = {1: 42.89945313, 2: 45.02945313, 4: 50.47695313, 8: 62.123125, 16: 84.1871875}  
  
    # Generate custom requests using the generate_requests method  
    num_requests = 1000  
    parser = argparse.ArgumentParser(description="Scheduler Simulator")  
    # Add arguments  
    parser.add_argument("--num_requests", type=int, default=1000, help="Number of requests to generate") 
    parser.add_argument("--policy", type=str, default="random", help="Specify scheduling policy")   
    # Parse the arguments  
    args = parser.parse_args()  
    # Access the parsed arguments  
    num_requests = args.num_requests  
    policy = args.policy
  
    simulator = SchedulerSimulator([], inference_delays, policy)  
    requests = simulator.generate_requests(num_requests, inference_delays)  
    simulator.requests = requests  
  
    goodput = simulator.run_simulation()  
  
    print(f"Goodput: {goodput}")  

