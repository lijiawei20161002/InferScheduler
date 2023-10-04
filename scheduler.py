import argparse  
import random  
  
  
class Request:  
    def __init__(self, tokens, deadline, arrival_time):  
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
        return selected_requests, batch_size  
  
    def run_one_iteration(self, processing_requests, goodput):  
        total_delay_now = 0  
  
        selected_requests, batch_size = self.scheduler(processing_requests)  
  
        for req in selected_requests:  
            req.tokens -= 1  
            if req.tokens == 0:  
                processing_requests.remove(req)  
                if self.current_time <= req.deadline:  
                    goodput += 1  # Increment goodput if request finishes before deadline  
            delay = self.calculate_delay(batch_size)  
            total_delay_now += delay  
  
        return total_delay_now, goodput  
  
    def run_simulation(self):  
        goodput = 0  
        processing_requests = []  
  
        while self.requests or processing_requests:  
            self.current_time += 1  
            arrived_requests = [req for req in self.requests if req.arrival_time <= self.current_time]  
            processing_requests.extend(arrived_requests)  
            self.requests = [req for req in self.requests if req not in arrived_requests]  
  
            _, goodput = self.run_one_iteration(processing_requests, goodput)  
  
        return goodput  
  
    def generate_requests(self, num_requests, min_tokens, max_tokens, inference_delays):  
        lambda_param = 1 / 1000  # Adjust this value to control the distribution of deadlines  
        requests = [  
            Request(  
                tokens := random.randint(min_tokens, max_tokens),  
                deadline=inference_delays[1] * tokens + int(random.expovariate(lambda_param)),  
                arrival_time=round(random.uniform(0, num_requests))  
            )  
            for _ in range(num_requests)  
        ]  
        return requests  
  
  
if __name__ == "__main__":  
    random.seed(42)  
  
    inference_delays = {1: 42.89945313, 2: 45.02945313, 4: 50.47695313, 8: 62.123125, 16: 84.1871875}  
  
    # Generate custom requests using the generate_requests method  
    num_requests = 1000  
    min_tokens = 10  
    max_tokens = 100  
  
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
    requests = simulator.generate_requests(num_requests, min_tokens, max_tokens, inference_delays)  
    simulator.requests = requests  
  
    goodput = simulator.run_simulation()  
  
    print(f"Goodput: {goodput}")  
