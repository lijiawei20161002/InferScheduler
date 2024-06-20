from collections import deque
from typing import Deque, Dict, List
import gurobipy as gp
import time
import sys
import numpy as np
import pandas as pd
sys.path.append('/data/jiawei_li/InferScheduler')
from predictor import Predictor
from vllm.sequence import SequenceGroup

class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        #print(seq_group.metrics.arrival_time)
        #return now - seq_group.metrics.deadline
        return now - seq_group.metrics.arrival_time

class RandomPolicy(Policy):

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        seq_groups = list(seq_groups)
        random.shuffle(seq_groups)
        return deque(seq_groups)

class DeadlinePrioritizePolicy(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return seq_group.metrics.deadline

class BiddingPolicy(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        remaining_tokens = seq_group.metrics.tokens - seq_group.metrics.processed_token
        remaining_iterations = (seq_group.metrics.deadline - now) / 0.02  
        return remaining_tokens / max(remaining_iterations, 1)

class OnlineSolverPolicy(Policy):
    def __init__(self, predictor: Predictor, planning_window_size: int = 15, max_batch_size: int = 16, reserve: int = 0):
        self.predictor = predictor
        self.planning_window_size = planning_window_size
        self.max_batch_size = max_batch_size
        self.reserve = reserve
        self.solved_priorities: Dict[int, float] = {}
        self.now = time.time()

    def create_dataframe_from_requests(self, current_requests: Deque[SequenceGroup]) -> pd.DataFrame:
        data = {
            'TIMESTAMP': [req.metrics.arrival_time for req in current_requests],
            'GeneratedTokens': [req.metrics.tokens for req in current_requests],
            'Deadline': [req.metrics.deadline for req in current_requests]
        }
        df = pd.DataFrame(data)
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
        return df

    def predict_request_distribution(self, processing_requests: Deque[SequenceGroup]) -> Dict[int, int]:
        df = self.create_dataframe_from_requests(processing_requests)
        features = self.predictor.create_feature(df)
        predictions = self.predictor.predict([features])[0]
        return predictions

    def create_virtual_requests(self, now, prediction_matrix: np.ndarray, time_labels: List[int], token_labels: List[int]) -> Deque[Dict]:
        virtual_requests = deque()
        print('========================')
        prediction_matrix = prediction_matrix.reshape(-1, 5)
        print('========================')
        print(prediction_matrix)
        for i in range(prediction_matrix.shape[0]):
            for j in range(prediction_matrix.shape[1]):
                num_requests = int(prediction_matrix[i, j])
                for _ in range(num_requests):
                    virtual_request = {
                        'request_id': f'virtual_{i}_{j}_{_}',
                        'tokens': token_labels[j],
                        'deadline': now +time_labels[i] * 0.001
                    }
                    virtual_requests.append(virtual_request)
        return virtual_requests
    
    def solve_and_assign_priorities(self, now: float, seq_groups: Deque[SequenceGroup]):
        """Solve the optimization problem and assign priorities based on the solution."""
        #prediction_matrix = self.predict_request_distribution(seq_groups)
        #virtual_requests = self.create_virtual_requests(now, prediction_matrix, self.predictor.time_labels[1:], self.predictor.token_labels[1:])
        #all_requests = list(seq_groups) + list(virtual_requests)
        all_reqeusts = list(seq_groups)

        N = len(all_requests)
        if N == 0:
            return
        T = self.planning_window_size

        # Create a new Gurobi model
        model = gp.Model("OnlineScheduler")
        model.Params.LogToConsole = 0
        model.setParam('LogFile', 'online.solver')
        model.Params.Presolve = -1

        # Decision variables
        x = model.addVars(N, T, vtype=gp.GRB.BINARY, name="x")
        finished = model.addVars(N, vtype=gp.GRB.BINARY, name="finished")
        b = self.max_batch_size

        # Objective: maximize the number of completed sequences plus sum of request processing
        objective = gp.quicksum(finished[i] for i in range(N)) + gp.quicksum(
            gp.quicksum(x[i, t] for t in range(T)) for i in range(N))
        model.setObjective(objective, gp.GRB.MAXIMIZE)

        # Constraints
        inference_time = 0.02
        for i, req in enumerate(all_requests):
            if isinstance(req, SequenceGroup):
                time_to_deadline = int((req.metrics.deadline - now)//inference_time)
                T_req = min(T, int(time_to_deadline))
                model.addConstr(
                    gp.quicksum(x[i, t] for t in range(T_req)) >= (req.metrics.tokens - req.metrics.processed_token) * finished[i],
                    f"Completion_{i}",
                )
            else:
                deadline_iter = int((req['deadline'] - now)//inference_time)
                print(req['deadline'], now, inference_time, deadline_iter)
                model.addConstr(
                    gp.quicksum(x[i, t] for t in range(deadline_iter)) >= req['tokens'] * finished[i],
                    f"Completion_{i}",
                )

        # Batch size constraints
        model.addConstr(gp.quicksum(x[i, 0] for i in range(N)) <= b)
        for t in range(1, self.planning_window_size):
            model.addConstr(gp.quicksum(x[i, t] for i in range(N)) <= b - self.reserve)

        # Solve the model
        model.optimize()

        if model.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            model.write("infeasible_model.ilp")

        # Assign priorities based on solver results
        self.solved_priorities = {}
        for i in range(N):
            priority_value = x[i, 0].X 
            #print(priority_value)
            self.solved_priorities[seq_groups[i].request_id] = priority_value

    def get_priority(self, now: float, seq_group: SequenceGroup) -> float:
        """Return the precomputed priority from the Gurobi solver results."""
        return self.solved_priorities.get(seq_group.request_id, 0)

    def sort_by_priority(self, now: float, seq_groups: Deque[SequenceGroup]) -> Deque[SequenceGroup]:
        """Solve the optimization problem and sort the sequence groups by the computed priorities."""
        self.solve_and_assign_priorities(now, seq_groups)
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))


class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS, 'online_solver': OnlineSolverPolicy}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        if policy_name == 'online_solver':
            predictor = Predictor(model_path='/data/jiawei_li/InferScheduler/models/model/random_forest.pkl')
            return cls._POLICY_REGISTRY[policy_name](predictor=predictor, **kwargs)
        else:
            return cls._POLICY_REGISTRY[policy_name](**kwargs)
