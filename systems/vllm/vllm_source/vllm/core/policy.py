from collections import deque
from typing import Deque
import gurobipy as gp

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


class OnlineSolverPolicy(Policy):
    def __init__(self, planning_window_size: int = 15, max_batch_size: int = 16, reserve: int = 0):
        self.planning_window_size = planning_window_size
        self.max_batch_size = max_batch_size
        self.reserve = reserve
        self.solved_priorities: Dict[int, float] = {}

    def solve_and_assign_priorities(self, now: float, seq_groups: Deque[SequenceGroup]):
        """Solve the optimization problem and assign priorities based on the solution."""
        N = len(seq_groups)
        if N == 0:
            return

        # Create a new Gurobi model
        model = gp.Model("OnlineScheduler")
        model.Params.LogToConsole = 0

        # Decision variables
        x = model.addVars(N, self.planning_window_size, vtype=gp.GRB.BINARY, name="x")
        finished = model.addVars(N, vtype=gp.GRB.BINARY, name="finished")
        b = model.addVar(vtype=gp.GRB.INTEGER, lb=0, name="b")  # Batch size variable

        # Objective: maximize the number of completed sequences plus sum of request processing
        objective = gp.quicksum(finished[i] for i in range(N)) + gp.quicksum(
            gp.quicksum(x[i, t] for t in range(self.planning_window_size)) for i in range(N))
        model.setObjective(objective, gp.GRB.MAXIMIZE)

        # Constraints
        for i, seq_group in enumerate(seq_groups):
            time_to_deadline = seq_group.metrics.deadline - now
            T = min(self.planning_window_size, int(time_to_deadline))
            #model.addConstr(gp.quicksum(x[i, t] for t in range(T)) >= seq_group.tokens * finished[i], f"Completion_{i}")
            model.addConstr(gp.quicksum(x[i, t] for t in range(T)) >= (10-seq_group.metrics.processed_token) * finished[i], f"Completion_{i}")

        # Batch size constraints
        model.addConstr(gp.quicksum(x[i, 0] for i in range(N)) <= self.max_batch_size)
        for t in range(1, self.planning_window_size):
            model.addConstr(gp.quicksum(x[i, t] for i in range(N)) <= self.max_batch_size - self.reserve)

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
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
