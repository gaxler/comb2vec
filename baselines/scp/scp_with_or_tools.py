import networkx as nx
from networkx.algorithms import bipartite
from ortools.constraint_solver import pywrapcp, solver_parameters_pb2

import numpy as np
import _pickle as pkl
from pathlib import Path


class SCPSolutions(object):

    def __init__(self, wall_time, objective):
        self.wall_time = wall_time
        self.objective = objective


class ScpStats(object):

    def __init__(self, graph, solutions):
        self._sets, self._items = bipartite.sets(graph)
        self.adjacency = dict(graph.adjacency())
        self._solutions = solutions

    @property
    def sets(self):
        return len(self._sets)

    @property
    def items(self):
        return len(self._items)

    def reduce_node_size(self, nodes, func):
        return func([len(self.adjacency[s]) for s in nodes])

    @property
    def reduced_set_size(self, func=np.mean):
        return self.reduce_node_size(self._sets, func=func)

    @property
    def reduced_items_size(self, func=np.mean):
        return self.reduce_node_size(self._items, func=func)

    @property
    def wall_time(self):
        return self._solutions[-1].wall_time

    @property
    def objective(self):
        return self._solutions[-1].objective

    def as_dict(self):
        d = {'wall_time': self.wall_time, 'objective': self.objective}
        return d


class SetCoverSolverFromGraph(object):

    def __init__(self, graph):
        self.graph = graph

        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("SetCover_CP", parameters)
        self.solver = solver

        self.sets, self.items = bipartite.sets(self.graph)
        self.set_variables = self.generate_set_variables()
        self.add_ge_sum_constraints()

    def graph_description(self):
        return '%d sets with universe of %d items' % (len(self.sets), len(self.items))

    def generate_set_variables(self):
        xs = {}
        for set_num in self.sets:
            xs[set_num] = self.solver.IntVar(0, 1, 'x%d' % set_num)
        return xs

    def _add_ge_sum_constraint(self, iterable_of_variables, sum_value=1):
        self.solver.Add(sum(iterable_of_variables) >= sum_value)

    def add_ge_sum_constraints(self, sum_value=1):
        for item_id in self.items:
            sets_containing_items = self.graph.neighbors(item_id)
            set_vars_to_constrain_on = [self.set_variables[idx] for idx in sets_containing_items]
            self._add_ge_sum_constraint(set_vars_to_constrain_on, sum_value=sum_value)

    def solve(self, step=1):
        solver = self.solver

        obj_expr = solver.IntVar(0, len(self.sets), "obj_expr")
        solver.Add(obj_expr == sum(self.set_variables.values()))
        objective = solver.Minimize(obj_expr, step)
        decision_builder = solver.Phase(list(self.set_variables.values()),
                                        solver.CHOOSE_RANDOM,
                                        solver.ASSIGN_RANDOM_VALUE)
        # Create a solution collector.
        collector = solver.AllSolutionCollector()
        # Add the decision variables.
        for v in self.set_variables.values():
            collector.Add(v)
        # Add the objective.
        collector.AddObjective(obj_expr)

        init_time = solver.WallTime()
        solver.Solve(decision_builder, [objective, collector])
        solve_time = solver.WallTime() - init_time

        solution_count = collector.SolutionCount()
        # print(self.graph_description())
        solutions = [SCPSolutions(objective=collector.ObjectiveValue(i), wall_time=solve_time)
                     for i in range(solution_count)]
            # for v in self.set_variables.values():
            #     print(' %s= ' % (v.DebugString()), (collector.Value(i, v)))
        return ScpStats(graph=self.graph, solutions=solutions)


def graph_iterator_from_path(path, glob_str='*.pkl'):
    path = Path(path)
    for graph_pickle in path.glob(glob_str):
        with graph_pickle.open('rb') as fp:
            yield pkl.load(fp)


if __name__ == '__main__':
    it = graph_iterator_from_path('/home/gregory/projects/nips18/data/scp/nodes_50_100')

    for g in it:
        test_g = SetCoverSolverFromGraph(graph=g)
        stats = test_g.solve()
        print('-----------------------\n')
