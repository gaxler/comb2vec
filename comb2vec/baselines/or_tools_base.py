import networkx as nx
import numpy as np

from ortools.constraint_solver import pywrapcp
from pathlib import Path
import _pickle as pkl


def graph_iterator_from_path(path, glob_str='*.pkl'):
    path = Path(path)
    for graph_pickle in path.glob(glob_str):
        with graph_pickle.open('rb') as fp:
            yield pkl.load(fp), graph_pickle.stem


class ORToolsIntegerSolutionBase(object):

    def __init__(self, wall_time, objective, solution_vector, **kwargs):
        self.wall_time = wall_time
        self.objective = objective
        self.solution_vector = solution_vector


class IntegerProgramSolverFromGraph(object):

    def __init__(self, graph):
        self.graph = graph

        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("MinVertexCover_CP", parameters)
        self.solver = solver

    def _add_ge_sum_constraint(self, iterable_of_variables, sum_value=1):
        self.solver.Add(sum(iterable_of_variables) >= sum_value)

    @staticmethod
    def numpy_adjecency_mat(adj_item):
        adj = dict(adj_item)
        N = len(adj)
        A = np.zeros((N, N))

        for first_node in adj:
            for second_node in adj[first_node]:
                A[first_node, second_node] = 1

        return A

    @staticmethod
    def solution_to_numpy_vector(optimization_vars_dict, solution):
        vector_dim = len(optimization_vars_dict)
        x = np.zeros(vector_dim, dtype=np.int32)

        for v in optimization_vars_dict:
            x[v] = solution.Value(optimization_vars_dict[v])

        return x
