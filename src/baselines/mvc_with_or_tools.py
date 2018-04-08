import _pickle as pkl
import numpy as np
from pathlib import Path

from or_tools_base import ORToolsIntegerSolutionBase, IntegerProgramSolverFromGraph, \
    graph_iterator_from_path


class MVCSolutions(ORToolsIntegerSolutionBase):
    pass


class MvcStats(object):

    def __init__(self, adj_matrix, solutions):
        self._adj_matrix = adj_matrix
        self._solutions = solutions

    @property
    def degrees(self):
        return self._adj_matrix.sum(0)

    @property
    def mean_degree(self):
        return np.mean(self.degrees)

    @property
    def std_degree(self):
        return np.std(self.degrees)

    @property
    def wall_time(self):
        return self._solutions[-1].wall_time

    @property
    def objective(self):
        return self._solutions[-1].objective

    def as_dict(self):
        d = {'wall_time': self.wall_time, 'objective': self.objective}
        return d


class MinVertexCoverSolverFromGraph(IntegerProgramSolverFromGraph):

    def __init__(self, graph):
        super(MinVertexCoverSolverFromGraph, self).__init__(graph=graph)
        self._adj_matrix = MinVertexCoverSolverFromGraph.numpy_adjecency_mat(graph.adjacency())

        self.decision_vars = self.generate_decision_variables()
        self.add_ge_sum_constraints()

    def generate_decision_variables(self):
        xs = {}
        for vertex_num in self.graph.nodes:
            xs[vertex_num] = self.solver.IntVar(0, 1, 'x%d' % vertex_num)
        return xs

    def add_ge_sum_constraints(self, sum_value=1):
        for edge in self.graph.edges:
            edge_decision_vars = (self.decision_vars[v] for v in edge)
            self._add_ge_sum_constraint(edge_decision_vars, sum_value=sum_value)

    def solve(self, step=1):
        solver = self.solver

        obj_expr = solver.IntVar(0, len(self.decision_vars), "obj_expr")
        solver.Add(obj_expr == sum(self.decision_vars.values()))
        objective = solver.Minimize(obj_expr, step)
        decision_builder = solver.Phase(list(self.decision_vars.values()),
                                        solver.CHOOSE_RANDOM,
                                        solver.ASSIGN_RANDOM_VALUE)
        # Create a solution collector.
        collector = solver.AllSolutionCollector()
        # Add the decision variables.
        for v in self.decision_vars.values():
            collector.Add(v)
        # Add the objective.
        collector.AddObjective(obj_expr)

        init_time = solver.WallTime()
        solver.Solve(decision_builder, [objective, collector])
        solve_time = solver.WallTime() - init_time

        solution_count = collector.SolutionCount()
        solutions = [MVCSolutions(objective=collector.ObjectiveValue(i), wall_time=solve_time,
                                  solution_vector=MinVertexCoverSolverFromGraph.solution_to_numpy_vector(
                                      self.decision_vars, collector.Solution(i)))
                     for i in range(solution_count)]
        return MvcStats(adj_matrix=self._adj_matrix, solutions=solutions)


if __name__ == '__main__':
    it = graph_iterator_from_path('/home/gregory/projects/nips18/data/mvc/nodes_100_100')

    save_path = Path('/home/gregory/projects/nips18/data/mvc/cp_solutions_100_100')

    if not save_path.exists():
        save_path.mkdir(parents=True)
    i = 0
    for g, g_description in it:
        i += 1
        test_g = MinVertexCoverSolverFromGraph(graph=g)
        stats = test_g.solve()
        for idx, solution in enumerate(stats._solutions):
            d = {'adj_mat': stats._adj_matrix, 'solution': solution.solution_vector,
                 'objective': solution.objective, 'time': solution.wall_time, 'description': g_description,
                 'degrees': stats.degrees, 'mu_deg': stats.mean_degree, 'sigma_deg': stats.std_degree}

            with (save_path / ('cp_solution_%05d_%s.pkl' % (idx, g_description))).open('wb') as fp:
                pkl.dump(d, fp)

        print('(%04d) saved %d soluyions' % (i, len(stats._solutions)))
        print('-----------------------\n')
