from pathlib import Path
import _pickle as pkl

from mvc_with_or_tools import graph_iterator_from_path, MinVertexCoverSolverFromGraph

if __name__ == '__main__':
    it = graph_iterator_from_path('/home/gregory/Academy/comb2vec/data/mvc/nodes_5_5')

    save_path = Path('/home/gregory/Academy/comb2vec/data/mvc/cp_solutions_5_5')

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

        print('(%04d) saved %d solutions' % (i, len(stats._solutions)))
        print('-----------------------\n')
