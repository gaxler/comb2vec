"""
Adapted from:

@article{dai2017learning,
  title={Learning Combinatorial Optimization Algorithms over Graphs},
  author={Dai, Hanjun and Khalil, Elias B and Zhang, Yuyu and Dilkina, Bistra and Song, Le},
  journal={arXiv preprint arXiv:1704.01665},
  year={2017}
}
"""

import _pickle as cp
import sys
from pathlib import Path

import networkx as nx
import numpy as np

if __name__ == '__main__':
    save_dir = None
    max_n = None
    min_n = None
    num_graph = None
    p = None
    graph_type = None
    m = None
    isConnected = 0
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-save_dir':
            save_dir = sys.argv[i + 1]
        if sys.argv[i] == '-max_n':
            max_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-min_n':
            min_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-num_graph':
            num_graph = int(sys.argv[i + 1])
        if sys.argv[i] == '-p':
            p = float(sys.argv[i + 1])
        if sys.argv[i] == '-graph_type':
            graph_type = sys.argv[i + 1]
        if sys.argv[i] == '-m':
            m = int(sys.argv[i + 1])
        if sys.argv[i] == '-connected':
            isConnected = int(sys.argv[i + 1])

    assert save_dir is not None
    assert max_n is not None
    assert min_n is not None
    assert num_graph is not None
    assert p is not None
    graph_types = {0: 'erdos_renyi', 1:'erdos_renyi', 2:'barabasi_albert'}

    seed = 1
    np.random.seed(seed=seed)

    m = np.random.randint(1, max_n)

    save_dir = str(Path(save_dir) / ('nodes_%s_%s' % (min_n, max_n)))

    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)

    print("Final Output: %s/gtype-%s-nrange-%d-%d-n_graph-%d-p-%.2f-m-%d.pkl" % (

    save_dir, graph_type, min_n, max_n, num_graph, p, m))
    print("Generating graphs...")
    g_list = []
    numgenerated = 0
    i = 0
    while numgenerated < num_graph:

        i += 1
        cur_n = np.random.randint(max_n - min_n + 1) + min_n

        graph_type = graph_types[np.random.randint(3)]

        if graph_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=p, seed=seed + i)
        elif graph_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=m, p=p, seed=seed + i)
        elif graph_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=m, seed=seed + i)

        if isConnected:
            # get largest connected component
            g_idx = max(nx.connected_components(g), key=len)
            gcc = g.subgraph(list(g_idx))
            # generate another graph if this one has fewer nodes than min_n
            if nx.number_of_nodes(gcc) < min_n:
                print("here")
                continue

            max_idx = max(gcc.nodes())
            if max_idx != nx.number_of_nodes(gcc) - 1:
                idx_map = {}
                for idx in gcc.nodes():
                    t = len(idx_map)
                    idx_map[idx] = t

                g = nx.Graph()
                g.add_nodes_from(range(0, nx.number_of_nodes(gcc)))
                for edge in gcc.edges():
                    g.add_edge(idx_map[edge[0]], idx_map[edge[1]])
                gcc = g
            max_idx = max(gcc.nodes())
            assert max_idx == nx.number_of_nodes(gcc) - 1

            # check number of nodes in induced subgraph
            numnodes = (len(list(nx.bfs_tree(gcc, gcc.nodes()[0]))))
            if numnodes < min_n or numnodes > max_n:
                print("invalid graph generated!")
                print(numnodes)
                sys.exit()

            g = gcc

        numgenerated += 1

        with open('%s/gtype-%s-nrange-%d-%d-n_graph-%d-p-%.2f-m-%d.pkl' % (
        save_dir, graph_type, min_n, max_n, numgenerated, p, m), 'ab') as fout:
            cp.dump(g, fout)

        print('Generated %d, iterations %d' %(numgenerated, i))
