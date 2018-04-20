import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import _pickle as pkl
import torch


class AdjMatSolutionPklDictDataset(Dataset):

    def __init__(self, pkl_folder, glob_str='*.pkl'):

        path = Path(pkl_folder)
        self.graphs = list(path.glob(glob_str))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph_pickle = self.graphs[idx]

        with graph_pickle.open('rb') as fp:
            graph_dict = pkl.load(fp)

        adjecency = graph_dict['adj_mat'].astype(np.float32)
        feasable_solution = graph_dict['solution'].astype(np.float32)

        adjecency = torch.from_numpy(adjecency)
        feasable_solution = torch.from_numpy(feasable_solution)

        return adjecency, feasable_solution


if __name__ == '__main__':
    test_dset = AdjMatSolutionPklDictDataset('/home/gregory/projects/nips18/data/mvc/cp_solutions_7_7')

    for workers in range(1, 16):
        loader = DataLoader(dataset=test_dset, batch_size=2, shuffle=True, num_workers=workers)

        tocs = []
        for _ in range(5):
            tic = time.time()
            for idx, (adj, sol) in enumerate(loader):
                pass
            toc = time.time() - tic
            tocs.append(toc)

        print('%d workers finished %d in %f secs' % (workers, idx, np.mean(tocs)))
        del loader
