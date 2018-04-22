import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import _pickle as pkl
import torch

from ..utils import encode_onehot_known_labels

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


def adj_mat_to_tensors(off_diag, dtype=np.float32):
    """
    Node Summation helper
    rel_rec are the column nodes rel_send are the row nodes
    """
    idx, send, rec = np.where(off_diag)
    final_mat_size = off_diag.shape[0] * off_diag.shape[1]
    rel_rec = np.array(encode_onehot_known_labels(idx * off_diag.shape[1] + rec, final_mat_size), dtype=dtype)
    rel_send = np.array(encode_onehot_known_labels(idx * off_diag.shape[1] + send, final_mat_size), dtype=dtype)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)
    return rel_rec, rel_send


def solution_is_cover(adj_mat, solution):
    """ A complementary of a cover is an independent set"""
    indep_set = torch.ones_like(solution) - solution
    indep_adj_mat = indep_set.unsqueeze(2) * adj_mat
    z = np.where(indep_adj_mat)
    selector_mat = torch.FloatTensor(encode_onehot_known_labels(z[0], num_labels=adj_mat.size(0)).T)
    count_of_valid_edges = torch.matmul(selector_mat, indep_adj_mat.permute(0, 2, 1)[z])
    is_indepe_set = torch.ones_like(count_of_valid_edges) - (count_of_valid_edges > 0).type(torch.FloatTensor)
    return is_indepe_set


class AdjMatWithGraphEncoding(AdjMatSolutionPklDictDataset):

    def __init__(self, num_nodes, pkl_folder, glob_str='*.pkl'):
        super(AdjMatWithGraphEncoding, self).__init__(pkl_folder=pkl_folder, glob_str=glob_str)
        self.num_nodes = num_nodes

    def __getitem__(self, idx):
        num_nodes = self.num_nodes

        adjecency, feasable_solution = super(AdjMatWithGraphEncoding, self).__getitem__(idx)
        rel_rec, rel_send = adj_mat_to_tensors(adjecency.numpy())
        indic = (torch.rand((feasable_solution.size(0), 1)) < 0.0025).type(torch.FloatTensor)
        ss = feasable_solution.size(1)

        sol_tensor = indic * feasable_solution + (torch.ones_like(indic) - indic) * torch.stack([z[torch.randperm(ss)] for z in feasable_solution])

        inputs = torch.eye(num_nodes).unsqueeze(0).expand(adjecency.size(0), num_nodes, num_nodes).contiguous()
        is_cover = solution_is_cover(adjecency, sol_tensor).unsqueeze(1)

        return inputs, sol_tensor, is_cover, rel_rec, rel_send




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
