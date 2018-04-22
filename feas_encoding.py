import time

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from comb2vec.models.graph_nn import MLPEncoder, SolutionFaeture
from comb2vec.data_loaders.adjacency_mat import AdjMatSolutionPklDictDataset, AdjMatWithGraphEncoding
from comb2vec.utils import encode_onehot, encode_onehot_known_labels

from torch.distributions import Bernoulli

import argparse
import os

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--nodes', type=int, default=20, help='number of nodes in a graph')
parser.add_argument('--z-dim', type=int, default=20, help='Node represenation dim')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--gpus', type=str, default=None, help='CUDA_VISBLE_DEVICES setting')
parser.add_argument('--data-path', type=str, default=None, help='path to data directory')
parser.add_argument('--enc-hiddens', type=int, default=1, help='Number of hidden encoder layeers')
parser.add_argument('--dec-hiddens', type=int, default=1, help='Number of hidden decoder layeers')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.gpus is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print('Using GPUs: %s' % args.gpus)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def get_loader(path_to_data, batch_size, num_workers=3, shuffle=False, pin_memory=True):
    dataset = AdjMatSolutionPklDictDataset(pkl_folder=path_to_data)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        pin_memory=pin_memory)
    return loader


num_nodes = args.nodes
train_loader = get_loader('../data/mvc/cp_solutions_%d_%d' % (num_nodes, num_nodes), batch_size=args.batch_size,
                          shuffle=True)

d = args.z_dim
encoder = MLPEncoder(num_nodes, d, d)
sol_classification = SolutionFaeture(feature_size=d, n_hid=d, n_out=1)

if args.cuda:
    encoder.cuda()
    sol_classification.cuda()

for k, v in encoder.state_dict().items():
    print('%s: %s' % (k, v.type()))

learning_rate = 1e-3
optimizer = optim.Adam(list(encoder.parameters()) + list(sol_classification.parameters()), lr=learning_rate)


def adj_mat_to_tensors(off_diag: np.array, dtype=np.float32) -> (torch.FloatTensor, torch.FloatTensor):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    encoder.train()
    sol_classification.train()
    adjust_learning_rate(optimizer, epoch)
    train_loss = 0
    valid_sols = 0
    it_time = 0
    back_time = 0

    # for idx, (inputs, sol_tensor, is_cover, rel_rec, rel_send) in enumerate(train_loader):
    for idx, (adj_mat, solution) in enumerate(train_loader):
        it_tix = time.time()
        rel_rec, rel_send = adj_mat_to_tensors(adj_mat.numpy())
        indic = (torch.rand((solution.size(0), 1)) < 0.5).type(torch.FloatTensor)
        ss = solution.size(1)

        sol_tensor = indic * solution + (torch.ones_like(indic) - indic) * torch.stack([z[torch.randperm(ss)] for z in solution])

        inputs = torch.eye(num_nodes).unsqueeze(0).expand(adj_mat.size(0), num_nodes, num_nodes).contiguous()
        is_cover = solution_is_cover(adj_mat, sol_tensor).unsqueeze(1)

        valid_sols += is_cover.mean() * is_cover.size(0)

        inputs = Variable(inputs)
        is_cover = Variable(is_cover)
        rel_rec = Variable(rel_rec)
        rel_send = Variable(rel_send)
        sol_tensor = Variable(sol_tensor)

        if args.cuda:
            rel_rec = rel_rec.cuda()
            rel_send = rel_send.cuda()
            inputs = inputs.cuda()
            is_cover = is_cover.cuda()
            sol_tensor = sol_tensor.cuda()

        back_tix = time.time()
        optimizer.zero_grad()
        encoded_nodes = encoder(inputs, rel_rec=rel_rec, rel_send=rel_send)
        sol_codes = torch.matmul(sol_tensor.unsqueeze(1), encoded_nodes).squeeze(1)

        logits = sol_classification(sol_codes)
        loss = sol_classification.bce_loss(logits, is_cover)

        loss.backward()
        train_loss += loss.data[0] * inputs.size(0)
        optimizer.step()
        it_toc = time.time() - it_tix
        back_toc = time.time() - back_tix
        it_time += it_toc
        back_time += back_toc
        if idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Av times {:.3f}/{:.3f}'.format(
                epoch, idx * len(inputs), len(train_loader.dataset), 100. * idx / len(train_loader), loss.data[0],
                back_time / (idx + 1), it_time / (idx + 1)))

    av_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f} Average valids: {:.4f}'.format(
        epoch, av_loss, valid_sols / len(train_loader.dataset)))

    return av_loss


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        av_loss = train(epoch)
