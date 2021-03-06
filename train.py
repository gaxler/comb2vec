import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from comb2vec.models.adjacency_mat_vae import GaussianGraphVAE
from comb2vec.data_loaders.adjacency_mat import AdjMatSolutionPklDictDataset

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
parser.add_argument('--nodes', type=int, default=7, help='number of nodes in a graph')
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
    dataset = AdjMatSolutionPklDictDataset(path_to_data)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        pin_memory=pin_memory)
    return loader


num_nodes = args.nodes
train_loader = get_loader('data_dir/mvc/cp_solutions_%d_%d' % (num_nodes, num_nodes), batch_size=args.batch_size,
                          shuffle=True)
graph2vec = GaussianGraphVAE(num_nodes=num_nodes, hid_dim=num_nodes * 52, z_dim=num_nodes * 3,
                             enc_kwargs={'num_hidden': args.enc_hiddens}, dec_kwargs={'num_hidden': args.dec_hiddens})

if args.cuda:
    graph2vec.cuda()

for k, v in graph2vec.state_dict().items():
    print('%s: %s' % (k, v.type()))

optimizer = optim.Adam(graph2vec.parameters(), lr=1e-3)


def train(epoch):
    graph2vec.train()
    train_loss = 0
    graph_rank = 0
    for idx, (adj_mat, solution) in enumerate(train_loader):
        adj_mat = Variable(adj_mat)
        if args.cuda:
            adj_mat = adj_mat.cuda()
        optimizer.zero_grad()
        recon_adj_mat, mu, logvar = graph2vec.forward(adj_mat)
        loss = graph2vec.loss_function(recon_adj_mat, adj_mat, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        graph_rank += adj_mat.sum(2).mean().data[0]*args.batch_size
        optimizer.step()
        if idx % args.log_interval == -1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(adj_mat), len(train_loader.dataset),
                       100. * idx / len(train_loader),
                       loss.data[0] / len(adj_mat)))

    print('====> Epoch: {} Average loss: {:.4f} Mean Rank: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset), graph_rank / len(train_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)

    sample = Variable(torch.randn(100, graph2vec.z_dim))
    if args.cuda:
        sample = sample.cuda()
    sampled_adj_mat = graph2vec.decode(sample).cpu()

    for idx, th in enumerate((0.9,)):
        sampled_adj_mat = sampled_adj_mat > th
        sampled_adj_mat = sampled_adj_mat.view(-1, num_nodes, num_nodes)
        mat = sampled_adj_mat.data.cpu().numpy()
        print(mat.shape)
        valids = np.mean([(np.allclose(m, m.T, atol=1e-8) and (np.count_nonzero(m) > 0)) for m in mat])
        avg_ranks = np.mean(np.sum(mat, axis=2), axis=1)
        print('(%d)[ %d ] %5.4f are valid matrices. %5.4f non zero. av range %4.2f' % (
        idx + 1, int(th * 100), valids, np.count_nonzero(mat) / mat.size, np.mean(avg_ranks)))
    # print('%3.2f mean rank' % (np.mean(avg_ranks)))
