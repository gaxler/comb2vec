import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from comb2vec.models import GaussianGraphVAE
from comb2vec.data_loaders import AdjMatSolutionPklDictDataset

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
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--gpus', type=str, default=None, help='CUDA_VISBLE_DEVICES setting')
parser.add_argument('--data-path', type=str, default=None, help='path to data directory')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.gpus is not None and args.cuda:
    os.environ['CUDA_VISBLE_DEVICES'] = args.gpus
    print('Using GPUs: %s' % args.gpus)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def get_loader(path_to_data, batch_size, num_workers=3, shuffle=False, pin_memory=True):
    dataset = AdjMatSolutionPklDictDataset(path_to_data)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        pin_memory=pin_memory)
    return loader


num_nodes = 100
train_loader = get_loader('data_dir/mvc/cp_solutions_%d_%d' % (num_nodes,num_nodes), batch_size=args.batch_size,
                          shuffle=True)
model = GaussianGraphVAE(num_nodes=num_nodes)

if args.cuda:
    model.cuda()

for k, v in model.state_dict().items():
    print('%s: %s' % (k, v.type()))

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for idx, (adj_mat, solution) in enumerate(train_loader):
        adj_mat = Variable(adj_mat)
        if args.cuda:
            adj_mat = adj_mat.cuda()
        optimizer.zero_grad()
        recon_adj_mat, mu, logvar = model.forward(adj_mat)
        loss = model.loss_function(recon_adj_mat, adj_mat, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(adj_mat), len(train_loader.dataset),
                       100. * idx / len(train_loader),
                       loss.data[0] / len(adj_mat)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))



for epoch in range(1, args.epochs + 1):
    train(epoch)

    sample = Variable(torch.randn(100, model.z_dim))
    if args.cuda:
        sample = sample.cuda()

    sampled_adj_mat = model.decode(sample).cpu() > 0.75
    sampled_adj_mat =sampled_adj_mat.view(-1, num_nodes, num_nodes)
    mat = sampled_adj_mat.data.cpu().numpy()
    valids = np.mean([np.allclose(m, m.T, atol=1e-8) for m in mat])
    avg_ranks = np.mean(np.sum(mat, axis=2), axis=1)
    print('(%d) %3.2f are valid matrices' % (epoch, valids))
    print('%3.2f mean rank' % (np.mean(avg_ranks)))

