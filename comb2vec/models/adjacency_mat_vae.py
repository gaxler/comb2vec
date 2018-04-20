from collections import OrderedDict

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class FCLayer(nn.Module):

    def __init__(self, chans_in, chans_out, activation=nn.ReLU):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(chans_in, chans_out)
        self.act_fn = activation()

    def forward(self, x):
        return self.act_fn(self.fc(x))


class GaussianEncoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, z_dim, num_hidden=1):
        super(GaussianEncoder, self).__init__()
        self.input_to_hidden = nn.Linear(inp_dim, hid_dim)
        self.hiddens = None
        if num_hidden > 0:
            self.hiddens = nn.Sequential(OrderedDict([('fc%d' % (l + 1), FCLayer(hid_dim, hid_dim, activation=nn.ReLU))
                                                      for l in range(num_hidden)]))
        self.mu_encode = nn.Linear(hid_dim, z_dim)
        self.logvar_encode = nn.Linear(hid_dim, z_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.input_to_hidden(x))
        if self.hiddens is not None:
            h1 = self.hiddens(h1)
        return self.mu_encode(h1), self.logvar_encode(h1)


class Decoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, z_dim, num_hidden=1):
        super(Decoder, self).__init__()
        self.from_z_to_hidden = nn.Linear(z_dim, hid_dim)
        self.hiddens = None
        if num_hidden > 0:
            self.hiddens = nn.Sequential(OrderedDict([('fc%d' % (l+1), FCLayer(hid_dim, hid_dim, activation=nn.ReLU))
                                                      for l in range(num_hidden)]))
        self.hidden_to_input = nn.Linear(hid_dim, inp_dim)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        h3 = self.relu(self.from_z_to_hidden(z))
        if self.hiddens is not None:
            h3 = self.hiddens(h3)
        return self.sigmoid(self.hidden_to_input(h3))


class GaussianGraphVAE(nn.Module):
    def __init__(self, num_nodes, enc_cls=GaussianEncoder, dec_cls=Decoder, hid_dim=400, z_dim=20,
                 enc_kwargs=None, dec_kwargs=None):
        super(GaussianGraphVAE, self).__init__()

        self.inp_dim = num_nodes**2
        self.z_dim = z_dim

        if enc_kwargs is None:
            enc_kwargs = {}

        if dec_kwargs is None:
            dec_kwargs = {}

        self.encoder = enc_cls(inp_dim=self.inp_dim, hid_dim=hid_dim, z_dim=z_dim, **enc_kwargs)
        self.decoder = dec_cls(inp_dim=self.inp_dim, hid_dim=hid_dim, z_dim=z_dim, **dec_kwargs)

    def encode(self, x):
        return self.encoder.forward(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.inp_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.inp_dim), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD