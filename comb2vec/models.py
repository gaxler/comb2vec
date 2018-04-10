from torch import nn

from torch.autograd import Variable


class GaussianEncoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, z_dim):
        super(GaussianEncoder, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.mu_encode = nn.Linear(hid_dim, z_dim)
        self.logvar_encode = nn.Linear(hid_dim, z_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        return self.mu_encode(h1), self.logvar_encode(h1)


class Decoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, z_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(z_dim, hid_dim)
        self.fc4 = nn.Linear(hid_dim, inp_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))


class GaussianGraphVAE(nn.Module):
    def __init__(self, num_nodes, enc_cls=GaussianEncoder, dec_cls=Decoder, hid_dim=400, z_dim=20):
        super(GaussianGraphVAE, self).__init__()

        self.inp_dim = num_nodes**2

        self.encoder = enc_cls(inp_dim=self.inp_dim, hid_dim=hid_dim, z_dim=z_dim)
        self.decoder = dec_cls(inp_dim=self.inp_dim, hid_dim=hid_dim, z_dim=z_dim)

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
