"""https://github.com/mhw32/multimodal-vae-public"""
import math

from ray.rllib.models.torch.misc import (
    normc_initializer,
    SlimFC,
)
import torch
import torch.nn as nn


class MVAE(nn.Module):
    """Variational Autoencoder.
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents, obs_len, queue_len, hidden_dim, c_group, training=True):
        super().__init__()
        self.obs_len = obs_len
        self.training = training
        self.n_latents = n_latents
        self.c_group = list(c_group.values())
        self.PoE = ProductOfExperts()
        # All agents share one encoder and decoder.
        self.belief_encoder = MessageEncoder(n_latents=n_latents, num_embeddings=queue_len + 1, hidden_dim=hidden_dim)
        self.belief_decoder = MessageDecoder(n_latents=n_latents, num_embeddings=queue_len + 1, hidden_dim=hidden_dim)

    def forward(self, obs, p):
        _mus, _logvars = [], []
        _mus_app, _logvars_app = _mus.append, _logvars.append
        for o in obs:
            if o is None:
                onem_o, onel_o = None, None
            else:
                onem_o, onel_o = self.belief_encoder(o)  # bx64
                onem_o = onem_o.view(1, 2, -1)  # bx2x32
                onel_o = onel_o.view(1, 2, -1)  # bx2x32

            _mus_app(onem_o)  # [bx2x32]xn
            _logvars_app(onel_o)  # [bx2x32]xn

        obs_recons, mus, logvars = [], [], []
        obs_app, mus_app, logvars_app = obs_recons.append, mus.append, logvars.append
        prior_mu, prior_logvar = prior_expert((1, 1, self.n_latents))
        for i, mu in enumerate(_mus):
            tmp_mu, tmp_logvar, tmp_reobs = [], [], []
            tmp_muapp, tmp_logvarapp, tmp_reobsapp = tmp_mu.append, tmp_logvar.append, tmp_reobs.append
            if mu is None:
                mus_app(None)
                logvars_app(None)
                obs_app(None)
            else:
                if p[i][0] < 0.5:
                    z = reparametrize(mu, _logvars[i], self.training)  # bx2x32
                    r_obs = self.belief_decoder(z)  # bx2x6
                    mus_app(mu.view(1, -1))
                    logvars_app(_logvars[i].view(1, -1))
                    obs_app(r_obs.view(1, -1))
                    continue
                for j in range(2):
                    poem = [prior_mu, mu[:, j, :].unsqueeze(1)]
                    poel = [prior_logvar, _logvars[i][:, j, :].unsqueeze(1)]
                    for others in self.c_group[i][j]:
                        id_schedueler = others[0]
                        id_obs = others[1]
                        if _mus[id_schedueler] is None:
                            poem.append(torch.zeros((1,1,6)))
                            poel.append(torch.zeros((1,1,6))) 
                        else:
                            poem.append(_mus[id_schedueler][:, id_obs, :].unsqueeze(1))
                            poel.append(_logvars[id_schedueler][:, id_obs, :].unsqueeze(1))
                    pd_mu, pd_logvar = self.PoE(torch.cat(poem, dim=1), torch.cat(poel, dim=1), dim=1)  # bx32
                    z = reparametrize(pd_mu, pd_logvar, self.training)  # bx32
                    tmp_muapp(pd_mu)
                    tmp_logvarapp(pd_logvar)
                    tmp_reobsapp(self.belief_decoder(z))
                mus_app(torch.cat(tmp_mu, dim=1))
                logvars_app(torch.cat(tmp_logvar, dim=1))
                obs_app(torch.cat(tmp_reobs, dim=1))

        return obs_recons, mus, logvars


class MessageEncoder(nn.Module):
    """Parametrizes q(z|y).
    We use a single inference network that encodes
    a single attribute.
    @param n_latents: integer
                      number of latent variable dimensions.
    """

    def __init__(self, n_latents, num_embeddings, hidden_dim):
        super().__init__()
        self.n_latents = n_latents

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, n_latents * 2 * 2))

        self.net.apply(init_weights)

    def forward(self, x):
        n_latents = 2 * self.n_latents
        x = self.net(x.float())
        x = x.view((-1, 2 * n_latents))
        return x[:, :n_latents], x[:, n_latents:]


class MessageDecoder(nn.Module):
    """Parametrizes p(y|z).
    We use a single generative network that decodes
    a single attribute.
    @param n_latents: integer
                      number of latent variable dimensions.
    """

    def __init__(self, n_latents, num_embeddings, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, num_embeddings))

        self.net.apply(init_weights)

    def forward(self, z):
        z = self.net(z)
        return z


class MLP(nn.Module):
    """
    input_dim: The dimension of input tensor.
    hidden_dim: The dimension of hidden layers.
    num_layers: The number of hidden layers.
    activation: The non_linearity used for each hidden layer.['relu', 'tanh', 'swish']
    last_activation: Whether to use activation for last layer.
    """

    def __init__(self, layers=None, activation='relu', last_activation=True):
        super().__init__()
        if layers is None:
            layers = []
        num_layers = len(layers)-1
        mlp_layers = []
        mlp_layers_app = mlp_layers.append
        activations = [activation] * num_layers
        if not last_activation:
            activations[-1] = None
        for i in range(num_layers):
            mlp_layers_app(SlimFC(in_size=layers[i],
                                  out_size=layers[i+1],
                                  initializer=normc_initializer(1.0),
                                  activation_fn=activations[i]))
        self.layers = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.layers(x)


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: B x M x D for M experts
    @param logvar: B x M x D for M experts
    """

    def forward(self, mu, logvar, eps=1e-8, dim=-1):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=dim) / torch.sum(T, dim=dim)
        pd_var = 1. / torch.sum(T, dim=dim)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * torch.sigmoid(x)


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu = torch.zeros(size)
    logvar = torch.log(torch.ones(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)


def reparametrize(mu, logvar, training=True):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu
