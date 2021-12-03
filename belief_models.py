"""https://github.com/mhw32/multimodal-vae-public"""

import torch
import torch.nn as nn


# class MVAE(nn.Module):
#     """Multimodal Variational Autoencoder.
#     @param n_latents: integer
#                       number of latent dimensions
#     """
#
#     def __init__(self, n_latents, n_modalities, n_embeddings, obs_len, schedulers, training, batch_size, comm_group):
#         super().__init__()
#         self.n_modalities = n_modalities
#         self.schedulers = schedulers
#         self.training = training
#         self.bs = batch_size
#         self.comm_group = comm_group
#         # Each agent has encoders ang decoders for each observing queue.
#         self.belief_encoders = nn.ModuleList(
#             [nn.ModuleList([AgentEncoder(n_latents, n_embeddings) for _ in range(obs_len)]) for _ in range(n_modalities)]
#         )
#         self.belief_decoders = nn.ModuleList(
#             [nn.ModuleList([AgentDecoder(n_latents) for _ in range(obs_len)]) for _ in range(n_modalities)]
#         )
#         self.experts = ProductOfExperts()
#         self.n_latents = n_latents
#
#     def reparametrize(self, mu, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = std.data.new(std.size()).normal_()
#             # return torch.clamp(eps.mul(std).add_(mu), 0)
#             # return F.softmax(eps.mul(std).add_(mu), dim=1)
#             return eps.mul(std).add_(mu)
#         else:  # return mean during inference
#             # return torch.clamp(mu, 0)
#             # return F.softmax(mu, dim=1)
#             return mu
#
#     def forward(self, p, obs):
#         """Forward pass through the MVAE.
#         @param obs: list of ?PyTorch.Tensors: NxL,
#                       where N is the number of schedulers and L is the length of observations.
#                       If a single belief is missing, pass None
#                       instead of a Tensor. Regardless if all beliefs
#                       are missing, still pass a list of <n_modalities> None's.
#         @param p: dict
#                       Probability, which decide whether to get the messages.
#         @return obs_recons: list of PyTorch.Tensors (n_modalities length)
#         """
#         mu, logvar = self.infer(p, obs)
#         obs_recons = []
#         # reparametrization trick to sample
#         for i, (_mu, _logvar) in enumerate(zip(mu, logvar)):
#             if _mu is None:
#                 obs_recons.append(None)
#                 continue
#             z = [self.reparametrize(__mu, __logvar) for __mu, __logvar in zip(_mu, _logvar)]
#             obs_recon = []
#             for j, _z in enumerate(z):
#                 obs_recon.append(self.belief_decoders[i][j](_z))
#             obs_recons.append(obs_recon)
#         return obs_recons, mu, logvar
#
#     def infer(self, p, obs):
#         # get the batch size
#         batch_size = self.bs
#
#         use_cuda = next(self.parameters()).is_cuda  # check if CUDA
#
#         # Compute the distributions of all the schedulers.
#         mus = []
#         logvars = []
#         for i, encoders in enumerate(self.belief_encoders):
#             if obs[i] is None:
#                 mus.append(None)
#                 logvars.append(None)
#                 continue
#             _mus = []
#             _logvars = []
#             _input = obs[i].long()
#             for j, encoder in enumerate(encoders):
#                 belief_mu, belief_logvar = encoder(_input[j].unsqueeze(0))
#                 _mus.append(belief_mu)
#                 _logvars.append(belief_logvar)
#             mus.append(_mus)
#             logvars.append(_logvars)
#         _index_map = {scheduler: i for i, scheduler in enumerate(self.schedulers)}
#
#         f_mu, f_logvar = [], []
#         # Each scheduler has a communication group, in which the agents share the same partial access.
#         for scheduler in self.schedulers:
#             # If the scheduler is done, then skip this scheduler.
#             if mus[_index_map[scheduler]] is None:
#                 f_mu.append(None)
#                 f_logvar.append(None)
#                 continue
#             comm_group = self.comm_group[scheduler]
#             _f_mu, _f_logvar = [], []
#             # Each queue has a group of schedulers, which can access to it.
#             for k, group in enumerate(comm_group):
#                 mu, logvar = prior_expert((1, batch_size, self.n_latents), use_cuda=use_cuda)
#                 mu = torch.cat((mu, mus[_index_map[scheduler]][k].unsqueeze(0)), dim=0)
#                 logvar = torch.cat((logvar, logvars[_index_map[scheduler]][k].unsqueeze(0)), dim=0)
#                 if not scheduler.silent and p[scheduler.name][k] > 0.5:
#                     server = scheduler.obs_servers[k]
#                     for member in group:
#                         if mus[_index_map[member]] is None or p[member.name][k] > 0.5:
#                             continue
#                         idx = member.obs_servers.index(server)
#                         mu = torch.cat((mu, mus[_index_map[member]][idx].unsqueeze(0)), dim=0)
#                         logvar = torch.cat((logvar, logvars[_index_map[member]][idx].unsqueeze(0)), dim=0)
#                 # product of experts to combine gaussians
#                 mu, logvar = self.experts(mu, logvar)
#                 _f_mu.append(mu)
#                 _f_logvar.append(logvar)
#             f_mu.append(_f_mu)
#             f_logvar.append(_f_logvar)
#         return f_mu, f_logvar
#
#     def show_grad(self):
#         print('encoder')
#         for name, weight in self.belief_encoders[0][0].named_parameters():
#             if weight.requires_grad:
#                 print(name, '_weight_grad', weight.grad.mean(), weight.grad.min(), weight.grad.max())
#         print('decoder')
#         for name, weight in self.belief_decoders[0][0].named_parameters():
#             if weight.requires_grad:
#                 (name, '_weight_grad', weight.grad.mean(), weight.grad.min(), weight.grad.max())


class VAE(nn.Module):
    """Variational Autoencoder.
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents, obs_len, queue_len, hidden_dim, training=True):
        super().__init__()
        self.obs_len = obs_len
        self.training = training
        self.n_latents = n_latents
        # All agents share one encoder and decoder.
        self.belief_encoder = AgentEncoder(n_latents=n_latents, num_embeddings=queue_len+1, hidden_dim=hidden_dim)
        self.belief_decoder = AgentDecoder(n_latents=n_latents, num_embeddings=queue_len+1, hidden_dim=hidden_dim)

    def forward(self, obs):
        obs_recons, mus, logvars = [], [], []
        obs_app, mus_app, logvars_app = obs_recons.append, mus.append, logvars.append
        for o in obs:
            if o is None:
                mu, logvar, obs_recon = None, None, None
            else:
                _obs_recons, _mus, _logvars = [], [], []
                _obs_app, _mus_app, _logvars_app = _obs_recons.append, _mus.append, _logvars.append
                for i in range(self.obs_len):
                    _mu, _logvar = self.belief_encoder(o[i])
                    _z = reparametrize(_mu, _logvar, self.training)
                    _obs_app(self.belief_decoder(_z))
                    _mus_app(_mu)
                    _logvars_app(_logvar)
                mu = torch.cat(_mus, dim=1)
                logvar = torch.cat(_logvars, dim=1)
                obs_recon = torch.cat(_obs_recons, dim=1)
            obs_app(obs_recon)
            mus_app(mu)
            logvars_app(logvar)
        return obs_recons, mus, logvars


class AgentEncoder(nn.Module):
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
            nn.Embedding(num_embeddings, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, n_latents * 2))

        self.net.apply(init_weights)

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x.long())
        x = x.view((-1, 2*n_latents))
        return x[:, :n_latents], x[:, n_latents:]


class AgentDecoder(nn.Module):
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
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, num_embeddings))

        self.net.apply(init_weights)

    def forward(self, z):
        z = self.net(z)
        return z


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
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

