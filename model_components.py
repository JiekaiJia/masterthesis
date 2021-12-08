"""https://github.com/mhw32/multimodal-vae-public"""

from ray.rllib.models.torch.misc import (
    normc_initializer,
    SlimFC,
)
import torch
import torch.nn as nn


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
        self.belief_encoder = MessageEncoder(n_latents=n_latents, num_embeddings=queue_len + 1, hidden_dim=hidden_dim)
        self.belief_decoder = MessageDecoder(n_latents=n_latents, num_embeddings=queue_len + 1, hidden_dim=hidden_dim)

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
            nn.Embedding(num_embeddings, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, n_latents * 2))

        self.net.apply(init_weights)

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x.long())
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

    def forward(self, mu, logvar, eps=1e-8, dim=-1, weighting=True):
        normalized_weights = 1
        if weighting:
            # computing weight
            weight_matrix = torch.clamp(0.5 * (logvar[:, 0, :].unsqueeze(1) - logvar + eps), min=0)
            sum_weights = torch.sum(weight_matrix, keepdim=True, dim=dim)
            normalized_weights = weight_matrix / sum_weights
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * normalized_weights * T, dim=dim) / torch.sum(normalized_weights * T, dim=dim)
        pd_var = 1. / torch.sum(normalized_weights * T, dim=dim)
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

# todo: implement weights in PoE
# def compute_weights(mus, logvars, power, weighting, prior_var=None, softmax=False):
#     """ Compute unnormalized weight matrix
#     Inputs :
#             -- mu_s, dimension: n_expert x n_test_points : predictive mean of each expert at each test point
#             -- var_s, dimension: n_expert x n_test_points : predictive variance of each expert at each test point
#             -- power, dimension : 1x1 : Softmax scaling
#             -- weighting, str : weighting method (variance/wass/uniform/diff_entr/no_weights)
#             -- prior_var, dimension: 1x1 : shared prior variance of expert GPs
#             -- soft_max_wass : logical : whether to use softmax scaling or fraction scaling
#
#     Output :
#             -- weight_matrix, dimension: n_expert x n_test_points : unnormalized weight of ith expert at jth test point
#     """
#
#     if weighting == 'variance':
#         weight_matrix = (-power * logvars).exp()
#
#     if weighting == 'wass':
#         wass = mus.pow(2) + (logvars - prior_var).pow(2)
#
#         if softmax:
#             weight_matrix = (power * wass).exp()
#         else:
#             weight_matrix = wass.pow(power)
#
#     if weighting == 'uniform':
#         weight_matrix = torch.ones(mus.shape, dtype=torch.float32) / mus.shape[0]
#
#     if weighting == 'diff_entr':
#         weight_matrix = 0.5 * (torch.log(prior_var) - torch.log(logvars))
#
#     if weighting == 'no_weights':
#         weight_matrix = 1
#
#     return weight_matrix.float()
#
#
# def normalize_weights(weight_matrix):
#     """ Compute unnormalized weight matrix
#     Inputs :
#             -- weight_matrix, dimension: n_expert x n_test_points : unnormalized weight of ith expert at jth test point
#
#
#     Output :
#             -- weight_matrix, dimension: n_expert x n_test_points : normalized weight of ith expert at jth test point
#     """
#
#     sum_weights = torch.sum(weight_matrix, dim=0)
#     weight_matrix = weight_matrix / sum_weights
#
#     return weight_matrix
#
#         # For all DgPs, normalized weights of experts requiring normalized weights and compute the aggegated local precisions
#         if method == 'PoE':
#             prec = tf.reduce_sum(prec_s, axis=0)
#
#         if method == 'gPoE':
#             weight_matrix = normalize_weights(weight_matrix)
#
#             prec = tf.reduce_sum(weight_matrix * prec_s, axis=0)
# var = 1 / prec
#
# mu = var * tf.reduce_sum(weight_matrix * prec_s * mu_s, axis=0)
#
# mu = tf.reshape(mu, (-1, 1))
# var = tf.reshape(var, (-1, 1))