"""https://github.com/mhw32/multimodal-vae-public"""

import torch
import torch.nn as nn


class GRUtrigger(nn.Module):

    def __init__(self, obs_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        """
        @param obs_size: observation length
        @param embedding_dim: length of embedding vector
        @param hidden_dim: Number of GRU's neuron
        @param layer_dim: Number of GRU's layer
        @param output_dim: length of output
        """
        super().__init__()
        # embedding
        self.embedding = nn.Embedding(obs_size, embedding_dim)
        # GRU
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x : [bacth, time_step, vocab_size]
        embeds = self.embedding(x)
        # embeds : [batch, time_step, embedding_dim]
        r_out, h_n = self.gru(embeds, None)
        # r_out : [batch, time_step, hidden_dim]
        out = self.fc1(r_out[:, -1, :])
        # out : [batch, time_step, output_dim]
        return out


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents, n_modalities, n_embeddings, obs_len, schedulers, training, batch_size, comm_group):
        super().__init__()
        self.n_modalities = n_modalities
        self.schedulers = schedulers
        self.training = training
        self.bs = batch_size
        self.comm_group = comm_group
        # Each agent has encoders ang decoders for each observing queue.
        self.belief_encoders = nn.ModuleList(
            [nn.ModuleList([AgentEncoder(n_latents, n_embeddings) for _ in range(obs_len)]) for _ in range(n_modalities)]
        )
        self.belief_decoders = nn.ModuleList(
            [nn.ModuleList([AgentDecoder(n_latents) for _ in range(obs_len)]) for _ in range(n_modalities)]
        )
        self.experts = ProductOfExperts()
        self.n_latents = n_latents

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            # return torch.clamp(eps.mul(std).add_(mu), 0)
            # return F.softmax(eps.mul(std).add_(mu), dim=1)
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            # return torch.clamp(mu, 0)
            # return F.softmax(mu, dim=1)
            return mu

    def forward(self, p, obs):
        """Forward pass through the MVAE.
        @param obs: list of ?PyTorch.Tensors: NxL,
                      where N is the number of schedulers and L is the length of observations.
                      If a single belief is missing, pass None
                      instead of a Tensor. Regardless if all beliefs
                      are missing, still pass a list of <n_modalities> None's.
        @param p: dict
                      Probability, which decide whether to get the messages.
        @return obs_recons: list of PyTorch.Tensors (n_modalities length)
        """
        mu, logvar = self.infer(p, obs)
        obs_recons = []
        # reparametrization trick to sample
        for i, (_mu, _logvar) in enumerate(zip(mu, logvar)):
            if _mu is None:
                obs_recons.append(None)
                continue
            z = [self.reparametrize(__mu, __logvar) for __mu, __logvar in zip(_mu, _logvar)]
            obs_recon = []
            for j, _z in enumerate(z):
                obs_recon.append(self.belief_decoders[i][j](_z))
            obs_recons.append(obs_recon)
        return obs_recons, mu, logvar

    def infer(self, p, obs):
        # get the batch size
        batch_size = self.bs

        use_cuda = next(self.parameters()).is_cuda  # check if CUDA

        # Compute the distributions of all the schedulers.
        mus = []
        logvars = []
        for i, encoders in enumerate(self.belief_encoders):
            if obs[i] is None:
                mus.append(None)
                logvars.append(None)
                continue
            _mus = []
            _logvars = []
            _input = obs[i].long()
            for j, encoder in enumerate(encoders):
                belief_mu, belief_logvar = encoder(_input[j].unsqueeze(0))
                _mus.append(belief_mu)
                _logvars.append(belief_logvar)
            mus.append(_mus)
            logvars.append(_logvars)
        _index_map = {scheduler: i for i, scheduler in enumerate(self.schedulers)}

        f_mu, f_logvar = [], []
        # Each scheduler has a communication group, in which the agents share the same partial access.
        for scheduler in self.schedulers:
            # If the scheduler is done, then skip this scheduler.
            if mus[_index_map[scheduler]] is None:
                f_mu.append(None)
                f_logvar.append(None)
                continue
            comm_group = self.comm_group[scheduler]
            _f_mu, _f_logvar = [], []
            # Each queue has a group of schedulers, which can access to it.
            for k, group in enumerate(comm_group):
                mu, logvar = prior_expert((1, batch_size, self.n_latents), use_cuda=use_cuda)
                mu = torch.cat((mu, mus[_index_map[scheduler]][k].unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, logvars[_index_map[scheduler]][k].unsqueeze(0)), dim=0)
                if not scheduler.silent and p[scheduler.name][k] > 0.5:
                    server = scheduler.obs_servers[k]
                    for member in group:
                        if mus[_index_map[member]] is None or p[member.name][k] > 0.5:
                            continue
                        idx = member.obs_servers.index(server)
                        mu = torch.cat((mu, mus[_index_map[member]][idx].unsqueeze(0)), dim=0)
                        logvar = torch.cat((logvar, logvars[_index_map[member]][idx].unsqueeze(0)), dim=0)
                # product of experts to combine gaussians
                mu, logvar = self.experts(mu, logvar)
                _f_mu.append(mu)
                _f_logvar.append(logvar)
            f_mu.append(_f_mu)
            f_logvar.append(_f_logvar)
        return f_mu, f_logvar

    def show_grad(self):
        print('encoder')
        for name, weight in self.belief_encoders[0][0].named_parameters():
            if weight.requires_grad:
                print(name, '_weight_grad', weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print('decoder')
        for name, weight in self.belief_decoders[0][0].named_parameters():
            if weight.requires_grad:
                print(name, '_weight_grad', weight.grad.mean(), weight.grad.min(), weight.grad.max())


class AgentEncoder(nn.Module):
    """Parametrizes q(z|y).
    We use a single inference network that encodes
    a single attribute.
    @param n_latents: integer
                      number of latent variable dimensions.
    """

    def __init__(self, n_latents, n_embeddings):
        super(AgentEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Embedding(n_embeddings, 128),
            Swish(),
            nn.Linear(128, 128),
            Swish(),
            nn.Linear(128, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x.long())
        return x[:, :n_latents], x[:, n_latents:]


class AgentDecoder(nn.Module):
    """Parametrizes p(y|z).
    We use a single generative network that decodes
    a single attribute.
    @param n_latents: integer
                      number of latent variable dimensions.
    """

    def __init__(self, n_latents):
        super(AgentDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents, 128),
            Swish(),
            nn.Linear(128, 128),
            Swish(),
            nn.Linear(128, 128),
            Swish(),
            nn.Linear(128, n_latents))

    def forward(self, z):
        z = self.net(z)
        return z  # NOTE: no sigmoid here. See train.py


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


if __name__ == '__main__':
    from torchsummary import summary
    model = GRUtrigger(obs_size=5, embedding_dim=128, hidden_dim=128, layer_dim=2, output_dim=5)
    summary(model, input_size=(5,), batch_size=-1)
