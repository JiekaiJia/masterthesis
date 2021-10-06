"""https://github.com/minqi/learning-to-communicate-pytorch"""

import json

from scipy.special import softmax
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torch.nn import functional as F

from dotdic import DotDic
from environment import MainEnv


class Arena:
    """Arena class is used to train the MVAE model."""
    def __init__(self, opt, env):
        self.opt = opt
        self.env = env
        self.loss = []
        self.writer = SummaryWriter()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(env.model.parameters(), lr=opt.lr)

    def create_episode(self):
        opt = self.opt
        episode = DotDic({})
        episode.steps = torch.zeros(opt.bs).int()
        episode.ended = torch.zeros(opt.bs).int()
        episode.step_records = []

        return episode

    def create_step_record(self):
        opt = self.opt
        n_servers = opt.obs_servers
        n_schdulers = opt.n_schedulers
        record = DotDic({})
        record.mu = [[torch.zeros(opt.bs, opt.queue_max_len+1) for _ in range(n_servers)] for _ in range(n_schdulers)]
        record.logvar = [[torch.zeros(opt.bs, opt.queue_max_len+1) for _ in range(n_servers)] for _ in range(n_schdulers)]
        record.obs = [torch.zeros(opt.bs, n_servers) for _ in range(n_schdulers)]
        record.recon_obs = [torch.zeros(opt.bs, n_servers, opt.queue_max_len+1) for _ in range(n_schdulers)]

        return record

    def train(self):
        opt = self.opt
        for e in range(opt.nepisodes):
            episode = self.run_episode()
            self.learn_from_episode(episode)
            self.writer.add_scalar('loss', self.loss[-1], global_step=e+1)
            print(f'Train Episode: {e+1}/{opt.nepisodes} Loss: {self.loss[-1]:.2f}')
        self.writer.close()

    def run_episode(self):
        env = self.env
        opt = self.opt
        obss = env.reset()

        step = 0
        episode = self.create_episode()
        dones = {'__all__': False}
        while not dones['__all__']:
            episode.step_records.append(self.create_step_record())
            # Random policy
            actions = {scheduler: softmax(action_space.sample()) for scheduler, action_space in env.action_spaces.items()}

            obss, r, dones, info = env.step(actions)

            print('timestep:', step+1)
            # print('state:', env.state())
            print('obs:', obss[0])
            print('re_obs', obss[1])
            print('re_obss', obss[4])
            # print('rewards:', r)
            # print('dones:', dones)
            # # print('messages:', info)
            print('_' * 80)
            for b in range(opt.bs):
                for i in range(opt.n_schedulers):
                    episode.step_records[step].obs[i][b, :] = obss[0][i]
                    for j in range(opt.obs_servers):
                        episode.step_records[step].recon_obs[i][b, j, :] = obss[1][i][j]
                        episode.step_records[step].mu[i][j][b, :] = obss[2][i][j]
                        episode.step_records[step].logvar[i][j][b, :] = obss[3][i][j]
                episode.steps[b] = step

            # Update step
            step += 1

        return episode

    def episode_loss(self, episode):
        total_loss = 0
        for step in range(episode.steps):
            total_loss += self.elbo_loss(episode.step_records[step].recon_obs,
                                         episode.step_records[step].obs,
                                         episode.step_records[step].mu,
                                         episode.step_records[step].logvar)
        return total_loss

    def elbo_loss(self, recon, data, mu, logvar, lambda_attrs=1.0, annealing_factor=1.):
        """Compute the ELBO for an arbitrary number of data modalities.
        @param recon: list of torch.Tensors/Variables
                      Contains one for each modality.
        @param data: list of torch.Tensors/Variables
                     Size much agree with recon.
        @param mu: Torch.Tensor
                   Mean of the variational distribution.
        @param logvar: Torch.Tensor
                       Log variance for variational distribution.
        @param lambda_image: float [default: 1.0]
                             weight for image BCE
        @param lambda_attr: float [default: 1.0]
                            weight for attribute BCE
        @param annealing_factor: float [default: 1]
                                 Beta - how much to weight the KL regularizer.
        """
        assert len(recon) == len(data), "must supply ground truth for every modality."
        n_modalities = len(recon)

        BCE = 0  # reconstruction cost
        KLD = 0
        for ix in range(n_modalities):
            for j in range(len(mu[ix])):
                BCE += lambda_attrs * self.criterion(recon[ix][:, j, :], data[ix][:, j].long())
                KLD += -0.5 * torch.sum(1 + logvar[ix][j] - mu[ix][j].pow(2) - logvar[ix][j].exp(), dim=1)
        ELBO = torch.mean(BCE + annealing_factor * KLD)
        return ELBO

    def learn_from_episode(self, episode):
        self.optimizer.zero_grad()
        loss = self.episode_loss(episode)
        self.write_loss(loss)
        loss.backward()
        self.optimizer.step()

    def write_loss(self, loss):
        self.loss.append(loss)


if __name__ == '__main__':
    with open('config/PartialAccess.json', 'r') as f:
        config = DotDic(json.loads(f.read()))
    env = MainEnv(config)
    arena = Arena(config, env)
    arena.train()

