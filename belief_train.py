"""https://github.com/minqi/learning-to-communicate-pytorch
This file is used to train belief models."""
import gym
from ray.tune.registry import register_env
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torch.optim as optim

from custom_env.environment import RLlibEnv
from rllib_ppo import RLlibAgent
from utils import DotDic, sigmoid


class Arena:
    """Arena class is used to train the VAE model."""
    def __init__(self, cfg, env, policy):
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.episode = DotDic({})
        self.writer = SummaryWriter()
        self.model = self.env.model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(env.model.parameters(), lr=cfg.lr)

    def create_episode(self):
        cfg = self.cfg
        self.episode = DotDic({})
        self.episode.steps = torch.zeros(cfg.bs).int()
        self.episode.ended = torch.zeros(cfg.bs).int()
        self.episode.step_records = []

    def create_step_record(self):
        cfg = self.cfg
        n_schdulers = cfg.n_schedulers
        n_servers = cfg.obs_servers
        record = DotDic({})
        record.mu = [torch.zeros(cfg.bs, n_servers*cfg.n_latents) for _ in range(n_schdulers)]
        record.logvar = [torch.zeros(cfg.bs, n_servers*cfg.n_latents) for _ in range(n_schdulers)]
        record.obs = [torch.zeros(cfg.bs, n_servers) for _ in range(n_schdulers)]
        record.decoding_obs = [torch.zeros(cfg.bs, cfg.obs_servers*(cfg.queue_max_len+1)) for _ in range(n_schdulers)]

        return record

    def train(self):
        cfg = self.cfg
        best_loss = float('inf')
        for e in range(cfg.nepisodes):
            if e % 100 == 0:
                show = True
            else:
                show = False
            self.run_episode(show)

            if e < cfg.annealing_episodes:
                # compute the KL annealing factor for the current episode in the current epoch
                annealing_factor = (float(e) / float(cfg.annealing_episodes))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            loss, recon_cost, kld = self.learn_from_episode(annealing_factor)

            self.writer.add_scalar('total_loss', loss.item(), global_step=e + 1)
            self.writer.add_scalar('recon_loss', recon_cost.item(), global_step=e + 1)
            self.writer.add_scalar('kld', kld.item(), global_step=e + 1)
            self.writer.add_scalar('annealing_factor', annealing_factor, global_step=e + 1)
            print('-'*80)
            print(f'Training Episode: {e+1}/{cfg.nepisodes} Loss: {loss.item():.2f}')
            # Save the model
            if e >= cfg.annealing_episodes and loss.item() < best_loss:
                best_loss = loss.item()
                self.save_model(e, loss, cfg.check_path)
                print('The model is improved ======> Model saved !!')
            # Delete the episode after training.
            self.episode.clear()
        self.writer.close()

    def test(self):
        for e in range(1):
            self.run_episode(True)
            # Delete the episode after training.
            self.episode.clear()

    def run_episode(self, show):
        env = self.env
        cfg = self.cfg
        obss = env.reset()

        step = 0
        self.create_episode()
        dones = {'__all__': False}
        while not dones['__all__']:
            self.episode.step_records.append(self.create_step_record())
            try:
                belief = [sigmoid(obs.squeeze(0).cpu().detach().numpy()) for obs in obss[3]]
                obss = {scheduler: obs for scheduler, obs in zip(env.schedulers, belief)}
            except KeyError:
                pass

            actions = self.policy.agent.compute_actions(obss, policy_id='shared')

            obss, r, dones, info = env.step(actions)
            recon_obs = [torch.argmax(F.softmax(decoding_obs.view((self.cfg.obs_servers, -1)), dim=1), dim=1)
                         for decoding_obs in obss[1]]
            if show:
                print('timestep:', step+1)
                # print('state:', env.state())
                print('obs:', obss[0])
                print('recon_obs:', recon_obs)
                print('real_obs:', obss[2])
                print('decoding_obs:', obss[1])
                print('mu:', obss[3])
                print('logvar:', obss[4])
                print('rewards:', r)
                # print('dones:', dones)
                # # print('messages:', info)
                print('_' * 80)
            for b in range(cfg.bs):
                for i in range(cfg.n_schedulers):
                    self.episode.step_records[step].obs[i][b, :] = obss[0][i]
                    self.episode.step_records[step].decoding_obs[i][b, :] = obss[1][i]
                    self.episode.step_records[step].mu[i][b, :] = obss[3][i]
                    self.episode.step_records[step].logvar[i][b, :] = obss[4][i]

                self.episode.steps[b] = step

            # Update step
            step += 1

    def learn_from_episode(self, annealing_factor):
        self.optimizer.zero_grad()
        loss, recon_cost, kld = self.episode_loss(annealing_factor)
        loss.backward()
        # self.write_grad()
        self.optimizer.step()

        return loss, recon_cost, kld

    def episode_loss(self, annealing_factor):
        total_loss, total_recon_cost, total_kld = 0, 0, 0
        episode = self.episode
        for step in range(episode.steps):
            ELBO, recon_cost, kld = self.elbo_loss(episode.step_records[step].decoding_obs,
                                                   episode.step_records[step].obs,
                                                   episode.step_records[step].mu,
                                                   episode.step_records[step].logvar,
                                                   annealing_factor=annealing_factor)
            total_loss += ELBO
            total_recon_cost += recon_cost
            total_kld += kld
        return total_loss, total_recon_cost, total_kld

    def elbo_loss(self, decoding_obs, target, mu, logvar, annealing_factor=1):
        """Compute the ELBO for an arbitrary number of data modalities.
        @param decoding_obs: list of torch.Tensors/Variables
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
        n_agents = len(decoding_obs)
        recon_cost = 0  # reconstruction cost
        kld = 0
        # n agents
        for ix in range(n_agents):
            for j in range(self.cfg.obs_servers):
                recon_cost += self.criterion(decoding_obs[ix].view((self.cfg.obs_servers, -1))[j, :].unsqueeze(0),
                                             target[ix][:, j].long())
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            kld += -0.5 * torch.sum(1 + logvar[ix] - mu[ix].pow(2) - logvar[ix].exp(), dim=1)
        ELBO = (recon_cost + annealing_factor * kld)/(n_agents*self.cfg.obs_servers)
        return ELBO, recon_cost/(n_agents*self.cfg.obs_servers), kld/(n_agents*self.cfg.obs_servers)

    def write_grad(self):
        self.model.show_grad()

    def save_model(self, episode, loss, checkpoint_path):
        try:
            torch.save({'epoch': episode + 1, 'state_dict': self.model.state_dict(), 'best_loss': loss.item(),
                        'optimizer': self.optimizer.state_dict()}, checkpoint_path +
                       f'/belief_encoder{self.cfg.n_schedulers}_{episode + 1}_{loss.item():.2f}.pth')
        except FileNotFoundError:
            import os
            os.mkdir(checkpoint_path)
            torch.save({'epoch': episode + 1, 'state_dict': self.model.state_dict(), 'best_loss': loss.item(),
                        'optimizer': self.optimizer.state_dict()},
                       checkpoint_path + f'/belief_encoder{self.cfg.n_schedulers}_{episode + 1}_{loss.item():.2f}.pth')


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_belief', action='store_true', default=False,
                        help='encodes observations to belief [default: False]')
    parser.add_argument('--test', action='store_true', default=False,
                        help='decide test model or train model [default: False]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    parser.add_argument('--restore', action='store_true', default=False,
                        help='Store the model from previous parameters [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    with open('./config/PartialAccess.json', 'r') as f:
        cfg = json.loads(f.read())

    cfg['use_belief'] = args.use_belief
    cfg['belief_training'] = not args.test

    cfg['num_workers'] = 0
    cfg['num_envs_per_worker'] = 1

    # Create test environment.
    env = RLlibEnv(DotDic(cfg))
    # Register env
    register_env(cfg['env_name'], lambda _: RLlibEnv(DotDic(cfg)))
    ppo_agent = RLlibAgent(cfg, env)

    path = None
    ppo_agent.load(path)
    env = gym.make(id="main_network-v0", cfg=DotDic(cfg))
    arena = Arena(DotDic(cfg), env, ppo_agent)

    if cfg['belief_training']:
        arena.train()
    else:
        arena.test()
    ppo_agent.shutdown()

