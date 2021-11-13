"""https://github.com/minqi/learning-to-communicate-pytorch
This file is used to train belief models."""
import gym
import numpy as np
from ray.tune.registry import register_env
from scipy.special import softmax
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim

from custom_env.environment import RLlibEnv
from rllib_ppo import RLlibAgent
from utils import DotDic


class Arena:
    """Arena class is used to train the MVAE model."""
    def __init__(self, opt, env, policy):
        self.opt = opt
        self.env = env
        self.policy = policy
        self.buffer = []
        self.buffer_size = opt.buffer_size
        self.episode = DotDic({})
        self.writer = SummaryWriter()
        self.n_models = opt.n_schedulers * opt.obs_servers
        self.model = self.env.model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(env.model.parameters(), lr=opt.lr)

    def create_episode(self):
        opt = self.opt
        self.episode = DotDic({})
        self.episode.steps = torch.zeros(opt.bs).int()
        self.episode.ended = torch.zeros(opt.bs).int()
        self.episode.step_records = []

    def create_step_record(self):
        opt = self.opt
        n_servers = opt.obs_servers
        n_schdulers = opt.n_schedulers
        record = DotDic({})
        record.mu = [[torch.zeros(opt.bs, opt.queue_max_len+1) for _ in range(n_servers)] for _ in range(n_schdulers)]
        record.logvar = [[torch.zeros(opt.bs, opt.queue_max_len+1) for _ in range(n_servers)]
                         for _ in range(n_schdulers)]
        record.obs = [torch.zeros(opt.bs, n_servers) for _ in range(n_schdulers)]
        record.recon_obs = [torch.zeros(opt.bs, n_servers, opt.queue_max_len+1) for _ in range(n_schdulers)]
        
        record.mu0 = [[torch.zeros(opt.bs, opt.queue_max_len+1) for _ in range(n_servers)] for _ in range(n_schdulers)]
        record.logvar0 = [[torch.zeros(opt.bs, opt.queue_max_len+1) for _ in range(n_servers)]
                          for _ in range(n_schdulers)]
        record.recon_obs0 = [torch.zeros(opt.bs, n_servers, opt.queue_max_len+1) for _ in range(n_schdulers)]

        record.mu1 = [[torch.zeros(opt.bs, opt.queue_max_len+1) for _ in range(n_servers)] for _ in range(n_schdulers)]
        record.logvar1 = [[torch.zeros(opt.bs, opt.queue_max_len + 1) for _ in range(n_servers)]
                          for _ in range(n_schdulers)]
        record.recon_obs1 = [torch.zeros(opt.bs, n_servers, opt.queue_max_len + 1) for _ in range(n_schdulers)]

        return record

    def train(self):
        opt = self.opt
        best_loss = float('inf')
        for e in range(opt.nepisodes):
            if e % 100 == 0:
                show = True
            else:
                show = False
            self.run_episode(show)

            if e < opt.annealing_episodes:
                # compute the KL annealing factor for the current episode in the current epoch
                annealing_factor = (float(e) / float(opt.annealing_episodes))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            loss = self.learn_from_episode(self.episode, annealing_factor)

            self.writer.add_scalar('loss', loss.item(), global_step=e+1)
            print('-'*80)
            print(f'Training Episode: {e+1}/{opt.nepisodes} Loss: {loss.item():.2f}')
            # Save the model
            if e >= opt.annealing_episodes and loss.item() < best_loss:
                best_loss = loss.item()
                self.save_model(e, loss, opt.check_path)
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
        opt = self.opt
        obss = env.reset()

        step = 0
        self.create_episode()
        dones = {'__all__': False}
        while not dones['__all__']:
            self.episode.step_records.append(self.create_step_record())
            try:
                belief = [softmax(torch.cat(obs, dim=0).cpu().detach().numpy(), axis=1) for obs in obss[2]]
                obss = {scheduler: obs for scheduler, obs in zip(env.schedulers, belief)}
            except KeyError:
                pass

            actions = self.policy.agent.compute_actions(obss, policy_id='shared')

            obss, r, dones, info = env.step(actions)
            if show:
                pass
                # print('timestep:', step+1)
                # print('state:', env.state())
                # print('obs:', obss[0])
                # print('re_obs', obss[1])
                # belief = {scheduler: torch.cat(obss[1][i], dim=0).cpu().numpy() for i, scheduler in enumerate(self.env.schedulers)}
                # print({k: np.concatenate((v, np.array(obss[0][i]).reshape(5, 1)), axis=1)for i, (k, v) in enumerate(belief.items())})
                # print('mu', obss[2])
                # print('logvar', obss[3])
                # print('re_obss', obss[4])
                # print('re_obs0', obss[5])
                # print('mu0', obss[6])
                # print('logvar0', obss[7])
                # print('re_obss0', obss[8])
                # print('real_obs', obss[9])
                # print('rewards:', r)
                # print('dones:', dones)
                # # print('messages:', info)
                # print('_' * 80)
            for b in range(opt.bs):
                for i in range(opt.n_schedulers):
                    self.episode.step_records[step].obs[i][b, :] = obss[0][i]
                    for j in range(opt.obs_servers):
                        self.episode.step_records[step].recon_obs[i][b, j, :] = obss[1][i][j]
                        self.episode.step_records[step].mu[i][j][b, :] = obss[2][i][j]
                        self.episode.step_records[step].logvar[i][j][b, :] = obss[3][i][j]

                        self.episode.step_records[step].recon_obs0[i][b, j, :] = obss[5][i][j]
                        self.episode.step_records[step].mu0[i][j][b, :] = obss[6][i][j]
                        self.episode.step_records[step].logvar0[i][j][b, :] = obss[7][i][j]

                        self.episode.step_records[step].recon_obs1[i][b, j, :] = obss[10][i][j]
                        self.episode.step_records[step].mu1[i][j][b, :] = obss[11][i][j]
                        self.episode.step_records[step].logvar1[i][j][b, :] = obss[12][i][j]
                self.episode.steps[b] = step

            # Update step
            step += 1

    def learn_from_episode(self, episode, annealing_factor):
        self.optimizer.zero_grad()
        loss = self.episode_loss(episode, annealing_factor)
        loss.backward()
        # self.write_grad()
        self.optimizer.step()

        return loss

    def episode_loss(self, episode, annealing_factor):
        total_loss = 0
        for step in range(episode.steps):
            total_loss += self.elbo_loss(episode.step_records[step].recon_obs,
                                         episode.step_records[step].obs,
                                         episode.step_records[step].mu,
                                         episode.step_records[step].logvar,
                                         annealing_factor=annealing_factor)
            total_loss += self.elbo_loss(episode.step_records[step].recon_obs0,
                                         episode.step_records[step].obs,
                                         episode.step_records[step].mu0,
                                         episode.step_records[step].logvar0,
                                         annealing_factor=annealing_factor)
            total_loss += self.elbo_loss(episode.step_records[step].recon_obs1,
                                         episode.step_records[step].obs,
                                         episode.step_records[step].mu1,
                                         episode.step_records[step].logvar1,
                                         annealing_factor=annealing_factor)
        return total_loss / self.n_models

    def elbo_loss(self, recon, data, mu, logvar, lambda_attrs=1.0, annealing_factor=1):
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
        # n agents
        for ix in range(n_modalities):
            # for each q state has an inference network.
            for j in range(len(mu[ix])):
                BCE += lambda_attrs * self.criterion(recon[ix][:, j, :], data[ix][:, j].long())
                # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                # https://arxiv.org/abs/1312.6114
                KLD += -0.5 * torch.sum(1 + logvar[ix][j] - mu[ix][j].pow(2) - logvar[ix][j].exp(), dim=1)
        ELBO = torch.mean(BCE + annealing_factor * KLD/self.opt.obs_servers)
        return ELBO

    def write_grad(self):
        self.model.show_grad()

    def save_model(self, episode, loss, checkpoint_path):
        try:
            torch.save({'epoch': episode + 1, 'state_dict': self.model.state_dict(), 'best_loss': loss.item(),
                        'optimizer': self.optimizer.state_dict()}, checkpoint_path +
                       f'/belief_encoder{self.opt.n_schedulers}_{episode + 1}_{loss.item():.2f}.pth')
        except FileNotFoundError:
            import os
            os.mkdir(checkpoint_path)
            torch.save({'epoch': episode + 1, 'state_dict': self.model.state_dict(), 'best_loss': loss.item(),
                        'optimizer': self.optimizer.state_dict()},
                       checkpoint_path + f'/belief_encoder{self.opt.n_schedulers}_{episode + 1}_{loss.item():.2f}.pth')


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, default=None,
                        help="gives the belief model's name [default: None]")
    parser.add_argument('-experiment_name', type=str, default=None,
                        help="gives this experiment's name [default: None]")
    parser.add_argument('--silent', action='store_true', default=False,
                        help='defines if scheduler can communicate [default: False]')
    parser.add_argument('--use_belief', action='store_true', default=False,
                        help='encodes observations to belief [default: False]')
    parser.add_argument('--test', action='store_true', default=False,
                        help='decide test model or train model [default: False]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    with open('./config/PartialAccess.json', 'r') as f:
        conf = json.loads(f.read())

    conf['use_belief'] = args.use_belief
    conf['silent'] = args.silent
    # todo: no use for test.
    conf['experiment_name'] = args.experiment_name
    conf['belief_training'] = not args.test
    if args.use_belief:
        assert args.model_name is not None, 'If use belief model, the model name must be given.'
        conf['model_name'] = args.model_name

    conf['num_workers'] = 0
    conf['num_envs_per_worker'] = 1

    # Create test environment.
    env = RLlibEnv(DotDic(conf))
    # Register env
    register_env(conf['env_name'], lambda _: RLlibEnv(DotDic(conf)))
    ppo_agent = RLlibAgent(conf, env)

    path = None
    # path = '/content/drive/MyDrive/Data Science/pythonProject/masterthesis/ray_results/PPO_noComm/PPO_rllib_network-v0_c762f_00000_0_2021-11-10_23-49-01/checkpoint_000350/checkpoint-350'
    # path = '/content/drive/MyDrive/Data Science/pythonProject/masterthesis/ray_results/PPO/PPO_rllib_network-v0_1757d_00000_0_2021-11-13_09-57-20/checkpoint_000350/checkpoint-350'
    ppo_agent.load(path)
    env = gym.make(id="main_network-v0", conf=DotDic(conf))
    arena = Arena(DotDic(conf), env, ppo_agent)
    # arena.test()
    arena.train()
    ppo_agent.shutdown()

