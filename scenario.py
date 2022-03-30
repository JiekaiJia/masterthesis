import random

import numpy as np

from agent import Scheduler, Server
from packages import CreatePackage


class BasicScenario:

    def __init__(self, cfg):
        random.seed(cfg.random_seed)
        self.cfg = cfg
        self.num_servers = cfg.n_servers
        self.num_schedulers = cfg.n_schedulers

        # Initialize schedulers.
        self.schedulers = [Scheduler(i + 1, cfg.act_syncT) for i in range(self.num_schedulers)]
        for i, scheduler in enumerate(self.schedulers):
            scheduler.name = f'scheduler_{i + 1}'

        # Initialize servers.
        self.servers = [Server(i + 1, cfg.act_syncT, cfg.serving_rate, cfg.max_q_len) for i in range(self.num_servers)]
        for i, server in enumerate(self.servers):
            server.name = f'server_{i + 1}'
        # Initialize packages and drop_pkgs dict.
        self.reset()

    def reset(self):
        # Clear servers' history length.
        for server in self.servers:
            server.reset()
        # Reset packages that should arrive at schedulers.
        self.packages = [CreatePackage(self.cfg.n_packages, self.cfg.arrival_rate, scheduler.name) for scheduler
                         in self.schedulers]
        self.drop_pkgs = {scheduler.name: 0 for scheduler in self.schedulers}

    def reward(self, scheduler):
        # If two schedulers have the same target queue, then the package will drop and get the reward -1.
        return -self.drop_pkgs[scheduler.name]

    def observation(self, scheduler):
        raise NotImplementedError()

    def schedulers_state(self):
        return {scheduler.name: len(scheduler) for scheduler in self.schedulers}

    def env_state(self):
        return {server.name: len(server) for server in self.servers}


class PartialAccessScenario(BasicScenario):
    """This scenario has a delay time for information transmission and each scheduler can only observe and access
    partial servers. The transmission delay will change every n seconds and the partial access is fixed initially."""

    def __init__(self, cfg):
        super().__init__(cfg)
        # Action dimension is the number of observed queues. The scheduler must send packages when receiving packages.
        self.cfg = cfg
        self.dim_a = cfg.num_obs_servers
        self.max_q = cfg.max_q_len
        self.delay_t = np.random.randint(1, high=4, size=self.num_schedulers)
        self.obs_server_id = {}
        # The servers that each scheduler can observe and access.
        idxs = [i for i in range(self.num_servers)]
        random.shuffle(idxs)
        for i, scheduler in enumerate(self.schedulers):
            self.obs_server_id[scheduler.name] = []
            obs_id_app = self.obs_server_id[scheduler.name].append
            scheduler_obs_app = scheduler.obs_servers.append
            for j in range(cfg.num_obs_servers):
                chosen_server = self.servers[idxs[(i+j) % len(idxs)]]
                obs_id_app(chosen_server.id)
                scheduler_obs_app(chosen_server)
                chosen_server.access_schedulers.append(scheduler)

        # Setting at least one scheduler for each server, which can get real-time observations.
        self.real_obs_s = {}
        for server in self.servers:
            ids = [i for i in range(len(server.access_schedulers))]
            nums = random.randint(1, len(ids)//2)
            random.shuffle(ids)
            self.real_obs_s[server] = []
            for i in range(nums):
                self.real_obs_s[server].append(server.access_schedulers[ids[i]])

    def reset_delay_t(self):
        self.delay_t = np.random.randint(1, high=4, size=self.num_schedulers)

    def observation(self, scheduler):
        # Observed delayed queue length
        queue_lens = []
        # Real-time queue length
        real_queue_lens = []
        delay_t = self.delay_t[scheduler.id - 1]
        for server in scheduler.obs_servers:
            if scheduler in self.real_obs_s[server]:
                queue_lens.append(server.history_len[-1])
            else:
                # If time step is smaller than delay time, then observe 0
                if len(server.history_len) <= delay_t:
                    queue_lens.append(0)
                # If time step is larger than delay time, then observe the queue length that is n time step ago.
                else:
                    if self.cfg.opposite:
                        queue_lens.append(self.max_q - server.history_len[-1])
                    else:
                        queue_lens.append(server.history_len[-1-delay_t])
            real_queue_lens.append(server.history_len[-1])
        return queue_lens, real_queue_lens
