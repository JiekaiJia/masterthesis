import random

import numpy as np

from agent import Scheduler, Server
from packages import CreatePackage


class BasicScenario:

    def __init__(self, conf):
        self.conf = conf

        # Initialize schedulers.
        self.schedulers = [Scheduler(i + 1) for i in range(conf.n_schedulers)]
        for i, scheduler in enumerate(self.schedulers):
            scheduler.name = f'scheduler_{i + 1}'

        # Initialize servers.
        self.servers = [Server(i + 1, conf.serving_rate, conf.queue_max_len) for i in range(conf.n_servers)]
        for i, server in enumerate(self.servers):
            server.name = f'server_{i + 1}'
        # Initialize packages and drop_pkgs dict.
        self.reset()

    def reset(self):
        for server in self.servers:
            server.reset()
        self.packages = [CreatePackage(self.conf.n_packages, self.conf.arrival_rate, scheduler.name) for scheduler
                         in self.schedulers]
        self.drop_pkgs = {scheduler.name: 0 for scheduler in self.schedulers}

    def benchmark_data(self, scheduler):
        raise NotImplementedError()

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
    partial servers. The transmission delay will change every 10 seconds and the partial access is fixed initially."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.delay_t = np.random.randint(1, high=4, size=self.conf.n_schedulers)
        self.obs_servers = cfg.obs_servers
        # Action dimension is the number of observed queues. The scheduler must send packages when receiving packages.
        self.dim_a = self.obs_servers
        # The servers that each scheduler can observe and access.
        idxs = [i for i in range(len(self.servers))]
        random.shuffle(idxs)
        for i, scheduler in enumerate(self.schedulers):
            for j in range(self.obs_servers):
                scheduler.obs_servers.append(self.servers[idxs[(i+j) % len(idxs)]])
                self.servers[idxs[(i + j) % len(idxs)]].access_schedulers.append(scheduler)

        # Setting a scheduler for each server, which can get real-time observations.
        self.real_obs_s = {}
        tmp = {}
        for server in self.servers:
            self.real_obs_s[server] = random.choice(server.access_schedulers)
            tmp[server.name] = self.real_obs_s[server].name

    def reset_delay_t(self):
        self.delay_t = np.random.randint(1, high=4, size=self.conf.n_schedulers)

    def observation(self, scheduler):
        # Observed delayed queue length
        queue_lens = []
        # Real-time queue length
        real_queue_lens = []
        delay_t = self.delay_t[scheduler.id - 1]
        for server in scheduler.obs_servers:
            if self.real_obs_s[server] == scheduler:
                queue_lens.append(server.history_len[-1])
            else:
                # If time step is smaller than delay time, then observe 0
                if len(server.history_len) <= delay_t:
                    queue_lens.append(0)
                # If time step is larger than delay time, then observe the queue length that is n time step ago.
                else:
                    queue_lens.append(server.history_len[-1-delay_t])
            real_queue_lens.append(server.history_len[-1])
        return queue_lens, real_queue_lens


class PartialcommScenario(PartialAccessScenario):
    """This scenario inherits from class PartialAccessScenario. It defines the communications between agents."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.comm_group = {}
        self.set_comm_group()

    def set_comm_group(self):
        # Each scheduler has a communication group, in which the agents share the same partial access.
        # comm_group is a dict, whose key is object scheduler, value is a list of lists, lists include the schedulers
        # that have the same access queues.
        for scheduler in self.schedulers:
            self.comm_group[scheduler] = []
            for k, server in enumerate(scheduler.obs_servers):
                tmp = []
                for other in server.access_schedulers:
                    if other is not scheduler:
                        tmp.append(other)
                self.comm_group[scheduler].append(tmp)
