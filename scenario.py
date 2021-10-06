import math
import random

import numpy as np

from agent import Scheduler, Server
from packages import CreatePackage


class BasicScenario:

    def __init__(self, conf):
        self.conf = conf

        # Initialize schedulers.
        self.schedulers = [Scheduler(i + 1) for i in range(conf['n_schedulers'])]
        for i, scheduler in enumerate(self.schedulers):
            scheduler.name = f'scheduler_{i + 1}'
            scheduler.silent = conf['silent']
            scheduler.c_noise = conf['c_noise']

        # Initialize servers.
        self.servers = [Server(i + 1, conf['serving_rate'], conf['queue_max_len']) for i in range(conf['n_servers'])]
        for i, server in enumerate(self.servers):
            server.name = f'server_{i + 1}'
        # Initialize packages and drop_pkgs dict.
        self.reset()

    def reset(self):
        self.packages = [CreatePackage(self.conf['n_packages'], self.conf['arrival_rate'], scheduler.name) for scheduler
                         in
                         self.schedulers]
        self.drop_pkgs = {scheduler.name: 0 for scheduler in self.schedulers}

    def benchmark_data(self, scheduler):
        raise NotImplementedError()

    def reward(self, scheduler):
        # If two schedulers have the same target queue, then the package will drop and get the reward -1.
        # If an agent sends a message, then it will get a cost reward.
        rew = 0
        for other in self.schedulers:
            if other is not scheduler:
                if self.send_msg(scheduler):
                    rew -= math.log(len(self.servers), 10)
            rew -= self.drop_pkgs[other.name]
        return rew

    def observation(self, scheduler):
        raise NotImplementedError()

    def env_state(self):
        return [len(server) for server in self.servers]

    def send_msg(self, scheduler):
        if any(scheduler.msg):
            return True
        return False


class SimpleScenario(BasicScenario):
    """This scenario has no delay time."""
    
    def __init__(self, conf):
        super().__init__(conf)
        self.dim_a = conf['n_servers'] + 1
        # Define communication action dimension.
        self.dim_c = conf['msg_bits']
        for i, scheduler in enumerate(self.schedulers):
            scheduler.msg = [0]*self.dim_c

    def observation(self, scheduler):
        eps = 0.5
        # Get queue length of each server.
        queue_lens = []
        for server in self.servers:
            if np.random.randn() < eps:
                queue_lens.append(len(server))
            else:
                queue_lens.append(0)

        scheduler.obs = queue_lens

        if scheduler.silent:
            return queue_lens
        # communication of all other schedulers
        comm = []
        for other in self.schedulers:
            if other is scheduler:
                continue
            comm.append(other.msg)
        return np.concatenate([queue_lens] + comm)


class PartialAccessScenario(BasicScenario):
    """This scenario has a delay time for information transmission and each scheduler can only observe and access
    partial servers. The transmission delay will change every 10 seconds and the partial access is fixed initially."""

    def __init__(self, conf):
        super().__init__(conf)
        self.delay_t = conf['delay_time']
        self.obs_servers = conf['obs_servers']
        self.frequency = conf['delay_change_frequency']
        self.dim_a = self.obs_servers + 1
        # The servers that each scheduler can observe and access.
        idxs = [i for i in range(len(self.servers))]
        random.shuffle(idxs)
        for i, scheduler in enumerate(self.schedulers):
            for j in range(self.obs_servers):
                scheduler.obs_servers.append(self.servers[idxs[(i+j) % len(idxs)]])
                self.servers[idxs[(i + j) % len(idxs)]].access_schedulers.append(scheduler)

    def set_delay_t(self, value):
        self.delay_t = value

    def observation(self, scheduler):
        # 95% confidence interval.
        eps = 1.65
        # Get queue length of each server.
        queue_lens = []
        for server in scheduler.obs_servers:
            # If time step is smaller than delay time, then observe 0
            if len(server.history_len) <= self.delay_t:
                queue_lens.append(0)
            # If time step is larger than delay time, then observe the queue length that is 2 time step ago.
            else:
                # if np.random.randn() < eps:
                #     queue_lens.append(server.history_len[-1-self.delay_t])
                # else:
                #     queue_lens.append(0)
                queue_lens.append(server.history_len[-1-self.delay_t])
        return queue_lens


class PartialcommScenario(PartialAccessScenario):
    """This scenario inherits from class PartialAccessScenario. It defines the communications between agents."""

    def __init__(self, conf):
        super().__init__(conf)
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
