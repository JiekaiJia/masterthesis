import math

import numpy as np

from agent import Scheduler, Server
from packages import CreatePackage


class Scenario:
    
    def __init__(self, conf):
        # Define action dimension and communication action dimension.
        self.dim_a = conf['n_servers'] + 1
        self.dim_c = conf['msg_bits']

        # Initialize schedulers.
        self.schedulers = [Scheduler(i+1, self.dim_c) for i in range(conf['n_schedulers'])]
        for i, scheduler in enumerate(self.schedulers):
            scheduler.name = f'scheduler_{i+1}'
            scheduler.silent = conf['silent']
            scheduler.c_noise = conf['c_noise']
        # Initialize packages.
        self.packages = [CreatePackage(conf['n_packages'], conf['arrival_rate'], scheduler.name) for scheduler in self.schedulers]
        # Initialize servers.
        self.servers = [Server(i+1, conf['serving_rate'], conf['queue_max_len']) for i in range(conf['n_servers'])]
        for i, server in enumerate(self.servers):
            server.name = f'server_{i+1}'

        self.drop_pkgs = {scheduler.name: 0 for scheduler in self.schedulers}

    def benchmark_data(self, scheduler):
        raise NotImplementedError()
    
    def reward(self, scheduler):
        # If two schedulers have the same target queue, then the package will drop and get the reward -1.
        # If an agent sends a messge, then it will get a cost reward.
        rew = 0
        for other in self.schedulers:
            if other is not scheduler:
                if self.send_msg(scheduler):
                    rew -= math.log(self.dim_c, 10)
        rew -= self.drop_pkgs[scheduler.name]
        return rew

    def observation(self, scheduler):
        eps = 0.5
        # Get queue length of each server.
        queue_lens = []
        for server in self.servers:
            if np.random.randn() < eps:
                queue_lens.append(len(server))
            else:
                queue_lens.append(0)

        if scheduler.silent:
            return queue_lens
        # communication of all other schedulers
        comm = []
        for other in self.schedulers:
            if other is scheduler:
                continue
            comm.append(other.msg)
        return np.concatenate([queue_lens] + comm)

    def env_state(self):
        return [len(server) for server in self.servers]

    def send_msg(self, scheduler):
        if any(scheduler.msg):
            return True
        return False
