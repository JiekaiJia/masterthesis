import numpy as np

from agent import Scheduler, Server
from packages import CreatePackage


class Scenario:
    
    def __init__(self, conf):

        self.dim_a = conf['n_servers'] + 1
        self.dim_c = conf['msg_bits']

        self.packages = [CreatePackage(conf['n_packages'], conf['arrival_rate']) for _ in range(conf['n_schedulers'])]

        self.schedulers = [Scheduler(i+1, self.dim_c) for i in range(conf['n_schedulers'])]
        for i, scheduler in enumerate(self.schedulers):
            scheduler.name = f'scheduler_{i+1}'
            scheduler.silent = conf['silent']
            scheduler.c_noise = conf['c_noise']

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
                    rew -= self.dim_c
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

        # queue_lens.append(len(scheduler))

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
        """"""
        if any(scheduler.msg):
            return True
        return False
