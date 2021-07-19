import numpy as np

from agent import Scheduler, Server
from packages import CreatePackage


class Scenario:
    
    def __init__(self, n_schedulers=3, n_servers=3, arrival_rate=9, serving_rate=10, queue_max_len=10, n_packages=100):

        self.dim_a = n_servers + 1
        self.dim_c = 2

        self.packages = [CreatePackage(n_packages, arrival_rate) for _ in range(n_schedulers)]

        self.schedulers = [Scheduler(i+1, self.dim_c) for i in range(n_schedulers)]
        for i, scheduler in enumerate(self.schedulers):
            scheduler.name = f'scheduler_{i+1}'
            scheduler.silent = True

        self.servers = [Server(i+1, serving_rate, queue_max_len) for i in range(n_servers)]
        for i, server in enumerate(self.servers):
            server.name = f'server_{i+1}'

        self.drop_pkgs = {scheduler.name: 0 for scheduler in self.schedulers}

    def benchmark_data(self, scheduler):
        pass
    
    def reward(self, scheduler):
        # If two schedulers have the same target queue, then the package will drop and get the reward -1
        # cost message
        rew = 0
        for other in self.schedulers:
            if other is not scheduler:
                if self.is_collision(other, scheduler):
                    rew -= 1
                if self.send_msg(scheduler):
                    rew -= self.dim_c
        rew -= self.drop_pkgs[scheduler.name]
        return rew

    def global_reward(self):
        # The average queue length of all the servers.
        rew = 0
        for server in self.servers:
            rew -= len(server)
        rew = rew/len(self.servers)
        return rew

    def observation(self, scheduler):
        eps = 2
        # Get queue length of each server.
        queue_lens = []
        for server in self.servers:
            if np.random.randn() < eps:
                queue_lens.append(len(server))
            else:
                queue_lens.append(0)
        # communication of all other schedulers
        comm = []
        for other in self.schedulers:
            if other is scheduler:
                continue
            comm.append(other.msg)
        return np.concatenate([queue_lens] + comm)

    def is_collision(self, scheduler1, scheduler2):
        if scheduler1.action == scheduler2.action:
            return True
        return False

    def send_msg(self, scheduler):
        if scheduler.msg != [0] * self.dim_c:
            return True
        return False
