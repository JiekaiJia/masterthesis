import numpy as np

from scheduler import Scheduler, Server


class Scenario:
    
    def __init__(self, n_schedulers=3, n_servers=3, sending_rate=10, serving_rate=10, queue_len=10):
        self.schedulers = [Scheduler(i, sending_rate) for i in range(n_schedulers)]
        for i, scheduler in enumerate(self.schedulers):
            scheduler.name = f'scheduler_{i}'
            scheduler.silent = True
        self.servers = [Server(i, serving_rate, queue_len) for i in range(n_servers)]
        for i, server in enumerate(self.servers):
            server.name = f'server_{i}'

    def reset_world(self, world, np_random):
        pass

    def benchmark_data(self, scheduler):
        pass

    def is_collision(self, scheduler1, scheduler2):
        pass
    
    def reward(self, scheduler):
        rew = 0
        if scheduler.collide:
            for a in self.schedulers:
                if self.is_collision(a, scheduler):
                    rew -= 1
        return rew

    def global_reward(self):
        rew = 0
        for l in self.servers:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.schedulers]
            rew -= min(dists)
        return rew

    def observation(self, scheduler, world):
        # get positions of all entities in this scheduler's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - scheduler.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other schedulers
        comm = []
        other_pos = []
        for other in world.schedulers:
            if other is scheduler:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - scheduler.state.p_pos)
        return np.concatenate([scheduler.state.p_vel] + [scheduler.state.p_pos] + entity_pos + other_pos + comm)