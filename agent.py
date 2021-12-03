"""This module provides the basic agent classes, which consist environment."""
import numpy as np

from utils import index


class Agent:
    def __init__(self, agent_id):
        # Starting from 1,2,3,4,...
        self.id = agent_id
        self.name = ''
        self.queue = []

    def reset(self):
        self.queue.clear()

    def receive(self, package):
        raise NotImplementedError()

    def send(self):
        raise NotImplementedError()

    def serve(self, pkg_arrival_t):
        raise NotImplementedError()


class Scheduler(Agent):
    """Scheduler actions:
                          1 --> send to server 1
                          2 --> send to server 2
    """
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.action = Action()
        self.pkg_count = 0
        # A list of server objects.
        self.obs_servers = []

    def __len__(self):
        return len(self.queue)

    def __bool__(self):
        if self.queue:
            return True
        return False

    def reset(self):
        super().reset()

    def receive(self, package):
        self.queue.append(package)

    def send(self):
        nums = len(self.queue)
        p = np.random.rand(nums)
        th_p = np.cumsum(self.action.a)
        packets = []
        for i, _p in enumerate(p):
            # According to threshold value, decide which server to send.
            act_index = index(th_p, _p)
            packets.append(self.queue.pop(0))
            packets[-1].target = act_index + 1 if not self.obs_servers else self.obs_servers[act_index].id
            packets[-1].sending_time = packets[-1].arriving_time + packets[-1].halt_time
        assert len(self.queue) == 0, "The scheduler queue must be empty after dispatching packages !!"
        # print([j.id for j in self.obs_servers])
        # print(self.action.a)
        # print(self.name + f' sends {len(packets)} packages to server {[a.target for a in packets]}.')
        return packets

    def update_halt_t(self):
        # Update package halt time, if packages are still in queue.
        for pkg in self.queue:
            pkg.halt_time += 1


class Server(Agent):
    def __init__(self, agent_id, serving_rate, queue_max_len):
        super().__init__(agent_id)
        self.serving_rate = serving_rate
        self.queue_max_len = queue_max_len
        self.leaving_pkgs = []
        self.access_schedulers = []
        self.history_len = [0]

    def __len__(self):
        return len(self.queue)

    def __bool__(self):
        # Decide if server queue is full.
        if len(self.queue) == self.queue_max_len:
            return True
        return False

    def reset(self):
        super().reset()
        self.history_len = [0]

    def receive(self, package):
        # The time package arrives at server
        pkg_arrival_t = package.sending_time
        self.serve(pkg_arrival_t)
        if len(self.queue) < self.queue_max_len:
            # If server's queue is not full, then receives a package.
            package.serving_time = self._serve_time()
            if not self.queue:
                package.departure_time = package.sending_time + package.serving_time
            else:
                package.departure_time = self.queue[-1].departure_time + package.serving_time

            self.queue.append(package)
        else:
            # If server's queue is full, then drop.
            return package
        return None

    def serve(self, pkg_arrival_t):
        if not self.queue:
            return  # print(f'{self.name} is empty.')

        n = len(self.queue)
        self.leaving_pkgs.clear()
        for i in range(n):
            # The former package leaves before current package arrives.
            # print(self.queue[0].departure_time, pkg_arrival_t)
            if self.queue and self.queue[0].departure_time <= pkg_arrival_t:
                self.leaving_pkgs.append(self.queue.pop(0))
            else:
                # print(f'{self.name} serves {len(self.leaving_pkgs)} packages and the queue length is {n} before serving.')
                return

    def _serve_time(self):
        t = np.random.exponential(1 / self.serving_rate)
        return t


class Action:
    def __init__(self):
        self.a = []
        self.c = 0
