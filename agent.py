"""This module provides the basic agent classes, which consist environment."""
import bisect

import math
import numpy as np


class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.c_noise = None
        self.name = ''
        self.queue = []

    def receive(self, package):
        raise NotImplementedError()

    def send(self):
        raise NotImplementedError()

    def serve(self, pkg_arrival_t):
        raise NotImplementedError()


class Scheduler(Agent):
    """Scheduler actions: 0 --> don't send message
                          1 --> send to server 1
                          2 --> send to server 2
    """
    def __init__(self, agent_id, msg_bits):
        super().__init__(agent_id)
        self.silent = True
        self.action = Action()
        self.msg = [0] * msg_bits
        self.pkg_count = 0

    def receive(self, package):
        self.queue.append(package)

    def __len__(self):
        return len(self.queue)

    def __bool__(self):
        if self.queue:
            return True
        return False

    def send(self):
        nums = len(self.queue)
        res = nums
        pkgs_to_send = [a * nums for a in self.action.a]
        packets = []
        while res > 0:
            act_index = np.argmax(pkgs_to_send)
            pkg_num = math.ceil(pkgs_to_send[act_index])
            pkgs_to_send[act_index] = 0
            res -= pkg_num
            if res < 0:
                pkg_num += res
            if act_index == 0:
                continue
            for _ in range(pkg_num):
                packets.append(self.queue.pop(0))
                packets[-1].target = act_index
                packets[-1].sending_time = packets[-1].arriving_time + packets[-1].halt_time
        # Update package halt time, if packages are still in queue.
        for pkg in self.queue:
            pkg.halt_time += 1
        # print(self.name + f' sends {len(packets)} packages to server {[a.target for a in packets]}.')
        return packets


class Server(Agent):
    def __init__(self, agent_id, serving_rate, queue_max_len):
        super().__init__(agent_id)
        self.serving_rate = serving_rate
        self.queue_max_len = queue_max_len
        self.leaving_pkgs = []

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

    def __len__(self):
        return len(self.queue)

    def __bool__(self):
        # Decide if server queue is full.
        if len(self.queue) == self.queue_max_len:
            return True
        return False

    def serve(self, pkg_arrival_t):
        n = len(self.queue)
        self.leaving_pkgs.clear()
        if not self.queue:
            return

        for i in range(n):
            # The former package leaves before current package arrives.
            if self.queue[0].departure_time <= pkg_arrival_t:
                self.leaving_pkgs.append(self.queue.pop(0))
                if not self.queue:
                    return
            else:
                return

    def _serve_time(self):
        t = np.random.exponential(1 / self.serving_rate)
        return t


class Action:
    def __init__(self):
        self.a = []
        self.c = 0

