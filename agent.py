"""This module provide the basic agent classes, which consist environment."""
import numpy as np


class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.name = ''
        self.queue = []

    def receive(self, package):
        pass

    def send(self):
        pass

    def serve(self):
        pass


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

    def has_package(self):
        if self.queue:
            return True
        return False

    def send(self):
        if self.action.a == 0:
            return None
        else:
            packet = self.queue.pop(0)
            packet.target = self.action.a
            return packet


class Server(Agent):
    def __init__(self, agent_id, serving_rate, queue_max_len):
        super().__init__(agent_id)
        self.serving_rate = serving_rate
        self.queue_max_len = queue_max_len
        self.acc_serving_t = 0
        self.busy_time = 0

    def receive(self, package):
        self.queue.append(package)

    def __len__(self):
        return len(self.queue)

    def __bool__(self):
        # Decide if server queue is full.
        if len(self.queue) == self.queue_max_len:
            return True
        return False

    def serve(self):
        if not self.queue:
            self.busy_time = 0
            return
        for package in self.queue:
            # Set a serving time for each package in queue.
            if not package.serving_time:
                package.serving_time = self._serve_time()

        if self.busy_time == 0:
            self.acc_serving_t = self.queue[0].serving_time
        while self.acc_serving_t <= self.busy_time:
            self.queue.pop(0)
            if self.queue:
                self.acc_serving_t += self.queue[0].serving_time
            else:
                return
        # Store how long a server works.
        self.busy_time += 1

    def _serve_time(self):
        t = np.random.exponential(1 / self.serving_rate)
        return t


class Action:
    def __init__(self):
        self.a = 0
        self.c = 0

