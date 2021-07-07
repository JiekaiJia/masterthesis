"""This module provide the basic agent classes, which consist environment."""


class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.name = ''
        self.queue = []

    def receive(self, job):
        pass

    def send(self, agent_to_send):
        pass

    def serve(self):
        pass


class Scheduler(Agent):
    def __init__(self, agent_id, sending_rate):
        super().__init__(agent_id)
        self.silent = True
        self.sending_rate = sending_rate
        self.action = 0

    def receive(self, job):
        self.queue.append(job)

    def send(self, agent_to_send):
        pass


class Server(Agent):
    def __init__(self, agent_id, serving_rate, queue_len):
        super().__init__(agent_id)
        self.serving_rate = serving_rate
        self.queue_len = queue_len

    def receive(self, job):
        self.queue.append(job)

    def serve(self):
        pass

    def __len__(self):
        return len(self.queue)

