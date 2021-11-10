import numpy as np


class BasicPolicy:
    def __init__(self, conf):
        self.conf = conf

    def get_actions(self, obs):
        raise NotImplementedError()


class RandomPolicy(BasicPolicy):
    def __init__(self, conf):
        super().__init__(conf)

    def get_actions(self, obs):
        # Dispatch packages to the queue with gaussian distributions.
        act = np.random.randn(len(obs)+1)
        return act


class ShortestQPolicy(BasicPolicy):
    def __init__(self, conf):
        super().__init__(conf)

    def get_actions(self, obs):
        # Dispatch packages to the queue with shortest queue length.
        act = [-float('inf')]*(len(obs)+1)
        idx = np.argmin(obs)
        act[idx] = 0
        return act

# class PPOPolicy(BasicPolicy):
#     def __init__(self, conf):
#         super().__init__(conf)
#
#     def get_actions(self, obs):
#         # Dispatch packages to the queue with shortest queue length.
#         act = [0]*(len(obs)+2)
#         idx = np.argmin(obs)
#         if obs[idx] < self.conf.queue_max_len:
#             act[idx+1] = 1
#         else:
#             act[0] = 1
#
#
# class PPOCommPolicy(BasicPolicy):
#     def __init__(self, conf):
#         super().__init__(conf)
#
#     def get_actions(self, obs):
#         # Dispatch packages to the queue with shortest queue length.
#         act = [0]*(len(obs)+2)
#         idx = np.argmin(obs)
#         if obs[idx] < self.conf.queue_max_len:
#             act[idx+1] = 1
#         else:
#             act[0] = 1
