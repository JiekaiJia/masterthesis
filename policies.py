import numpy as np


class BasicPolicy:
    def __init__(self, conf, env):
        self.conf = conf
        self.action_spaces = env.action_spaces
        self.act_dim = env.action_spaces[env.schedulers[0]].shape[0]

    def compute_actions(self, obs):
        raise NotImplementedError()


class RandomPolicy(BasicPolicy):
    def __init__(self, conf, env):
        super().__init__(conf, env)

    def compute_gactions(self, obs):
        # Dispatch packages to the queues with gaussian distributions.
        act = np.random.randn(self.act_dim)
        return act

    def compute_actions(self, obs):
        # Dispatch packages to a random queue.
        idx = np.random.choice(range(self.act_dim))
        act = [-float('inf')] * self.act_dim
        act[idx] = 0
        return act


class JSQPolicy(BasicPolicy):
    def __init__(self, conf, env):
        super().__init__(conf, env)

    def compute_actions(self, obs):
        # Dispatch packages to the queue with shortest queue length.
        act = [-float("inf")] * self.act_dim
        obs = np.asarray(obs)
        if obs[0] > obs[1]:
            act[1] = 0
        elif obs[0] < obs[1]:
            act[0] = 0
        else:
            idx = np.random.choice(range(self.act_dim))
            act[idx] = 0

        return act
