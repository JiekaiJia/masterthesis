import numpy as np
import torch
from ray.rllib.policy.policy import Policy


class JSQHeuristic(Policy):
    """Choose the shortest queue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # Dispatch packages to the queue with shortest queue length.
        def select(obs):
            act = np.asarray([-float("inf")] * len(obs))
            obs = np.asarray(obs)
            if obs[0] > obs[1]:
                act[1] = 0
            elif obs[0] < obs[1]:
                act[0] = 0
            else:
                idx = np.random.choice(range(len(obs)))
                act[idx] = 0
            return torch.from_numpy(act).view(1, len(obs))

        return torch.cat([select(x) for x in obs_batch], dim=0), [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
