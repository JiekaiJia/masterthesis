import gym
from gym import spaces
import numpy as np

from scenario import Scenario


class QueueingNetwork(gym.Env):

    def __init__(self, scenario, local_ratio):
        self.seed()

        self.scenario = scenario
        self.local_ratio = local_ratio

        self.schedulers = [scheduler.name for scheduler in self.scenario.schedulers]
        self.num_schedulers = len(self.schedulers)
        self._index_map = {scheduler.name: idx for idx, scheduler in enumerate(self.scenario.schedulers)}

        max_q_len = max([server.queue_max_len for server in self.scenario.servers])

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for scheduler in self.scenario.schedulers:
            action_dim = self.scenario.dim_a
            if not scheduler.silent:
                action_dim *= self.scenario.dim_c

            obs_dim = len(self.scenario.observation(scheduler))
            state_dim += obs_dim
            self.action_spaces[scheduler.name] = spaces.Discrete(action_dim)
            self.observation_spaces[scheduler.name] = spaces.Box(low=0, high=max_q_len, shape=(obs_dim,), dtype=np.int32)

        self.state_space = spaces.Box(low=0, high=max_q_len, shape=(state_dim,), dtype=np.int32)

        self.rewards = {}
        self.dones = {scheduler: False for scheduler in self.schedulers}
        self.acc_drop_pkgs = {scheduler: 0 for scheduler in self.schedulers}

        self.steps = 0

        self.current_actions = [0] * self.num_schedulers

    def reset(self):
        return {scheduler: np.zeros(self.observation_spaces[scheduler].shape) for scheduler in self.schedulers}

    def seed(self, seed=None):
        pass

    def state(self):
        pass

    def _execute_step(self):
        # Set action for each scheduler
        for i, scheduler in enumerate(self.scenario.schedulers):
            action = self.current_actions[i] % self.scenario.dim_a
            comm_action = self.current_actions[i] // self.scenario.dim_a
            if not scheduler.silent:
                scenario_action = [action, comm_action]
            else:
                scenario_action = [action]

            self._set_action(scenario_action, scheduler)

        for scheduler in self.scenario.schedulers:
            key = scheduler.name
            self.scenario.drop_pkgs[key] = 0
            if self.dones[key]:
                continue
            packages = self.scenario.packages[self._index_map[key]].packages
            while packages and self.steps >= packages[0].arriving_time:
                scheduler.receive(packages[0])
                package = scheduler.send()
                if package:
                    # If server queue is not full.
                    if not self.scenario.servers[package.target-1]:
                        self.scenario.servers[package.target-1].receive(package)
                    else:
                        self.scenario.drop_pkgs[key] += 1
                        self.acc_drop_pkgs[key] += 1
                packages.pop(0)
            if not packages:
                self.dones[key] = True

        for server in self.scenario.servers:
            server.serve()

        global_reward = 0.
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward())

        for scheduler in self.scenario.schedulers:
            if self.dones[scheduler.name]:
                scheduler_reward = 0
            else:
                scheduler_reward = float(self.scenario.reward(scheduler))
            if self.local_ratio is not None:
                reward = global_reward * (1 - self.local_ratio) + scheduler_reward * self.local_ratio
            else:
                reward = scheduler_reward

            self.rewards[scheduler.name] = reward

    # set env action for a particular agent
    def _set_action(self, action, scheduler):
        scheduler.action.a = action[0]
        action = action[1:]
        if not scheduler.silent:
            # communication action
            scheduler.action.c = np.zeros(self.scenario.dim_c)
            scheduler.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, actions):
        for scheduler, action in actions.items():
            self.current_actions[self._index_map[scheduler]] = action

        self._execute_step()

        observations = {scheduler: self.scenario.observation(self.scenario.schedulers[self._index_map[scheduler]]) for scheduler in self.schedulers}
        rewards = self.rewards
        dones = self.dones
        drop_pkgs = self.scenario.drop_pkgs

        self.steps += 1
        return observations, rewards, dones, drop_pkgs


def make_env():
    scenario = Scenario()
    return QueueingNetwork(scenario, 0.5)


if __name__ == '__main__':
    env = make_env()
    obs = env.reset()
    print(obs)
    for i in range(20):
        obs, r, dones, info = env.step({scheduler: action_space.sample() for scheduler, action_space in env.action_spaces.items()})
        # obs, r, dones, info = env.step(
        #     {scheduler:1 for scheduler, action_space in env.action_spaces.items()})
        print('timestep', i+1)
        print(obs)
        print(r)
        print(info)
        print('_' * 80)

