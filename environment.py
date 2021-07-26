import json

import gym
from gym import spaces
import numpy as np
from scipy.special import softmax

from scenario import Scenario


class QueueingNetwork(gym.Env):
    def __init__(self, scenario):
        self.seed()

        self.scenario = scenario

        self.schedulers = [scheduler.name for scheduler in self.scenario.schedulers]
        self.servers = [server.name for server in self.scenario.servers]
        self.num_schedulers = len(self.schedulers)
        self._index_map = {scheduler.name: idx for idx, scheduler in enumerate(self.scenario.schedulers)}

        max_q_len = max([server.queue_max_len for server in self.scenario.servers])

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        for scheduler in self.scenario.schedulers:
            action_dim = self.scenario.dim_a
            if not scheduler.silent:
                action_dim *= self.scenario.dim_c

            obs_dim = len(self.scenario.observation(scheduler))
            self.action_spaces[scheduler.name] = spaces.Box(low=0, high=1, shape=(action_dim,), dtype=np.float32)
            # self.action_spaces[scheduler.name] = spaces.Discrete(action_dim)
            self.observation_spaces[scheduler.name] = spaces.Box(low=0, high=max_q_len, shape=(obs_dim,), dtype=np.int32)

        self.rewards = {}
        self.dones = {scheduler: False for scheduler in self.schedulers}
        self.acc_drop_pkgs = {scheduler: 0 for scheduler in self.schedulers}

        self.steps = 0

        self.current_actions = {}

    def reset(self):
        return {scheduler: np.zeros(self.observation_spaces[scheduler].shape) for scheduler in self.schedulers}

    def state(self):
        return self.scenario.env_state()

    def _execute_step(self):
        # Set action for each scheduler
        for i, scheduler in enumerate(self.scenario.schedulers):
            # Because actions are continuous actions*message actions, we choose message actions
            # according to continuous actions norm.
            row = -1
            if not scheduler.silent:
                row = self.scenario.dim_c
            acts = self.current_actions[i].reshape(row, -1)
            act_norm = [np.linalg.norm(acts[x, :]) for x in range(row)]
            idx = np.argmax(act_norm)

            comm_action = idx
            action = acts[idx, :].reshape(-1,)

            if not scheduler.silent:
                scenario_action = [softmax(action), comm_action]
            else:
                scenario_action = [softmax(action)]

            self._set_action(scenario_action, scheduler)
        # Q length before server receives packages.
        pre_q_len = {server.name: len(server) for server in self.scenario.servers}

        # Send packages according to actions.
        for scheduler in self.scenario.schedulers:
            key = scheduler.name
            self.scenario.drop_pkgs[key] = 0
            packages = self.scenario.packages[self._index_map[key]].packages
            if not scheduler and not packages:
                self.dones[key] = True
            self._update_msg(scheduler)
            if self.dones[key]:
                continue
            # Scheduler will receive the packages.
            rec_pkgs = 0
            while packages and self.steps >= packages[0].arriving_time:
                rec_pkgs += 1
                scheduler.receive(packages.pop(0))
            print(key + f' receives {rec_pkgs} packages.')
            # Send packages according to actions
            if not scheduler:
                continue

            send_pkgs = scheduler.send()

            if not send_pkgs:
                continue
            # If server queue is not full.
            for package in send_pkgs:
                if not self.scenario.servers[package.target-1]:
                    self.scenario.servers[package.target-1].receive(package)
                else:
                    self.scenario.drop_pkgs[key] += 1
                    self.acc_drop_pkgs[key] += 1

        # Q length after server receives packages.
        cur_q_len = {server.name: len(server) for server in self.scenario.servers}
        for s in self.servers:
            print(s + f' receives {cur_q_len[s] - pre_q_len[s]} packages.')

        # Serve the package.
        for server in self.scenario.servers:
            server.serve()
            print(f'{len(server.leaving_pkgs)} packages left ' + server.name + '.')

        # Calculate the rewards.
        for scheduler in self.scenario.schedulers:
            if self.dones[scheduler.name]:
                scheduler_reward = 0
            else:
                scheduler_reward = float(self.scenario.reward(scheduler))

            self.rewards[scheduler.name] = scheduler_reward

    def _set_action(self, action, scheduler):
        # set env action for a particular agent
        scheduler.action.a = action[0]
        action = action[1:]
        if not scheduler.silent:
            # communication action
            scheduler.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def _update_msg(self, scheduler):
        # set communication messages (directly for now)
        scheduler.msg = np.zeros(self.scenario.dim_c)
        if not scheduler.silent and not self.dones[scheduler.name]:
            noise = np.random.randn(*scheduler.msg.shape) * scheduler.c_noise if scheduler.c_noise else 0.0
            scheduler.msg[scheduler.action.c] = 1
            scheduler.msg += noise

    def step(self, actions):
        for scheduler, action in actions.items():
            self.current_actions[self._index_map[scheduler]] = action

        self._execute_step()

        observations = {scheduler: self.scenario.observation(self.scenario.schedulers[self._index_map[scheduler]]) for scheduler in self.schedulers}
        rewards = self.rewards
        dones = self.dones
        drop_pkgs = self.scenario.drop_pkgs

        self.steps += 1
        return observations, rewards, dones, {'drop_pkgs': drop_pkgs, 'env_state': self.state()}


class RawEnv(QueueingNetwork):
    def __init__(self, conf):
        scenario = Scenario(conf)
        super().__init__(scenario)
        self.metadata['name'] = "simple_queueing_network_v1"


def all_done(dones):
    for _, done in dones.items():
        if not done:
            return False
    else:
        return True


if __name__ == '__main__':
    with open('./config/simple.json', 'r') as f:
        config = json.loads(f.read())
    env = RawEnv(config)
    obs = env.reset()
    dones = env.dones
    acc_r, t = 0, 0
    while not all_done(dones):
        t += 1
        # Random policy
        actions = {scheduler: softmax(action_space.sample()) for scheduler, action_space in env.action_spaces.items()}
        obs, r, dones, info = env.step(actions)
        for _, _r in r.items():
            acc_r += _r
        print('timestep:', t)
        print('obs:', obs)
        print('rewards:', r)
        print('dones:', dones)
        print(info)
        print('_' * 80)

    print('acc_rewwards:', acc_r)
    print('acc_drop_pkgs:', env.acc_drop_pkgs)

