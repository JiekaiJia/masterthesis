import copy
import json

import gym
from gym import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scipy.special import softmax

from scenario import SimpleScenario, FixPartialAccessScenario


class QueueingNetwork(gym.Env):
    def __init__(self, scenario):
        self.scenario = scenario

        self.schedulers = [scheduler.name for scheduler in self.scenario.schedulers]
        self.servers = [server.name for server in self.scenario.servers]
        self.num_schedulers = len(self.schedulers)
        self._index_map = {scheduler.name: idx for idx, scheduler in enumerate(self.scenario.schedulers)}
        self.messages = {scheduler.name: scheduler.msg for idx, scheduler in enumerate(self.scenario.schedulers)}

        max_q_len = max([server.queue_max_len for server in self.scenario.servers])

        # set action spaces and observation spaces.
        self.action_spaces = dict()
        self.observation_spaces = dict()
        for scheduler in self.scenario.schedulers:
            action_dim = self.scenario.dim_a
            if not scheduler.silent:
                action_dim *= self.scenario.dim_c

            obs_dim = len(self.scenario.observation(scheduler))
            self.action_spaces[scheduler.name] = spaces.Box(low=0, high=1, shape=(action_dim,), dtype=np.float32)
            # observe its own length.
            self.observation_spaces[scheduler.name] = spaces.Box(low=0, high=max_q_len, shape=(obs_dim,), dtype=np.int32)

        # global discrete timestep.
        self.steps = 1

    def reset(self):
        self.rewards = {}
        self.schedulers = [scheduler.name for scheduler in self.scenario.schedulers]
        self.infos = {scheduler: None for scheduler in self.schedulers}
        self.dones = {scheduler: False for scheduler in self.schedulers}
        self.acc_drop_pkgs = {scheduler: 0 for scheduler in self.schedulers}
        self.current_actions = {}
        self.steps = 1
        self.scenario.reset()
        return {scheduler: np.zeros(self.observation_spaces[scheduler].shape) for scheduler in self.schedulers}

    def state(self):
        return self.scenario.env_state()

    def _execute_step(self):
        # Set action for each scheduler
        for name in self.schedulers:
            scheduler = self.scenario.schedulers[self._index_map[name]]
            # Because actions are continuous actions*message actions, we choose message actions
            # according to continuous actions norm.
            row = -1
            if not scheduler.silent:
                row = self.scenario.dim_c
            acts = self.current_actions[self._index_map[name]].reshape(row, -1)
            act_norm = [np.linalg.norm(acts[x, :]) for x in range(row)]
            idx = np.argmax(act_norm)

            comm_action = idx
            action = acts[idx, :].reshape(-1,)

            if not scheduler.silent:
                scenario_action = [softmax(action), comm_action]
            else:
                scenario_action = [softmax(action)]

            self._set_action(scenario_action, scheduler)

        # Collect all schedulers' sending packages in order of time
        transmit_q = []
        for name in self.schedulers:
            scheduler = self.scenario.schedulers[self._index_map[name]]
            key = scheduler.name
            self.scenario.drop_pkgs[key] = 0

            packages = self.scenario.packages[self._index_map[key]].packages
            # If scheduler has no packages, then it will be set to done.
            if not scheduler and not packages:
                self.dones[key] = True
            self._update_msg(scheduler)
            if self.dones[key]:
                continue
            rec_pkgs = 0
            # Scheduler will receive the packages when packages arriving time is smaller than global discrete timestep.
            while packages and self.steps >= packages[0].arriving_time:
                rec_pkgs += 1
                scheduler.receive(packages.pop(0))
            # print(key + f' receives {rec_pkgs} packages.')
            if not scheduler:
                continue
            # Scheduler sends the packages.
            send_pkgs = scheduler.send()

            if not send_pkgs:
                continue
            # Collect the packages.
            transmit_q.extend(send_pkgs)
        # Sort packages in order of time
        transmit_q.sort(key=lambda x: x.sending_time, reverse=False)

        for package in transmit_q:
            # Servers receive and serve the packages
            dropped_pkgs = self.scenario.servers[package.target - 1].receive(package)
            # If a package is dropped, then the corresponding scheduler would store the number.
            if dropped_pkgs:
                self.scenario.drop_pkgs[dropped_pkgs.sender] += 1
                self.acc_drop_pkgs[dropped_pkgs.sender] += 1
        else:
            # Serve the packages that should departure before current timestep.
            for server in self.scenario.servers:
                server.serve(self.steps)

        # Calculate the rewards.
        for name in self.schedulers:
            scheduler = self.scenario.schedulers[self._index_map[name]]
            if self.dones[name]:
                scheduler_reward = 0
            else:
                scheduler_reward = float(self.scenario.reward(scheduler))

            self.rewards[scheduler.name] = scheduler_reward

    def _set_action(self, action, scheduler):
        # set env action for an agent
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
        # 1.method: Message is a mapping of observation.
        scheduler.msg = np.zeros(self.scenario.dim_c)
        if not scheduler.silent and not self.dones[scheduler.name]:
            noise = np.random.randn(*scheduler.msg.shape) * scheduler.c_noise if scheduler.c_noise else 0.0
            scheduler.msg[scheduler.action.c] = 1
            scheduler.msg += noise
            self.messages[scheduler.name] = scheduler.msg

        # 2.method: Message indicates weather to send observations, 0: not send, 1: send.
        # if not scheduler.silent and not self.dones[scheduler.name]:
        #     scheduler.msg = scheduler.action.c

    def step(self, actions):
        for scheduler in self.schedulers:
            if self.dones[scheduler]:
                del self.dones[scheduler]
                del self.rewards[scheduler]
                del self.infos[scheduler]
                self.schedulers.remove(scheduler)

        for scheduler, action in actions.items():
            self.current_actions[self._index_map[scheduler]] = action

        self._execute_step()

        observations = {scheduler: self.scenario.observation(self.scenario.schedulers[self._index_map[scheduler]]) for scheduler in self.schedulers}
        rewards = self.rewards
        dones = self.dones
        infos = self.infos

        drop_pkgs = self.scenario.drop_pkgs

        self.steps += 1
        return observations, rewards, dones, infos


class RawEnv(QueueingNetwork):
    def __init__(self, conf):
        scenario = FixPartialAccessScenario(conf)
        super().__init__(scenario)
        self.metadata['name'] = 'simple_queueing_network_v1'


class RLlibEnv(MultiAgentEnv):
    """Wraps Queueing env to be compatible with RLLib multi-agent."""

    def __init__(self, conf):
        """Create a new queueing network env compatible with RLlib."""

        self.raw_env = RawEnv(conf)

        self.observation_spaces = self.raw_env.observation_spaces
        self.action_spaces = self.raw_env.action_spaces
        self.schedulers = self.raw_env.schedulers

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs_dict: New observations for each ready agent.
        """
        obss = self.raw_env.reset()
        self.acc_drop_pkgs = self.raw_env.acc_drop_pkgs
        return obss

    def step(self, actions):
        obss, rews, dones, infos = self.raw_env.step(actions)
        infos = {k: {'done': done} for k, done in dones.items()}
        _dones = [v for _, v in dones.items()]
        dones_ = copy.deepcopy(dones)
        dones_['__all__'] = all(_dones)
        return obss, rews, dones_, infos


class MainEnv:
    """"""

    def __init__(self, conf):
        """"""

        self.raw_env = RawEnv(conf)

        self.observation_spaces = self.raw_env.observation_spaces
        self.action_spaces = self.raw_env.action_spaces
        self.schedulers = self.raw_env.schedulers
        self.messages = self.raw_env.messages

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs_dict: New observations for each ready agent.
        """
        obss = self.raw_env.reset()
        self.acc_drop_pkgs = self.raw_env.acc_drop_pkgs
        return obss

    def step(self, actions):
        obss, rews, dones, infos = self.raw_env.step(actions)
        infos = self.messages
        _dones = [v for _, v in dones.items()]
        dones_ = copy.deepcopy(dones)
        dones_['__all__'] = all(_dones)
        return obss, rews, dones_, infos


if __name__ == '__main__':
    with open('./config/FixPartialAccess.json', 'r') as f:
        config = json.loads(f.read())
    env = MainEnv(config)
    dones = {}
    for i in range(1):
        obs = env.reset()
        dones['__all__'] = False
        acc_r, t = 0, 0
        while not dones['__all__']:
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
            print('messages:', info)
            print('_' * 80)

        print('acc_rewwards:', acc_r)
        print('acc_drop_pkgs:', env.acc_drop_pkgs)

