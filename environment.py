import copy
import json

import gym
import torch
from gym import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scipy.special import softmax

from models import MVAE
from scenario import SimpleScenario, PartialAccessScenario, PartialcommScenario


class BasicNetwork(gym.Env):
    def __init__(self, scenario):
        self.scenario = scenario

        self.schedulers = [scheduler.name for scheduler in self.scenario.schedulers]
        self.servers = [server.name for server in self.scenario.servers]
        self.num_schedulers = len(self.schedulers)
        self._index_map = {scheduler.name: idx for idx, scheduler in enumerate(self.scenario.schedulers)}
        self.messages = {scheduler.name: scheduler.msg for idx, scheduler in enumerate(self.scenario.schedulers)}

        # global discrete time step.
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

    def _execute_step(self):
        # Collect all schedulers' sending packages with the order of time
        transmit_q = []
        for name in self.schedulers:
            scheduler = self.scenario.schedulers[self._index_map[name]]
            key = scheduler.name
            self.scenario.drop_pkgs[key] = 0

            packages = self.scenario.packages[self._index_map[key]].packages
            # If scheduler has no packages, then it will be set to done.
            if not scheduler and not packages:
                self.dones[key] = True

            # Not every environment needs update messages.
            try:
                self._update_msg(scheduler)
            except AttributeError:
                pass

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
        # Sort packages with the order of time
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
        scheduler.msg = np.zeros(self.scenario.dim_c)
        if not scheduler.silent and not self.dones[scheduler.name]:
            noise = np.random.randn(*scheduler.msg.shape) * scheduler.c_noise if scheduler.c_noise else 0.0
            scheduler.msg[scheduler.action.c] = 1
            scheduler.msg += noise
            self.messages[scheduler.name] = scheduler.msg

    def dlt_agent(self, scheduler):
        if self.dones[scheduler]:
            del self.dones[scheduler]
            del self.rewards[scheduler]
            del self.infos[scheduler]
            self.schedulers.remove(scheduler)

    def _action_trans(self):
        raise NotImplementedError

    def state(self):
        return self.scenario.env_state()

    def observe(self):
        return {scheduler: self.scenario.observation(self.scenario.schedulers[self._index_map[scheduler]]) for scheduler in self.schedulers}

    def step(self, actions):
        raise NotImplementedError()


class QueueingNetwork1(BasicNetwork):
    """In this network the schedulers have access to all servers but with only partial observations.
    Each scheduler broadcasts messages."""
    def __init__(self, scenario):
        super().__init__(scenario)

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

    def reset(self):
        super().reset()
        return {scheduler: np.zeros(self.observation_spaces[scheduler].shape) for scheduler in self.schedulers}

    def _action_trans(self):
        # Set action for each scheduler
        for name in self.schedulers:
            scheduler = self.scenario.schedulers[self._index_map[name]]
            # Because actions are continuous actions*message actions, we choose message actions
            # according to continuous actions norm.
            row = -1
            if not scheduler.silent:
                row = self.scenario.dim_c
            acts = self.current_actions[name].reshape(row, -1)
            act_norm = [np.linalg.norm(acts[x, :]) for x in range(row)]
            idx = np.argmax(act_norm)

            comm_action = idx
            action = acts[idx, :].reshape(-1,)

            if not scheduler.silent:
                scenario_action = [softmax(action), comm_action]
            else:
                scenario_action = [softmax(action)]

            self._set_action(scenario_action, scheduler)

    def step(self, actions):
        # Check whether the scheduler is done, if done then delete it.
        for scheduler in self.schedulers:
            self.dlt_agent(scheduler)
        # Set the current action distribution.
        for scheduler, action in actions.items():
            self.current_actions[scheduler] = action

        self._action_trans()
        self._execute_step()

        self.steps += 1
        return self.observe(), self.rewards, self.dones, self.infos


class QueueingNetwork2(BasicNetwork):
    """In this network the schedulers observe queue state with a transmit delay. This delay time will change every
    n steps. Besides, the schedulers only have partial access and observability to the queue."""
    def __init__(self, scenario):
        super().__init__(scenario)
        self.frequency = self.scenario.frequency

        self.max_q_len = max([server.queue_max_len for server in self.scenario.servers])

        # set action spaces and observation spaces.
        self.action_spaces = dict()
        self.observation_spaces = dict()
        for scheduler in self.scenario.schedulers:
            action_dim = self.scenario.dim_a

            obs_dim = len(self.scenario.observation(scheduler))
            # Action space is the action distribution + communication probability.
            self.action_spaces[scheduler.name] = spaces.Box(low=0, high=1, shape=(action_dim+1,), dtype=np.float32)
            # observe queue's length.
            self.observation_spaces[scheduler.name] = spaces.Box(low=0, high=self.max_q_len, shape=(obs_dim,), dtype=np.int32)

    def reset(self):
        super().reset()
        # Probability indicates whether the scheduler need communication.
        self.p = {}
        return {scheduler: np.zeros(self.observation_spaces[scheduler].shape) for scheduler in self.schedulers}

    def _action_trans(self):
        # Set action for each scheduler
        for name in self.schedulers:
            scheduler = self.scenario.schedulers[self._index_map[name]]
            scheduler.action.a = softmax(self.current_actions[name])

    def step(self, actions):
        # Check whether the scheduler is done, if done then delete it.
        # for scheduler in self.schedulers:
        #     self.dlt_agent(scheduler)
        # Set the current action distribution and communication probability.
        for scheduler, action in actions.items():
            self.current_actions[scheduler] = action[:-1]
            self.p[scheduler] = action[-1]

        self._action_trans()
        self._execute_step()

        for server in self.scenario.servers:
            server.history_len.append(len(server))
        # Change the delay time every n steps.
        if self.steps % self.frequency == 0:
            self.scenario.set_delay_t(np.random.randint(0, 6))
        self.steps += 1
        return self.observe(), self.rewards, self.dones, self.infos


class QueueingNetwork3(QueueingNetwork2):
    """This network is based on the configurations of Queueingnetwork2, and addresses beliefs over queue state
    as observations and messages between agents. Schedulers have a probability to decide whether to receive messages."""
    def __init__(self, scenario):
        super().__init__(scenario)
        self.training = self.scenario.conf['training']
        self.bs = self.scenario.conf['bs']

        # set observation spaces.
        for scheduler in self.scenario.schedulers:
            # Belief as observation, the scheduler thinks how many packages are in the queue.
            self.observation_spaces[scheduler.name] = spaces.Box(low=0, high=1,
                                                                 shape=(self.scenario.obs_servers, self.max_q_len + 1),
                                                                 dtype=np.int32)

        # Not every scenario has the comm_group.
        try:
            self.comm_group = self.scenario.comm_group
        except AttributeError:
            pass

        self.model = MVAE(self.max_q_len + 1, len(self.schedulers), self.max_q_len + 1,
                          self.observation_spaces[self.schedulers[0]].shape[0],
                          self.scenario.schedulers, self.training, self.bs, self.comm_group)

    def step(self, actions):
        observations, rewards, dones, infos = super().step(actions)
        # Transform observations to NxL tensors, where N is the number of schedulers
        # and L is the length of observations.
        obs = []
        for _, v in observations.items():
            obs.append(torch.from_numpy(np.array(v)))
        if self.training:
            self.model.train()
            self.p = {scheduler: 1 for scheduler in self.schedulers}
        else:
            self.model.eval()
        recon_obs, mu, logvar = self.model(self.p, obs)

        recon_obss = []
        for x in recon_obs:
            recon_obss.append(torch.cat([y.argmax().unsqueeze(0) for y in x], dim=0))

        # beliefs = {scheduler: mu[i].cpu().detach().numpy().tolist() for i, scheduler in enumerate(self.schedulers)}

        return [obs, recon_obs, mu, logvar, recon_obss], self.rewards, self.dones, self.infos


class RawEnv(QueueingNetwork3):
    def __init__(self, conf):
        scenario = PartialcommScenario(conf)
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
        # Check whether the scheduler is done, if done then delete it.
        for scheduler in self.schedulers:
            self.dlt_agent(scheduler)
        obss, rews, dones, infos = self.raw_env.step(actions)
        beliefs = {scheduler: obss[2][i].cpu().detach().numpy().tolist() for i, scheduler in enumerate(self.schedulers)}
        infos = {k: {'done': done} for k, done in dones.items()}
        _dones = [v for _, v in dones.items()]
        dones_ = copy.deepcopy(dones)
        dones_['__all__'] = all(_dones)
        return beliefs, rews, dones_, infos

    def dlt_agent(self, scheduler):
        if self.dones[scheduler]:
            del self.dones[scheduler]
            del self.rewards[scheduler]
            del self.infos[scheduler]
            self.schedulers.remove(scheduler)


class MainEnv:
    """"""

    def __init__(self, conf):
        """"""

        self.raw_env = RawEnv(conf)

        self.observation_spaces = self.raw_env.observation_spaces
        self.action_spaces = self.raw_env.action_spaces
        self.schedulers = self.raw_env.schedulers
        self.model = self.raw_env.model
        self.messages = self.raw_env.messages

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs_dict: New observations for each ready agent.
        """
        obss = self.raw_env.reset()
        self.acc_drop_pkgs = self.raw_env.acc_drop_pkgs
        return obss

    def state(self):
        return self.raw_env.state()

    def step(self, actions):
        obss, rews, dones, infos = self.raw_env.step(actions)
        infos = self.messages
        _dones = [v for _, v in dones.items()]
        dones_ = copy.deepcopy(dones)
        dones_['__all__'] = all(_dones)
        return obss, rews, dones_, infos


class VectorEnv:
    def __init__(self, make_env_fn, n):
        self.envs = tuple(make_env_fn() for _ in range(n))

    # Call this only once at the beginning of training (optional):
    def seed(self, seeds):
        assert len(self.envs) == len(seeds)
        return tuple(env.seed(s) for env, s in zip(self.envs, seeds))

    # Call this only once at the beginning of training:
    def reset(self):
        return tuple(env.reset() for env in self.envs)

    # Call this on every timestep:
    def step(self, actions):
        assert len(self.envs) == len(actions)
        return_values = []
        for env, a in zip(self.envs, actions):
            observation, reward, done, info = env.step(a)
            if done:
                observation = env.reset()
            return_values.append((observation, reward, done, info))
        return tuple(return_values)

    # Call this at the end of training:
    def close(self):
        for env in self.envs:
            env.close()


if __name__ == '__main__':
    with open('config/PartialAccess.json', 'r') as f:
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
            print('state:', env.state())
            print('obs:', obs[1])
            print('rewards:', r)
            print('dones:', dones)
            # print('messages:', info)
            print('_' * 80)

        print('acc_rewwards:', acc_r)
        print('acc_drop_pkgs:', env.acc_drop_pkgs)

