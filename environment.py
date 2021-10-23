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
        # Set the current action distribution and communication probability.
        for scheduler, action in actions.items():
            self.current_actions[scheduler] = action[:-1]
            self.p[scheduler] = action[-1]

        self._action_trans()
        self._execute_step()

        for server in self.scenario.servers:
            server.history_len.append(len(server))
        # Change the delay time every n steps.
        # todo: Can delay time changing be useful?
        if self.steps % self.frequency == 0:
            self.scenario.reset_delay_t()
        self.steps += 1
        return self.observe(), self.rewards, self.dones, self.infos


class QueueingNetwork3(QueueingNetwork2):
    """This network is based on the configurations of Queueingnetwork2, and addresses beliefs over queue state
    as observations and messages between agents. Schedulers have a probability to decide whether to receive messages."""
    def __init__(self, scenario):
        super().__init__(scenario)
        self.training = self.scenario.conf.training
        self.bs = self.scenario.conf.bs

        # set observation spaces.
        for scheduler in self.scenario.schedulers:
            # Belief as observation, the scheduler thinks how many packages are in the queue.
            self.observation_spaces[scheduler.name] = spaces.Box(low=float('-inf'), high=float('inf'), shape=(self.scenario.obs_servers, self.max_q_len + 1 + 1), dtype=np.float32)

        # Not every scenario has the comm_group.
        try:
            self.comm_group = self.scenario.comm_group
        except AttributeError:
            pass

        self.model = MVAE(self.max_q_len + 1, len(self.schedulers), self.max_q_len + 1,
                          self.observation_spaces[self.schedulers[0]].shape[0],
                          self.scenario.schedulers, self.training, self.bs, self.comm_group)

        if not self.training:
            self.model.load_state_dict(
                torch.load(self.scenario.conf.check_path + f'/belief_encoder{self.num_schedulers}.pth')['state_dict']
            )

    def step(self, actions):
        observations, rewards, dones, infos = super().step(actions)
        # Transform observations to NxL tensors, where N is the number of schedulers
        # and L is the length of observations.
        obs = [None] * self.num_schedulers
        real_obs = [None] * self.num_schedulers
        for k, v in observations.items():
            obs[self._index_map[k]] = torch.from_numpy(np.array(v[0]))
            real_obs[self._index_map[k]] = torch.from_numpy(np.array(v[1]))

        if self.training:
            self.model.train()
            self.p = {scheduler: 1 for scheduler in self.schedulers}
            recon_obs, mu, logvar = self.model(self.p, obs)
        else:
            self.model.eval()
            with torch.no_grad():
                recon_obs, mu, logvar = self.model(self.p, obs)
        recon_obs0, mu0, logvar0 = self.model({scheduler: 0 for scheduler in self.schedulers}, obs)

        recon_obss = []
        for x in recon_obs:
            if x is None:
                recon_obss.append(torch.zeros((1, self.scenario.obs_servers)))
            else:
                recon_obss.append(torch.cat([y.argmax().unsqueeze(0) for y in x], dim=0))
        recon_obss0 = []
        for x in recon_obs0:
            if x is None:
                recon_obss0.append(torch.zeros((1, self.scenario.obs_servers)))
            else:
                recon_obss0.append(torch.cat([y.argmax().unsqueeze(0) for y in x], dim=0))

        return (obs, recon_obs, mu, logvar, recon_obss, recon_obs0, mu0, logvar0, recon_obss0, real_obs), self.rewards, self.dones, self.infos


class QueueingNetwork4(QueueingNetwork3):
    """This environment deletes the done schedulers during episodes to keep compatible with RLlib."""

    def __init__(self, scenario):
        super().__init__(scenario)

    def step(self, actions):
        # Check whether the scheduler is done, if done then delete it.
        for scheduler in self.schedulers:
            self.dlt_agent(scheduler)
        return super().step(actions)


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
        self.schedulers = self.raw_env.schedulers
        self.acc_drop_pkgs = self.raw_env.acc_drop_pkgs
        return obss

    def step(self, actions):
        obss, rews, dones, infos = self.raw_env.step(actions)
        self.schedulers = self.raw_env.schedulers
        belief = {scheduler: torch.cat(obss[1][self.raw_env._index_map[scheduler]], dim=0).cpu().numpy() for scheduler in self.schedulers}
        new_obs = {k: np.concatenate((v, np.array(obss[0][self.raw_env._index_map[k]]).reshape(self.raw_env.scenario.conf.obs_servers, 1)), axis=1) for k, v in belief.items()}
        dones_ = {k: dones[k] for k in self.schedulers}
        dones_['__all__'] = all(dones.values())
        infos = {k: {'done': dones[k]} for k in self.schedulers}
        return new_obs, rews, dones_, infos


class MainEnv:
    """"""
    def __init__(self, conf):
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
    def __init__(self, make_env_fn, n, config):
        self.envs = tuple(make_env_fn(config) for _ in range(n))
        self.action_spaces = self.envs[0].action_spaces

    # Call this only once at the beginning of training (optional):
    def seed(self, seeds):
        assert len(self.envs) == len(seeds)
        return tuple(env.seed(s) for env, s in zip(self.envs, seeds))

    def state(self):
        return tuple(env.state() for env in self.envs)

    def reset(self):
        return_value = tuple(env.reset() for env in self.envs)
        self.acc_drop_pkgs = (env.acc_drop_pkgs for env in self.envs)
        return return_value

    # Call this on every timestep:
    def step(self, actions):
        assert len(self.envs) == len(actions)
        return_values = []
        for env, a in zip(self.envs, actions):
            observation, reward, done, info = env.step(a)
            if done['__all__']:
                observation = env.reset()
            return_values.append((observation, reward, done, info))
        return tuple(return_values)


if __name__ == '__main__':
    from dotdic import DotDic
    with open('config/PartialAccess.json', 'r') as f:
        config = json.loads(f.read())
    config = DotDic(config)
    config.training = False
    env = RLlibEnv(config)
    for _ in range(3):
        obss = env.reset()
        dones = {'__all__': False}
        t = 0
        while not dones['__all__']:
            t += 1
            # Random polic
            actions = {scheduler: action_space.sample() for scheduler, action_space in env.action_spaces.items()}
            obs, r, dones, info = env.step(actions)
            print('timestep:', t)
            print('obs:', obs.keys())
            print('rewards:', r.keys())
            print('dones:', dones.keys())
            print('messages:', info.keys())
            print('_' * 80)
    # num_envs = config.bs
    # envs = VectorEnv(MainEnv, num_envs, config)
    # for i in range(1):
    #     obss = envs.reset()
    #     dones = ({'__all__': False} for _ in range(num_envs))
    #     acc_r = [0 for _ in range(num_envs)]
    #     t = 0
    #     while not all(done['__all__'] for done in dones):
    #         t += 1
    #         # Random policy
    #         actions = [{scheduler: softmax(action_space.sample()) for scheduler, action_space in envs.action_spaces.items()} for _ in range(num_envs)]
    #         obs, r, dones, info = envs.step(actions)
    #         for i, rr in enumerate(r):
    #             for _, _r in rr.items():
    #                 acc_r[i] += _r
    #         print('timestep:', t)
    #         print('state:', envs.state())
    #         print('obs:', (o[0] for o in obs))
    #         print('rewards:', r)
    #         print('dones:', dones)
    #         # print('messages:', info)
    #         print('_' * 80)
    #
    #     print('acc_rewwards:', acc_r)
    #     print('acc_drop_pkgs:', envs.acc_drop_pkgs)
