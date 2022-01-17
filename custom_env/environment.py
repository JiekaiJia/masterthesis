import copy
import logging

import gym
from gym.spaces import Box, Discrete, Dict, Tuple
from gym.utils import seeding
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scipy.special import softmax
import torch

from model_components import MVAE
from scenario import PartialAccessScenario
from utils import sigmoid

logger = logging.getLogger(__name__)


class BasicNetwork(gym.Env):
    def __init__(self, cfg, scene_cls):
        self.seed(2021)
        self.scenario = scene_cls(cfg)
        self.act_frequency = cfg.act_frequency
        self.num_schedulers = cfg.n_schedulers
        self.servers = [server.name for server in self.scenario.servers]
        self.schedulers = [scheduler.name for scheduler in self.scenario.schedulers]
        self.index_map = {scheduler: idx for idx, scheduler in enumerate(self.schedulers)}

        # global discrete time step.
        self.steps = 1
        # local discrete time step
        self.local_steps = 0

    def reset(self):
        self.rewards = {}
        self.schedulers = [scheduler.name for scheduler in self.scenario.schedulers]
        self.p = []
        self.infos = {scheduler: None for scheduler in self.schedulers}
        self.dones = {scheduler: False for scheduler in self.schedulers}
        self.acc_drop_pkgs = {scheduler: 0 for scheduler in self.schedulers}
        self.current_actions = {}
        self.steps = 1
        self.scenario.reset()

    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

    def _execute_step(self):
        # Collect all schedulers" sending packages in order of time
        transmit_q = []
        for name in self.schedulers:
            scheduler = self.scenario.schedulers[self.index_map[name]]
            self.scenario.drop_pkgs[name] = 0

            packages = self.scenario.packages[self.index_map[name]].packages
            # If scheduler has no packages, then it will be set to done.
            if not scheduler and not packages:
                self.dones[name] = True

            if self.dones[name]:
                continue
            rec_pkgs = 0
            # Scheduler will receive the packages when packages arriving time is smaller than global discrete timestep.
            while packages and self.steps >= packages[0].arriving_time:
                rec_pkgs += 1
                scheduler.receive(packages.pop(0))
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
            scheduler = self.scenario.schedulers[self.index_map[name]]
            if self.dones[name]:
                scheduler_reward = 0
            else:
                scheduler_reward = float(self.scenario.reward(scheduler))

            self.rewards[name] = scheduler_reward

    def _action_trans(self):
        raise NotImplementedError

    def state(self):
        return self.scenario.env_state()

    def schedulers_state(self):
        return self.scenario.schedulers_state()

    def observe(self):
        return {scheduler: self.scenario.observation(self.scenario.schedulers[self.index_map[scheduler]]) for scheduler
                in self.schedulers}

    def step(self, actions):
        raise NotImplementedError()


class DelayedNetwork(BasicNetwork):
    """In this network the schedulers observe queue state with a transmitting delay. This delay will change every
    n steps. Besides, the schedulers only have partial access and observability to the queue."""

    def __init__(self, cfg, scene_cls):
        super().__init__(cfg, scene_cls)
        self.max_q_len = cfg.max_q_len
        self.delay_change_frequency = cfg.delay_change_frequency

        # set action spaces and observation spaces.
        obs_dim = cfg.num_obs_servers
        action_dim = self.scenario.dim_a
        self.action_spaces, self.observation_spaces = {}, {}
        for scheduler in self.schedulers:
            # Action is the probability distribution sending packages to servers.
            self.action_spaces[scheduler] = Box(low=0, high=1, shape=(action_dim,), dtype=np.float32)
            # Observe queue length.
            self.observation_spaces[scheduler] = Box(low=0, high=self.max_q_len, shape=(obs_dim,), dtype=np.int32)

    def reset(self):
        super().reset()
        return {scheduler: np.zeros(self.observation_spaces[scheduler].shape, dtype=np.int32) for scheduler in self.schedulers}

    def _action_trans(self):
        # Set action for each scheduler
        for name in self.schedulers:
            scheduler = self.scenario.schedulers[self.index_map[name]]
            scheduler.action.a = softmax(self.current_actions[name])

    def no_send_step(self):
        for name in self.schedulers:
            scheduler = self.scenario.schedulers[self.index_map[name]]
            self.scenario.drop_pkgs[name] = 0

            packages = self.scenario.packages[self.index_map[name]].packages
            # If scheduler has no packages, then it will be set to done.
            if not scheduler and not packages:
                self.dones[name] = True

            if self.dones[name]:
                continue
            # Scheduler will receive the packages when packages arriving time is smaller than global discrete timestep.
            while packages and self.steps >= packages[0].arriving_time:
                scheduler.receive(packages.pop(0))
            # Update package halt time.
            scheduler.update_halt_t()
            if not scheduler:
                continue

        # Serve the packages that should departure before current timestep.
        for server in self.scenario.servers:
            server.serve(self.steps)

        # Calculate the rewards.
        for name in self.schedulers:
            self.rewards[name] = 0

    def step(self, actions):
        # Whether do actions or not.
        if self.steps > 1 and self.local_steps < self.act_frequency:
            self.local_steps += 1
            self.no_send_step()
        else:
            # print("do actions")
            self.local_steps = 0
            # Make sure action dimension to be correct.
            # for k, v in actions.items():
            #     assert len(v) == self.action_spaces[k].shape[0], f"Wrong action dimension!! Expected {self.action_spaces[k].shape[0]}, got {len(v)}"
            #     break
            # Set the current action distribution.
            for scheduler, action in actions.items():
                self.current_actions[scheduler] = action

            self._action_trans()
            self._execute_step()

        for server in self.scenario.servers:
            server.history_len.append(len(server))
        # Change the delay time every n steps.
        # todo: Can delay time changing be useful?
        if self.steps % self.delay_change_frequency == 0:
            self.scenario.reset_delay_t()
        self.steps += 1
        return self.observe(), self.rewards, self.dones, self.infos


class BeliefNetwork(DelayedNetwork):
    """This network is based on the configurations of DelayedNetwork, and addresses beliefs over queue state
    as observations and messages between agents. Schedulers have a probability to decide whether to receive messages."""

    def __init__(self, cfg, scene_cls):
        super().__init__(cfg, scene_cls)
        self.belief_training = cfg.belief_training
        self.obs_server_id = self.scenario.obs_server_id

        # set observation spaces.
        action_dim = self.scenario.dim_a
        for scheduler in self.schedulers:
            # Action is the probability distribution sending packages to servers.
            self.action_spaces[scheduler] = Box(low=0, high=1, shape=(action_dim+1,), dtype=np.float32)
            # Belief as observation, the scheduler thinks how many packages are in the queue.
            self.observation_spaces[scheduler] = Box(low=float("-inf"), high=float("inf"),
                                                                     shape=(cfg.n_latents*cfg.num_obs_servers,),
                                                                     dtype=np.float32)
        self.c_group = {}
        for scheduler in self.schedulers:
            self.c_group[scheduler] = []  # [[(),()], [(),()]]
            me_app = self.c_group[scheduler].append
            obs_id = self.obs_server_id[scheduler]
            for id in obs_id:
                l = []
                l_app = l.append
                for i, other in enumerate(self.schedulers):
                    if other != scheduler:
                        other_obs_id = self.obs_server_id[other]
                        if id in other_obs_id:
                            pos = other_obs_id.index(id)
                            l_app((i, pos))
                me_app(l)

        self.model = MVAE(cfg.n_latents, cfg.num_obs_servers, cfg.max_q_len,
                         cfg.belief_hidden_dim, c_group=self.c_group, training=self.belief_training)

        if not self.belief_training:
            try:
                # Must use absolute path, otherwise the other actors except main actor can"t find model parameters.
                # self.model.load_state_dict(torch.load(cfg.restore_from)["state_dict"])
                self.model.load_state_dict(torch.load("/content/drive/MyDrive/DataScience/pythonProject/masterthesis/model_states/belief_encoder3_3955_109.81.pth")["state_dict"])
                print("The model restores from the trained model.")
            except FileNotFoundError:
                print("No existed trained model, using initial parameters.")

    def step(self, actions):
        _actions = {}
        for k, v in actions.items():
            _actions[k] = v[:-1]
            self.p.append(sigmoid(v[-1]))
        observations, rewards, dones, infos = super().step(_actions)
        # Transform observations to NxL tensors, where N is the number of schedulers
        # and L is the length of observations.
        obs = [None] * self.num_schedulers
        real_obs = [None] * self.num_schedulers
        for k, v in observations.items():
            obs[self.index_map[k]] = torch.from_numpy(np.array(v[0]))
            real_obs[self.index_map[k]] = torch.from_numpy(np.array(v[1]))

        if self.belief_training:
            self.model.train()
            decoding, mu, logvar = self.model(obs, self.p)
        else:
            self.model.eval()
            with torch.no_grad():
                decoding, mu, logvar = self.model(obs, self.p)

        return (obs, decoding, real_obs, mu, logvar), self.rewards, self.dones, self.infos


class MainEnv(gym.Env):
    """"""

    def __init__(self, cfg):
        self.use_belief = cfg.use_belief
        if self.use_belief:
            self.raw_env = BeliefNetwork(cfg, PartialAccessScenario)
            self.model = self.raw_env.model
            print("Now using environment with belief!!!")
        else:
            self.raw_env = DelayedNetwork(cfg, PartialAccessScenario)
            print("Now using environment without belief!!!")

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

    def state(self):
        return self.raw_env.state()

    def schedulers_state(self):
        return self.raw_env.schedulers_state()

    def step(self, actions):
        obss, rews, dones, infos = self.raw_env.step(actions)
        _dones = [v for _, v in dones.items()]
        dones_ = copy.deepcopy(dones)
        dones_["__all__"] = all(_dones)
        return obss, rews, dones_, infos


class RLlibEnv(MultiAgentEnv):
    """Wraps Queueing env to be compatible with RLLib multi-agent."""

    def __init__(self, cfg):
        """Create a new queueing network env compatible with RLlib."""
        self.use_belief = cfg.use_belief

        if self.use_belief:
            rlenv = make_rllibenv(BeliefNetwork)
            print("Now using environment with belief!!!")
        else:
            rlenv = make_rllibenv(DelayedNetwork)
            print("Now using environment without belief!!!")

        self.raw_env = rlenv(cfg, PartialAccessScenario)

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
        env = self.raw_env
        obss, rews, dones, infos = env.step(actions)
        total_r = sum(rews.values())
        new_rews = {k: total_r for k, _ in rews.items()}
        self.schedulers = env.schedulers
        dones_ = {k: dones[k] for k in self.schedulers}
        dones_["__all__"] = all(dones.values())
        # infos = {k: obss[k][1] for k in self.schedulers}
        infos = {k: {"done": dones[k]} for k in self.schedulers}
        if self.use_belief:
            tmp_obs = [obs.cpu().detach().numpy().reshape(2, -1) if obs is not None else None for obs in obss[3]]
            norm_obs = [(obs - np.min(obs, axis=1).reshape(2, 1)) / (
                        np.max(obs, axis=1).reshape(2, 1) - np.min(obs, axis=1).reshape(2, 1) + 1e-8) if obs is not None else None for obs in
                        tmp_obs]
            belief = [(obs / np.sum(obs, axis=1).reshape(2, 1)).reshape(-1) if obs is not None else None for obs in norm_obs]
            new_obs = {scheduler: belief[env.index_map[scheduler]] if belief[env.index_map[scheduler]] is not None else np.zeros((12,)) for scheduler in self.schedulers}
        else:
            new_obs = {k: v[0] for k, v in obss.items()}
        # ("real_obs:", {k: v[1] for k, v in obss.items()})
        return new_obs, new_rews, dones_, infos


class SuperObsEnv(MultiAgentEnv):
    def __init__(self, cfg):
        self.seed(2021)

        self.cfg = cfg
        self.raw_env = make_rllibenv(DelayedNetwork)(cfg, PartialAccessScenario)

        self.scenario = self.raw_env.scenario
        self.schedulers = self.raw_env.schedulers
        self.num_schedulers = cfg.n_schedulers
        self.action_spaces = self.raw_env.action_spaces
        self.init_schedulers = copy.deepcopy(self.schedulers)
        self.obs_server_id = self.scenario.obs_server_id

        # Set the observation spaces.
        num_obs_servers = cfg.num_obs_servers
        self.observation_spaces = {}
        agent_observation_space = self.raw_env.observation_spaces[self.schedulers[0]]
        for scheduler in self.schedulers:
            self.observation_spaces[scheduler] = Dict(
                {"self": agent_observation_space,
                 "self_id": Box(low=0, high=self.num_schedulers, shape=(1, ), dtype=np.int32),
                 "real_obs": agent_observation_space,
                 "others": Tuple((agent_observation_space,) * (self.num_schedulers - 1)),
                 # "others_real_obs": Tuple((agent_observation_space,) * (self.num_schedulers - 1)),
                 "other_ids": Box(low=0, high=self.num_schedulers, shape=(self.num_schedulers - 1, ), dtype=np.int32),
                 "obs_mask": Tuple(
                     (Tuple(
                         (Box(low=0, high=1, shape=(num_obs_servers, ), dtype=np.int32),)
                         * (self.num_schedulers - 1)),)
                     * cfg.num_obs_servers),
                 })

        # The other schedulers observed server id.
        self.others_obs_id = {}
        for scheduler in self.schedulers:
            self.others_obs_id[scheduler] = []
            others_id_app = self.others_obs_id[scheduler].append
            for other in self.schedulers:
                if other != scheduler:
                    others_id_app(self.obs_server_id[other])
        
        self.obs_mask = {scheduler: [[np.zeros((num_obs_servers,), dtype=np.int32)
                                      for _ in range(self.num_schedulers - 1)] for _ in range(cfg.num_obs_servers)]
                         for scheduler in self.schedulers}

        self.set_obs_mask()

    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs_dict: New observations for each ready agent.
        """
        obss = self.raw_env.reset()
        self.schedulers = self.raw_env.schedulers
        self.id_map = {scheduler: i for i, scheduler in enumerate(self.schedulers)}
        self.acc_drop_pkgs = self.raw_env.acc_drop_pkgs
        new_obss = {scheduler: {"self": np.asarray(obs, dtype=np.int32),
                                "self_id": np.asarray([self.id_map[scheduler]], dtype=np.int32),
                                "real_obs": np.asarray(obs, dtype=np.int32),
                                "others": tuple((np.zeros_like(obs, dtype=np.int32),)*(self.num_schedulers-1)),
                                # "others_real_obs": tuple((np.zeros_like(obs, dtype=np.int32),)*(self.num_schedulers-1)),
                                "other_ids": np.asarray([self.id_map[other] for other in self.schedulers if other != scheduler], dtype=np.int32),
                                "obs_mask": tuple(self.obs_mask[scheduler]),
                                } for scheduler, obs in obss.items()}

        return new_obss

    def step(self, actions):
        env = self.raw_env
        obss, rews, dones, infos = env.step(actions)
        total_r = sum(rews.values())
        new_rews = {k: total_r for k, _ in rews.items()}
        self.schedulers = env.schedulers
        dones_ = {k: dones[k] for k in self.schedulers}
        dones_["__all__"] = all(dones.values())
        infos = {k: {"done": dones[k]} for k in self.schedulers}
        new_obss = {}
        for scheduler, obs in obss.items():
            others = []
            others_app = others.append
            for other in self.init_schedulers:
                if other != scheduler:
                    try:
                        others_app(np.asarray(obss[other][0], dtype=np.int32))
                    except KeyError:
                        others_app(np.zeros_like(obs[0], dtype=np.int32))
            # Update the new observations.
            new_obss[scheduler] = {"self": np.asarray(obs[0], dtype=np.int32),
                                   "self_id": np.asarray([self.id_map[scheduler]], dtype=np.int32),
                                   "real_obs": np.asarray(obs[1], dtype=np.int32),
                                   "others": tuple(others),
                                   "other_ids": np.asarray([self.id_map[other] for other in self.init_schedulers if other != scheduler], dtype=np.int32),
                                   "obs_mask": tuple(self.obs_mask[scheduler]),}
        # ("real_obs:", {k: v[1] for k, v in obss.items()})
        return new_obss, new_rews, dones_, infos

    def set_obs_mask(self):
        for scheduler in self.schedulers:
            for j, s_id in enumerate(self.obs_server_id[scheduler]):
                for i, others in enumerate(self.others_obs_id[scheduler]):
                    if s_id not in others:
                        continue
                    self.obs_mask[scheduler][j][i][others.index(s_id)] = 1


def make_rllibenv(cls):
    class RLlibEnv(cls):
        """This environment deletes the done schedulers during episodes to keep compatible with RLlib."""

        def __init__(self, cfg, scene_cls):
            super().__init__(cfg, scene_cls)

        def step(self, actions):
            # Check whether the scheduler is done, if done then delete it.
            for scheduler in self.schedulers:
                self.dlt_agent(scheduler)
            return super().step(actions)

        def dlt_agent(self, scheduler):
            if self.dones[scheduler]:
                del self.dones[scheduler]
                del self.rewards[scheduler]
                del self.infos[scheduler]
                self.schedulers.remove(scheduler)

    return RLlibEnv
