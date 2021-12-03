from copy import deepcopy

import numpy as np
import ray
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import scipy.stats as st
import torch
import tqdm

from custom_env.environment import RLlibEnv, SuperObsEnv
from RLlib_custom_models import SuperObsModel
from logger import get_logger
from utils import DotDic, sigmoid

logger = get_logger(__name__, "PPO_test_data.log")


class RLlibAgent:
    # https://github.com/ray-project/ray/issues/9123
    def __init__(self, cfg, env):
        self.env = env
        self.cfg = cfg
        self.stop_criteria = {
            "training_iteration": cfg["training_iteration"],
        }

        # Initialize ray and trainer object
        ray.init(
            ignore_reinit_error=True,
            # local_mode=True,
            # log_to_driver=False
        )

    def set_config(self):
        cfg = self.cfg
        env = self.env
        # Gets default training configuration.
        config = deepcopy(get_trainer_class(cfg["alg_name"])._default_config)

        # === Settings for Rollout Worker processes ===
        # Use GPUs if `RLLIB_NUM_GPUS` env var set to > 0.
        # config["num_gpus"] = 1
        # int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        # config["num_gpus_per_worker"] = (1-0.0001)/3
        # Number of rollout worker actors to create for parallel sampling.
        config["num_workers"] = cfg["num_workers"]  # euler 20
        config["num_envs_per_worker"] = cfg["num_envs_per_worker"]

        # === Settings for the Trainer process ===
        # Whether layers should be shared for the value function.
        config["model"] = {
            "custom_model": "model",
            "custom_model_config": {
                "n_latents": cfg["n_latents"],
                "belief_hidden_dim": cfg["belief_hidden_dim"],
                "hidden_dim": cfg["PPO_hidden_dim"],
            },
        }
        config["framework"] = "torch"

        # === Environment Settings ===
        config["env"] = cfg["env_name"]
        # the env_creator function via the register env lambda below.
        # config["env_config"] = {"max_cycles": max_cycles, "num_agents": num_agents, "local_ratio": local_ratio}

        # # === Debug Settings ===
        # # Periodically print out summaries of relevant internal dataflow(DEBUG, INFO, WARN, or ERROR.)
        config["log_level"] = "INFO"
        config["no_done_at_end"] = True

        # === Settings for Multi-Agent Environments ===
        # Configuration for multi-agent setup with policy sharing:
        config["multiagent"] = {
            # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
            # of (policy_cls, obs_space, act_space, config). This defines the
            # observation and action spaces of the policies and any extra config.
            "policies": {
                "shared": (None, env.observation_spaces[agent], env.action_spaces[agent], {}) for agent in
                env.schedulers
            },
            # Function mapping agent ids to policy ids.
            "policy_mapping_fn": lambda agent_id: "shared",
        }
        return config

    def train(self):
        """
        Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
        :param stop_criteria: Dict with stopping criteria.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune"s ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        # Train
        analysis = ray.tune.run(
            self.cfg["alg_name"],
            stop=self.stop_criteria,
            config=self.set_config(),
            name=self.cfg["experiment_name"],
            checkpoint_at_end=True,
            local_dir=conf["local_dir"],
            checkpoint_freq=conf["checkpoint_freq"],
            # resume=True
            # restore="/content/drive/MyDrive/Data Science/pythonProject/masterthesis/ray_results/PPO/PPO_rllib_network-v0_e22d5_00000_0_2021-11-13_22-34-37/checkpoint_000350/checkpoint-350"
        )
        return analysis

    def load_exp_results(self, path):
        analysis = ray.tune.Analysis(path)
        return analysis

    def get_checkpoints_path(self, analysis):
        checkpoint_path = analysis.get_best_checkpoint(
            trial=analysis.get_best_logdir(metric="episode_reward_mean", mode="max"), metric="episode_reward_mean",
            mode="max")
        return checkpoint_path

    def load(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent"s saved checkpoint (only used for RLlib agents)
        """
        conf = self.cfg
        trainer_cls = get_trainer_class(conf["alg_name"])
        self.agent = trainer_cls(config=self.set_config(), env=conf["env_name"])
        if path:
            self.agent.restore(path)

    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        mean_episode_rewards, mean_episode_length, total_drp_pkg_rate, mean_comm_count = [], [], [], []
        std_episode_rewards, std_episode_length, std_drp_pkg_rate, std_comm_count = [], [], [], []
        # act_frequencies = args.act_frequency
        act_frequencies = range(6)
        for act_frequency in act_frequencies:
            episode_rewards, episode_length, episode_drp_pkg_rate, episode_comm_count = [], [], [], []
            self.cfg["act_frequency"] = act_frequency
            # instantiate env class
            env = RLlibEnv(DotDic(self.cfg))
            # run until episode ends
            num_e = 120
            for _ in tqdm.tqdm(range(num_e)):
                obss = env.reset()
                episode_reward, step, comm_count = 0, 0, 0
                dones = {"__all__": False}
                drop_pkg = {scheduler: 0 for scheduler in env.schedulers}
                while not dones["__all__"]:
                    step += 1
                    # logger.info(f"timestep: {step}")
                    actions = self.agent.compute_actions(obss, policy_id="shared")
                    logger.info(f"actions: {actions}")
                    for k, a in actions.items():
                        p = sigmoid(a[self.cfg["obs_servers"]:])
                        for _p in p:
                            if _p > 0.5:
                                comm_count += 1
                    obss, rewards, dones, infos = env.step(actions)
                    # logger.info(f"obs: {obss}")
                    # logger.info(f"reward: {reward}")
                    # logger.info("-"*40)
                    for k, v in rewards.items():
                        episode_reward += v
                for k, v in env.acc_drop_pkgs.items():
                    drop_pkg[k] += v
                episode_length.append(step)
                episode_rewards.append(episode_reward)
                episode_comm_count.append(comm_count / (self.cfg["n_schedulers"] * self.cfg["comm_act_dim"] * step))
                episode_drp_pkg_rate.append(sum(drop_pkg.values()) / (len(drop_pkg) * self.cfg["n_packages"]))
            mean_r = np.mean(episode_rewards)
            mean_p = np.mean(episode_drp_pkg_rate)
            mean_c = np.mean(episode_comm_count)
            mean_comm_count.append(mean_c)
            std_comm_count.append(
                st.t.interval(0.95, len(episode_comm_count) - 1, loc=mean_c, scale=st.sem(episode_comm_count))
            )
            mean_episode_rewards.append(mean_r)
            std_episode_rewards.append(
                st.t.interval(0.95, len(episode_rewards) - 1, loc=mean_r, scale=st.sem(episode_rewards)))
            total_drp_pkg_rate.append(mean_p)
            std_drp_pkg_rate.append(
                st.t.interval(0.95, len(episode_drp_pkg_rate) - 1, loc=mean_p, scale=st.sem(episode_drp_pkg_rate)))
            print(f"act_frequency: {act_frequency}")
            print(f"mean episode rewards: {mean_episode_rewards[-1]}, std episode rewards: {std_episode_rewards[-1]}")
            print(f"communication rate: {mean_comm_count[-1]}")
            print(
                f"total mean dropped packages rate: {total_drp_pkg_rate[-1]}, std_drp_pkg_rate: {std_drp_pkg_rate[-1]}")

        print("Summary:")
        logger.info(f"act_frequency{tuple([i for i in act_frequencies])}")
        logger.info(
            f"mean_episode_rewards: {tuple(mean_episode_rewards)}, std episode rewards: {tuple(std_episode_rewards)}")
        logger.info(f"total_drp_pkg_rate: {tuple(total_drp_pkg_rate)}, std_drp_pkg_rate: {tuple(std_drp_pkg_rate)}")
        logger.info(f"communication rate: {mean_comm_count}, std_comm_rate: {std_comm_count}")

    def shutdown(self):
        ray.shutdown()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", type=str, default=None,
                        help="gives the belief model's name [default: None]")
    parser.add_argument("-experiment_name", type=str, default=None,
                        help="gives this experiment's name [default: None]")
    parser.add_argument("--silent", action="store_true", default=False,
                        help="defines if scheduler can communicate [default: False]")
    parser.add_argument("--use_belief", action="store_true", default=False,
                        help="encodes observations to belief [default: False]")
    parser.add_argument("--test", action="store_true", default=False,
                        help="decide test model or train model [default: False]")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="enables CUDA training [default: False]")
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    with open("./config/PartialAccess.json", "r") as f:
        conf = json.loads(f.read())

    conf["use_belief"] = args.use_belief
    conf["silent"] = args.silent
    conf["experiment_name"] = args.experiment_name
    conf["belief_training"] = False
    if args.use_belief:
        assert args.model_name is not None, "If use belief model, the model name must be given."
        conf["model_name"] = args.model_name

    if args.test:
        conf["num_workers"] = 0
        conf["num_envs_per_worker"] = 1
    else:
        conf["num_workers"] = 2
        conf["num_envs_per_worker"] = 10

    # Create test environment.
    env = SuperObsEnv(DotDic(conf))
    # Register env
    register_env(conf["env_name"], lambda _: SuperObsEnv(DotDic(conf)))
    ModelCatalog.register_custom_model("model", SuperObsModel)
    ppo_agent = RLlibAgent(conf, env)

    if args.test:
        # analysis = ppo_agent.load_exp_results(f"./ray_results/{conf["experiment_name"]}")
        # path = ppo_agent.get_checkpoints_path(analysis)
        path = None
        # path = "/content/drive/MyDrive/Data Science/pythonProject/masterthesis/ray_results/PPO_noComm/PPO_rllib_network-v0_bda74_00000_0_2021-11-12_23-10-35/checkpoint_000350/checkpoint-350"
        # path = "/content/drive/MyDrive/Data Science/pythonProject/masterthesis/ray_results/PPO/PPO_rllib_network-v0_671d9_00000_0_2021-11-14_21-47-03/checkpoint_000700/checkpoint-700"
        # path = "/content/drive/MyDrive/Data Science/pythonProject/masterthesis/ray_results/PPO/PPO_rllib_network-v0_7690b_00000_0_2021-11-16_16-44-27/checkpoint_000350//checkpoint-350"
        ppo_agent.load(path)
        ppo_agent.test()
    else:
        ppo_agent.train()
    ppo_agent.shutdown()
