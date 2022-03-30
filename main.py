from copy import deepcopy
import time
import collections

import numpy as np
import ray
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import scipy.stats as st
from scipy.special import softmax
import tqdm

from custom_env.environment import SuperObsEnv, NormalEnv
from custom_loss import PPOTrainer
from custom_models import (CommNetModel, BicNetModel, ATVCModel)
from logger import get_logger
from utils import DotDic
from policies import JSQHeuristic


class RLlibAgent:
    # https://github.com/ray-project/ray/issues/9123
    def __init__(self, cfg, env):
        self.env = env
        self.cfg = cfg
        self.env_cls = cfg["env_cls"]
        self.alg_name = cfg["alg_name"] if not cfg["customize"] else PPOTrainer
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
        rllib_cfg = deepcopy(get_trainer_class(cfg["alg_name"])._default_config)

        # === Settings for Rollout Worker processes ===
        # Use GPUs if `RLLIB_NUM_GPUS` env var set to > 0.
        # rllib_cfg["num_gpus"] = 1
        # int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        # rllib_cfg["num_gpus_per_worker"] = (1-0.0001)/3
        # Number of rollout worker actors to create for parallel sampling.
        rllib_cfg["num_workers"] = cfg["num_workers"]
        rllib_cfg["num_envs_per_worker"] = cfg["num_envs_per_worker"]

        # === Settings for the Trainer process ===
        if cfg["customize"]:
            rllib_cfg["model"] = {
                "custom_model": "model",
                "custom_model_config": {
                    "silent": cfg["silent"],
                    "attention": cfg["attention"],
                    "n_latents": cfg["n_latents"],
                    "hidden_dim": cfg["PPO_hidden_dim"],
                },
            }
        else:
            rllib_cfg['model'] = {
                'fcnet_hiddens': [cfg["PPO_hidden_dim"], cfg["PPO_hidden_dim"]],
            }

        rllib_cfg["framework"] = "torch"

        # === Environment Settings ===
        rllib_cfg["env"] = cfg["env_name"]
        # # === Debug Settings ===
        # # Periodically print out summaries of relevant internal dataflow(DEBUG, INFO, WARN, or ERROR.)
        rllib_cfg["log_level"] = "WARN"
        rllib_cfg["no_done_at_end"] = True

        # === Settings for Multi-Agent Environments ===
        # Configuration for multi-agent setup with policy sharing:
        test_agent = env.schedulers[0]
        observation_space = env.observation_spaces[test_agent]
        action_space = env.action_spaces[test_agent]
        policies = {"shared": PolicySpec(observation_space=observation_space, action_space=action_space)}
        if cfg["JSQ"]:
          policies["JSQ"] = PolicySpec(policy_class=JSQHeuristic, observation_space=observation_space, action_space=action_space)

        def select_policy(agent_id, episode, **kwargs):
            if cfg["JSQ"]:
                return "JSQ"
            else:
                return "shared"

        rllib_cfg["multiagent"] = {
            "policies_to_train": ["shared"],
            "policies": policies,
            # Function mapping agent ids to policy ids.
            "policy_mapping_fn": select_policy,
        }
        return rllib_cfg

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
            self.alg_name,
            stop=self.stop_criteria,
            config=self.set_config(),
            name=self.cfg["experiment_name"],
            checkpoint_at_end=True,
            local_dir=cfg["local_dir"],
            checkpoint_freq=cfg["checkpoint_freq"],
            # resume=True
            # restore="/content/drive/MyDrive/DataScience/pythonProject/masterthesis/ray_results/PPO_1e-6autoencoder1e-4_att_z_6agents/CustomPPO_rllib_network-v0_b22da_00000_0_2022-03-22_02-42-20/checkpoint_000450/checkpoint-450"
        )
        return analysis

    def load(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent"s saved checkpoint (only used for RLlib agents)
        """
        trainer_cls = get_trainer_class(cfg["alg_name"])
        self.agent = trainer_cls(config=self.set_config(), env=cfg["env_name"])
        if path:
            self.agent.restore(path)

    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        mean_episode_rewards, mean_episode_length, total_drp_pkg_rate, mean_comm_count = [], [], [], []
        std_episode_rewards, std_episode_length, std_drp_pkg_rate, std_comm_count = [], [], [], []
        act_frequencies = range(1, 18)
        for act_syncT in act_frequencies:
            episode_rewards, episode_length, episode_drp_pkg_rate, episode_comm_count = [], [], [], []
            self.cfg["act_syncT"] = act_syncT
            # instantiate env class
            env = self.env_cls(DotDic(self.cfg))
            # intention map
            intention_map = np.zeros((6, 6))
            counter = np.zeros((6, 6))
            # run until episode ends
            num_e = 1000
            for _ in tqdm.tqdm(range(num_e)):
                obss = env.reset()
                infos = collections.defaultdict(dict)
                infos["scheduler_1"]["obs"] = [0, 0]
                episode_reward, step, comm_count = 0, 0, 0
                dones = {'__all__': False}
                drop_pkg = {scheduler: 0 for scheduler in env.schedulers}
                while not dones['__all__']:
                    step += 1
                    # logger.info(f'timestep: {step}')
                    actions = self.agent.compute_actions(obss, policy_id=cfg["policy"])
                    if cfg["policy"] == "JSQ":
                      actions = {k: v.numpy() for k, v in actions.items()}

                    if "scheduler_1" in obss:
                        o = infos["scheduler_1"]["obs"]
                        p = softmax(actions["scheduler_1"])[0]
                        if p >= 0.5:
                            intention_map[o[0], o[1]] += 1
                        counter[o[0], o[1]] += 1

                    obss, rewards, dones, infos = env.step(actions)
                    logger.info(f'obs: {obss}')
                    logger.info(f'infos: {infos}')
                    logger.info('-'*40)
                    episode_reward += list(rewards.values())[0]
                for k, v in env.acc_drop_pkgs.items():
                    drop_pkg[k] += v
                episode_length.append(step)
                episode_rewards.append(episode_reward)
                episode_drp_pkg_rate.append(sum(drop_pkg.values()) / (len(drop_pkg) * self.cfg['n_packages']))
            mean_r = np.mean(episode_rewards)
            mean_p = np.mean(episode_drp_pkg_rate)
            mean_episode_rewards.append(mean_r)
            std_episode_rewards.append(
                st.t.interval(0.95, len(episode_rewards) - 1, loc=mean_r, scale=st.sem(episode_rewards)))
            total_drp_pkg_rate.append(mean_p)
            std_drp_pkg_rate.append(
                st.t.interval(0.95, len(episode_drp_pkg_rate) - 1, loc=mean_p, scale=st.sem(episode_drp_pkg_rate)))
            print(f'act_syncT: {act_syncT}')
            print(f'mean episode rewards: {mean_episode_rewards[-1]}, std episode rewards: {std_episode_rewards[-1]}')
            print(
                f'total mean dropped packages rate: {total_drp_pkg_rate[-1]}, std_drp_pkg_rate: {std_drp_pkg_rate[-1]}')

        intention_map /= (counter + 1e-8)
        print('Summary:')
        logger.info(f'act_syncT{tuple([i for i in act_frequencies])}')
        logger.info(
            f'mean_episode_rewards: {tuple(mean_episode_rewards)}, std episode rewards: {tuple(std_episode_rewards)}')
        logger.info(f'total_drp_pkg_rate: {tuple(total_drp_pkg_rate)}, std_drp_pkg_rate: {tuple(std_drp_pkg_rate)}')
        logger.info(f'intention map:{intention_map}')
        logger.info(f'counts for each combination {counter}')

    def shutdown(self):
        ray.shutdown()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("-experiment_name", type=str, default=None,
                        help="gives this experiment's name [default: None]")
    parser.add_argument("--silent", action="store_true", default=False,
                        help="defines if scheduler can communicate [default: False]")
    parser.add_argument("--default", action="store_true", default=False,
                        help="defines if scheduler uses cutomized models [default: False]")
    parser.add_argument("--test", action="store_true", default=False,
                        help="decide test model or train model [default: False]")
    parser.add_argument("--JSQ", action="store_true", default=False,
                        help="decide use JSQ [default: False]")
    parser.add_argument("--true_obs", action="store_true", default=False,
                        help="decide use true observation [default: False]")
    parser.add_argument("--opposite", action="store_true", default=False,
                        help="decide use delayed observation or opposite [default: False]")
    args = parser.parse_args()

    with open("./config/PartialAccess.json", "r") as f:
        cfg = json.loads(f.read())

    cfg["silent"] = args.silent
    cfg["JSQ"] = args.JSQ
    cfg["true_obs"] = args.true_obs
    cfg["opposite"] = args.opposite
    cfg["customize"] = not args.default
    if args.experiment_name:
        cfg["experiment_name"] = args.experiment_name
    if args.test:
        logger = get_logger(__name__, f"PPO_test_{args.experiment_name}_{str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))}.log")
        cfg["num_workers"] = 0
        cfg["num_envs_per_worker"] = 1
    else:
        cfg["num_workers"] = 2
        cfg["num_envs_per_worker"] = 10

    # Create test environment.
    if cfg["customize"]:
        Env = SuperObsEnv
        if args.experiment_name[:4] == "ATVC":
            print("Using ATVC")
            ModelCatalog.register_custom_model("model", ATVCModel)
        elif args.experiment_name[:4] == "BiCN":
            print("Using BiCNet")
            ModelCatalog.register_custom_model("model", BicNetModel)
        elif args.experiment_name[:4] == "Comm":
            print("Using CommNet")
            ModelCatalog.register_custom_model("model", CommNetModel)
    else:
        print("Using 2-layer MLP")
        Env = NormalEnv
    cfg["env_cls"] = Env
    test_env = Env(DotDic(cfg))
    # Register env
    register_env(cfg["env_name"], lambda _: Env(DotDic(cfg)))
    ppo_agent = RLlibAgent(cfg, test_env)

    if args.test:
        path = None
        cfg["policy"] = "JSQ"
        if not args.JSQ:
            cfg["policy"] = "shared"
            # path = "/content/drive/MyDrive/DataScience/pythonProject/masterthesis/ray_results/PPO_1e-6autoencoder1e-4_att_3agents_opposite/CustomPPO_rllib_network-v0_e6536_00000_0_2022-03-26_00-51-41/checkpoint_000300/checkpoint-300"
            path = "/content/drive/MyDrive/DataScience/pythonProject/masterthesis/ray_results/PPO_same_r_3agents_delayed/PPO_rllib_network-v0_94ed0_00000_0_2022-03-29_01-57-26/checkpoint_000300/checkpoint-300"
        ppo_agent.load(path)
        ppo_agent.test()
    else:
        ppo_agent.train()
    ppo_agent.shutdown()
