from copy import deepcopy

import ray
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env
import torch

from custom_env.environment import RLlibEnv
from utils import DotDic


class RLlibAgent:
    # https://github.com/ray-project/ray/issues/9123
    def __init__(self, conf, env):
        self.env = env
        self.conf = conf
        self.stop_criteria = {
            'training_iteration': conf['training_iteration'],
        }

        # Initialize ray and trainer object
        ray.init(
            ignore_reinit_error=True,
            local_mode=True,
            # log_to_driver=False
        )

    def set_config(self):
        conf = self.conf
        # Gets default training configuration.
        config = deepcopy(get_trainer_class(conf['alg_name'])._default_config)

        # === Settings for Rollout Worker processes ===
        # Use GPUs if `RLLIB_NUM_GPUS` env var set to > 0.
        # config['num_gpus'] = 1
        # int(os.environ.get('RLLIB_NUM_GPUS', '0'))
        # config['num_gpus_per_worker'] = (1-0.0001)/3
        # Number of rollout worker actors to create for parallel sampling.
        config['num_workers'] = 2  # euler 20
        config['num_envs_per_worker'] = 10

        # config['train_batch_size'] = 600  # todo: increased.

        # === Settings for the Trainer process ===
        # Whether layers should be shared for the value function.
        config['model'] = {
            'fcnet_hiddens': [256, 256],  # todo: [256, 256]
            'fcnet_activation': 'tanh',  # todo: tanh
            # 'vf_share_layers': False,
            # 'use_lstm': True,
            # 'max_seq_len': 40,
            # 'lstm_use_prev_action': True,
            # 'lstm_use_prev_reward': True,
        }

        # === Environment Settings ===
        config['env'] = conf['env_name']
        # the env_creator function via the register env lambda below.
        # config['env_config'] = {'max_cycles': max_cycles, 'num_agents': num_agents, 'local_ratio': local_ratio}

        # # === Debug Settings ===
        # # Periodically print out summaries of relevant internal dataflow(DEBUG, INFO, WARN, or ERROR.)
        config['log_level'] = 'WARN'
        config['no_done_at_end'] = True

        # === Settings for Multi-Agent Environments ===
        # Configuration for multi-agent setup with policy sharing:
        config['multiagent'] = {
            # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
            # of (policy_cls, obs_space, act_space, config). This defines the
            # observation and action spaces of the policies and any extra config.
            'policies': {
                'shared': (None, env.observation_spaces[agent], env.action_spaces[agent], {}) for agent in
                env.schedulers
            },
            # Function mapping agent ids to policy ids.
            'policy_mapping_fn': lambda agent_id: 'shared',
        }
        return config

    def train(self):
        """
        Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
        :param stop_criteria: Dict with stopping criteria.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        # Train
        analysis = ray.tune.run(
            self.conf['alg_name'],
            stop=self.stop_criteria,
            config=self.set_config(),
            name=self.conf['experiment_name'],
            checkpoint_at_end=True,
            local_dir=conf['local_dir'],
            checkpoint_freq=conf['checkpoint_freq'],
            # restore='/content/drive/MyDrive/Data Science/pythonProject/masterthesis/ray_results/PPO_noComm/PPO_rllib_network-v0_c8e5f_00000_0_2021-11-02_00-49-50/checkpoint_002400/checkpoint-2400'
        )
        return analysis
    
    def load_exp_results(self, path):
        analysis = ray.tune.Analysis(path)
        return analysis

    def get_checkpoints_path(self, analysis):
        checkpoint_path = analysis.get_best_checkpoint(trial=analysis.get_best_logdir(metric='episode_reward_mean', mode='max'), metric='episode_reward_mean', mode='max')
        return checkpoint_path

    def load(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        trainer_cls = get_trainer_class(self.conf['alg_name'])
        self.agent = trainer_cls(config=self.set_config(), env=conf['env_name'])
        self.agent.restore(path)

    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        env = self.env

        # run until episode ends
        episode_reward, steps = 0, 0
        drop_pkg = {scheduler: 0 for scheduler in env.schedulers}
        for _ in range(60):
            step = 0
            done = {'__all__': False}
            obs = env.reset()
            while not done['__all__']:
                step += 1
                actions = self.agent.compute_actions(obs, policy_id='shared')
                # print('actions:',actions)
                obs, reward, done, info = env.step(actions)
                # print('timestep:', step)
                # print('obs:', obs)
                # print('reward:', reward)
                # print('-'*40)
                for k, v in reward.items():
                    episode_reward += v
            for k, v in env.acc_drop_pkgs.items():
                drop_pkg[k] += v
            steps += step 
        print(f'mean episode rewards: {episode_reward/60:.2f}')
        print(f'mean episode length: {steps/60:.2f}')
        print('mean dropped packages rate:', {k: round(v / (60 * self.conf['n_packages']), 2) for k, v in drop_pkg.items()})

        return episode_reward

    def shutdown(self):
        ray.shutdown()


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, default=None,
                        help="gives the belief model's name [default: None]")
    parser.add_argument('-experiment_name', type=str, default=None,
                        help="gives this experiment's name [default: None]")
    parser.add_argument('--silent', action='store_true', default=False,
                        help='defines if scheduler can communicate [default: False]')
    parser.add_argument('--use_belief', action='store_true', default=False,
                        help='encodes observations to belief [default: False]')
    parser.add_argument('--test', action='store_true', default=False,
                        help='decide test model or train model [default: False]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    with open('./config/PartialAccess.json', 'r') as f:
        conf = json.loads(f.read())
    
    conf['use_belief'] = args.use_belief
    conf['silent'] = args.silent
    conf['experiment_name'] = args.experiment_name
    conf['belief_training'] = False
    if args.use_belief:
        assert args.model_name is not None, 'If use belief model, the model name must be given.'
        conf['model_name'] = args.model_name

    # Create test environment.
    env = RLlibEnv(DotDic(conf))
    # Register env
    register_env(conf['env_name'], lambda _: RLlibEnv(DotDic(conf)))
    ppo_agent = RLlibAgent(conf, env)

    if args.test:
        analysis = ppo_agent.load_exp_results(f'./ray_results/{conf["experiment_name"]}')
        # path = ppo_agent.get_checkpoints_path(analysis)
        path = '/content/drive/MyDrive/Data Science/pythonProject/masterthesis/ray_results/PPO_noComm/PPO_rllib_network-v0_12944_00000_0_2021-11-09_02-02-20/checkpoint_000300/checkpoint-300'
        # path = '/content/drive/MyDrive/Data Science/pythonProject/masterthesis/ray_results/PPO/PPO_rllib_network-v0_38f7a_00000_0_2021-10-31_16-25-55/checkpoint_000350/checkpoint-350'
        # path = '/content/drive/MyDrive/Data Science/pythonProject/masterthesis/ray_results/PPO/PPO_rllib_network-v0_7a832_00000_0_2021-11-09_09-29-03/checkpoint_000150/checkpoint-150'
        ppo_agent.load(path)
        ppo_agent.test()
    else:
        ppo_agent.train()
    ppo_agent.shutdown()
