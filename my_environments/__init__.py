from gym.envs.registration import register

register(
    id='q_network-v0',
    entry_point='my_environments.environment:QueueingNetwork'
)
