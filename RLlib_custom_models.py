import numpy as np
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import (
    normc_initializer,
    SlimFC,
)
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils import try_import_torch

from model_components import (
    MessageEncoder,
    MessageDecoder,
    reparametrize,
    RNNEncoder,
    LinearLayers,
    prior_expert,
    ProductOfExperts
)

torch, nn = try_import_torch()


# class Model(TorchModelV2, nn.Module):
#     def __init__(
#         self,
#         obs_space,
#         action_space,
#         num_outputs,
#         model_config,
#         name,
#         encoder_out_features,
#         shared_nn_out_features_per_agent,
#         value_state_encoder_cnn_out_features,
#         share_observations,
#     ):
#         TorchModelV2.__init__(
#             self, obs_space, action_space, num_outputs, model_config, name
#         )
#         nn.Module.__init__(self)
#
#         self.encoder_out_features = encoder_out_features
#         self.shared_nn_out_features_per_agent = shared_nn_out_features_per_agent
#         self.value_state_encoder_cnn_out_features = value_state_encoder_cnn_out_features
#         self.share_observations = share_observations
#
#         self.n_agents = len(obs_space.original_space)
#         self.outputs_per_agent = int(num_outputs / self.n_agents)
#
#         obs_shape = obs_space.original_space[0].shape
#         ###########
#         # NN main body
#         self.action_encoder = nn.Sequential(
#             SlimFC(in_size=obs_shape[0],
#                    out_size=128,
#                    initializer=normc_initializer(1.0),
#                    activation_fn='swish'),
#             SlimFC(in_size=128,
#                    out_size=128,
#                    initializer=normc_initializer(1.0),
#                    activation_fn='swish'),
#             SlimFC(in_size=128,
#                    out_size=2*self.encoder_out_features,
#                    initializer=normc_initializer(1.0),
#                    activation_fn=None),
#         )
#
#         self.value_encoder = copy.deepcopy(self.action_encoder)
#
#         share_n_agents = self.n_agents if self.share_observations else 1
#         self.action_shared = nn.Sequential(
#             SlimFC(in_size=self.encoder_out_features * share_n_agents,
#                    out_size=hidden_dim,
#                    initializer=normc_initializer(1.0),
#                    activation_fn='tanh'),
#             SlimFC(in_size=hidden_dim,
#                    out_size=hidden_dim,
#                    initializer=normc_initializer(1.0),
#                    activation_fn='tanh'),
#             SlimFC(in_size=hidden_dim,
#                    out_size=hidden_dim,
#                    initializer=normc_initializer(1.0),
#                    activation_fn='tanh'),
#             SlimFC(in_size=hidden_dim,
#                    out_size=self.shared_nn_out_features_per_agent * share_n_agents,
#                    initializer=normc_initializer(1.0),
#                    activation_fn='tanh'),
#         )
#         self.value_shared = copy.deepcopy(self.action_shared)
#         ###########
#         # Action NN head
#         action_post_logits = [
#             SlimFC(in_size=self.shared_nn_out_features_per_agent,
#                    out_size=128,
#                    initializer=normc_initializer(1.0),
#                    activation_fn='tanh'),
#             SlimFC(in_size=128,
#                    out_size=self.outputs_per_agent,
#                    initializer=normc_initializer(1.0),
#                    activation_fn=None),
#         ]
#         self.action_output = nn.Sequential(*action_post_logits)
#         ###########
#         # Value NN head
#         value_post_logits = [
#             SlimFC(in_size=self.shared_nn_out_features_per_agent,
#                    out_size=128,
#                    initializer=normc_initializer(1.0),
#                    activation_fn='tanh'),
#             SlimFC(in_size=128,
#                    out_size=1,
#                    initializer=normc_initializer(1.0),
#                    activation_fn=None),
#         ]
#         self.value_output = nn.Sequential(*value_post_logits)
#
#     @override(ModelV2)
#     def forward(self, input_dict, state, seq_lens):
#         batch_size = input_dict["obs"][0].shape[0]
#         device = input_dict["obs"][0].device
#
#         action_feature_map = torch.zeros(
#             batch_size, self.n_agents, self.encoder_out_features
#         ).to(device)
#         value_feature_map = torch.zeros(
#             batch_size, self.n_agents, self.encoder_out_features
#         ).to(device)
#         for i in range(self.n_agents):
#             agent_obs = input_dict["obs"][i]
#             a_mu_logvar = self.action_encoder(agent_obs)
#             a_mu = a_mu_logvar[:, :self.encoder_out_features]
#             a_logvar = a_mu_logvar[:, self.encoder_out_features:]
#             action_feature_map[:, i] = torch.sigmoid(self.reparametrize(a_mu, a_logvar))
#             v_mu_logvar = self.value_encoder(agent_obs)
#             v_mu = v_mu_logvar[:, :self.encoder_out_features]
#             v_logvar = v_mu_logvar[:, self.encoder_out_features:]
#             value_feature_map[:, i] = torch.sigmoid(self.reparametrize(v_mu, v_logvar))
#         # todo:PoE implementation.
#         if self.share_observations:
#             # We have a big common shared center NN so that all agents have access to the encoded observations of all agents
#             action_shared_features = self.action_shared(
#                 action_feature_map.view(
#                     batch_size, self.n_agents * self.encoder_out_features
#                 )
#             ).view(batch_size, self.n_agents, self.shared_nn_out_features_per_agent)
#             value_shared_features = self.value_shared(
#                 value_feature_map.view(
#                     batch_size, self.n_agents * self.encoder_out_features
#                 )
#             ).view(batch_size, self.n_agents, self.shared_nn_out_features_per_agent)
#         else:
#             # Each agent only has access to its own local observation
#             shared_features = torch.empty(
#                 batch_size, self.n_agents, self.shared_nn_out_features_per_agent
#             ).to(device)
#             for i in range(self.n_agents):
#                 shared_features[:, i] = self.shared(feature_map[:, i])
#
#         outputs = torch.empty(batch_size, self.n_agents, self.outputs_per_agent).to(
#             device
#         )
#         values = torch.empty(batch_size, self.n_agents).to(device)
#
#         for i in range(self.n_agents):
#             outputs[:, i] = self.action_output(shared_features[:, i])
#             values[:, i] = self.value_output(shared_features[:, i]).squeeze(1)
#
#         self._cur_value = values
#
#         return outputs.view(batch_size, self.n_agents * self.outputs_per_agent), state
#
#     @override(ModelV2)
#     def value_function(self):
#         assert self._cur_value is not None, "must call forward() first"
#         return self._cur_value
#
#     def reparametrize(self, mu, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = std.data.new(std.size()).normal_()
#             return eps.mul(std).add_(mu)
#         else:  # return mean during inference
#             return mu


# class SimpleModel(TorchModelV2, nn.Module):
#     def __init__(
#             self,
#             obs_space,
#             action_space,
#             num_outputs,
#             model_config,
#             name,
#             n_latents,
#             hidden_dim,
#     ):
#         TorchModelV2.__init__(
#             self, obs_space, action_space, num_outputs, model_config, name
#         )
#         nn.Module.__init__(self)
#
#         self.n_agents = len(obs_space.original_space)
#         self.outputs_per_agent = int(num_outputs / self.n_agents)
#
#         self.obs_shape = obs_space.original_space[0].shape
#         obs_shape = self.obs_shape
#         ###########
#         # NN main body
#         share_n_agents = self.n_agents if self.share_observations else 1
#         self.action_shared = nn.Sequential(
#             SlimFC(in_size=obs_shape[0] * share_n_agents,
#                    out_size=hidden_dim,
#                    initializer=normc_initializer(1.0),
#                    activation_fn='tanh'),
#             SlimFC(in_size=hidden_dim,
#                    out_size=hidden_dim,
#                    initializer=normc_initializer(1.0),
#                    activation_fn='tanh'),
#             SlimFC(in_size=hidden_dim,
#                    out_size=hidden_dim*share_n_agents,
#                    initializer=normc_initializer(1.0),
#                    activation_fn='tanh'),
#         )
#
#         self.value_shared = copy.deepcopy(self.action_shared)
#         ###########
#         # Action NN head
#         action_post_logits = [
#             SlimFC(in_size=hidden_dim,
#                    out_size=self.outputs_per_agent,
#                    initializer=normc_initializer(0.01),
#                    activation_fn=None),
#         ]
#         self.action_output = nn.Sequential(*action_post_logits)
#         ###########
#         # Value NN head
#         value_post_logits = [
#             SlimFC(in_size=hidden_dim,
#                    out_size=1,
#                    initializer=normc_initializer(0.01),
#                    activation_fn=None),
#         ]
#         self.value_output = nn.Sequential(*value_post_logits)
#
#     @override(ModelV2)
#     def forward(self, input_dict, state, seq_lens):
#         batch_size = input_dict["obs"][0].shape[0]
#         device = input_dict["obs"][0].device
#
#         feature_map = torch.zeros(
#             batch_size, self.n_agents, self.obs_shape[0]
#         ).to(device)
#         for i in range(self.n_agents):
#             feature_map[:, i] = input_dict["obs"][i]
#
#         if self.share_observations:
#             # We have a big common shared center NN so that all agents have access to the encoded observations of all agents
#             action_shared_features = self.action_shared(
#                 feature_map.view(
#                     batch_size, self.n_agents * self.obs_shape[0]
#                 )
#             ).view(batch_size, self.n_agents, hidden_dim)
#             value_shared_features = self.value_shared(
#                 feature_map.view(
#                     batch_size, self.n_agents * self.obs_shape[0]
#                 )
#             ).view(batch_size, self.n_agents, hidden_dim)
#         else:
#             # Each agent only has access to its own local observation
#             action_shared_features = torch.empty(
#                 batch_size, self.n_agents, hidden_dim
#             ).to(device)
#             value_shared_features = torch.empty(
#                 batch_size, self.n_agents, hidden_dim
#             ).to(device)
#             for i in range(self.n_agents):
#                 action_shared_features[:, i] = self.action_shared(feature_map[:, i])
#                 value_shared_features[:, i] = self.value_shared(feature_map[:, i])
#
#         outputs = torch.empty(batch_size, self.n_agents, self.outputs_per_agent).to(
#             device
#         )
#         values = torch.empty(batch_size, self.n_agents).to(device)
#
#         for i in range(self.n_agents):
#             outputs[:, i] = self.action_output(action_shared_features[:, i])
#             values[:, i] = self.value_output(value_shared_features[:, i]).squeeze(1)
#
#         self._cur_value = values
#
#         return outputs.view(batch_size, self.n_agents * self.outputs_per_agent), state
#
#     @override(ModelV2)
#     def value_function(self):
#         assert self._cur_value is not None, "must call forward() first"
#         return self._cur_value


class SuperObsModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 n_latents,
                 hidden_dim):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self.n_latents = n_latents
        self.obs_space = obs_space.original_space
        obs_shape = self.obs_space['self'].shape
        encoding_dim = hidden_dim * 2
        # observation encoder
        self.obs_encoder = nn.Sequential(
            SlimFC(in_size=obs_shape[0],
                   out_size=hidden_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn='relu'),
            SlimFC(in_size=hidden_dim,
                   out_size=hidden_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn='relu'),
            SlimFC(in_size=hidden_dim,
                   out_size=encoding_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn='relu'),
        )
        # message distribution
        self.msg_dist = nn.Sequential(
            SlimFC(in_size=encoding_dim,
                   out_size=hidden_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn='relu'),
            SlimFC(in_size=hidden_dim,
                   out_size=hidden_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn='relu'),
            SlimFC(in_size=hidden_dim,
                   out_size=2 * n_latents,
                   initializer=normc_initializer(1.0),
                   activation_fn=None),
        )
        # communication method
        self.PoE = ProductOfExperts()
        # NN main body
        self.hidden_layers = LinearLayers(input_dim=(n_latents+encoding_dim), hidden_dim=hidden_dim)
        self.action_branch = nn.Sequential(
            SlimFC(in_size=hidden_dim,
                   out_size=num_outputs,
                   initializer=normc_initializer(0.01),
                   activation_fn=None))
        self.value_branch = nn.Sequential(
            SlimFC(in_size=hidden_dim,
                   out_size=1,
                   initializer=normc_initializer(0.01),
                   activation_fn=None))
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        device = obs['self'].device
        shape = obs['self'].shape

        # obs = restore_original_dimensions(input_dict['obs'], self.obs_space, "torch")
        # self observation encoding
        me = self.obs_encoder(obs['self'])
        # others' messages encoding and concat.
        mu, logvar = prior_expert((shape[0], 1, self.n_latents))
        others_obs = torch.cat([o.unsqueeze(2).permute(0, 2, 1) for o in obs['others']], dim=1)
        obs_encoding = self.obs_encoder(others_obs)
        m_dist = self.msg_dist(obs_encoding)
        self.m_mu = m_dist[:, :, :self.n_latents]
        self.m_logvar = m_dist[:, :, self.n_latents:]
        mus = torch.cat((mu, self.m_mu), dim=1)
        logvars = torch.cat((logvar, self.m_logvar), dim=1)
        msg_mu, msg_logvar = self.PoE(mus, logvars, dim=1)
        msg_z = reparametrize(msg_mu, msg_logvar)
        total = torch.cat((me, msg_z), dim=1)
        self._features = self.hidden_layers(total)
        action_out = self.action_branch(self._features)
        return action_out, state


class SuperObsRNNModel(TorchRNN, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 n_latents,
                 hidden_dim):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self.obs_space = obs_space.original_space
        obs_shape = self.obs_space['self'].shape
        self.gru_state_size = n_latents * obs_shape[0]
        # observation encoder
        self.fc1 = nn.Linear(obs_shape[0], hidden_dim)
        self.gru = nn.GRU(hidden_dim, self.gru_state_size, batch_first=True)
        # message distribution
        self.msg_dist = nn.Sequential(
            SlimFC(in_size=self.gru_state_size,
                   out_size=hidden_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn='relu'),
            SlimFC(in_size=hidden_dim,
                   out_size=hidden_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn='relu'),
            SlimFC(in_size=hidden_dim,
                   out_size=2 * self.gru_state_size,
                   initializer=normc_initializer(1.0),
                   activation_fn=None),
        )
        # communication method
        self.PoE = ProductOfExperts()
        # NN main body
        self.hidden_layers = LinearLayers(input_dim=self.gru_state_size, hidden_dim=hidden_dim)
        self.action_branch = nn.Sequential(
            SlimFC(in_size=hidden_dim,
                   out_size=num_outputs,
                   initializer=normc_initializer(0.01),
                   activation_fn=None))
        self.value_branch = nn.Sequential(
            SlimFC(in_size=hidden_dim,
                   out_size=1,
                   initializer=normc_initializer(0.01),
                   activation_fn=None))
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [self.fc1.weight.new(1, self.gru_state_size).zero_().squeeze(0)]*10
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        device = inputs.device
        shape = inputs.shape
        obs = restore_original_dimensions(inputs, self.obs_space, "torch")
        # self observation encoding
        me = torch.relu(self.fc1(obs['self']))
        gru_out_me, h_me = self.gru(me, torch.unsqueeze(state[0], 0))
        # others' messages encoding and concat.
        mu, logvar = prior_expert((shape[0], shape[1], self.gru_state_size, 1))
        msg_mus, msg_logvars = [mu.to(device)], [logvar.to(device)]
        states = [torch.squeeze(h_me, 0)]
        states_app, msg_mu_app, msg_logvar_app = states.append, msg_mus.append, msg_logvars.append
        for i, other in enumerate(obs['others']):
            f = torch.relu(self.fc1(other))
            gru_o, h_o = self.gru(f, torch.unsqueeze(state[i+1], 0))
            dist = self.msg_dist(gru_o).unsqueeze(3)
            msg_mu_app(dist[:, :, :self.gru_state_size, :])
            msg_logvar_app(dist[:, :, self.gru_state_size:, :])
            states_app(torch.squeeze(h_o, 0))
        cat_mu = torch.cat(msg_mus, dim=-1)
        cat_logvar = torch.cat(msg_logvars, dim=-1)
        msg_mu, msg_logvar = self.PoE(cat_mu, cat_logvar)
        msg_z = reparametrize(msg_mu, msg_logvar)
        self._features = self.hidden_layers(0.5*msg_z+0.5*gru_out_me)
        action_out = self.action_branch(self._features)
        return action_out, states
