from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    SlimFC,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

from belief_models import AgentEncoder, AgentDecoder, reparametrize

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
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        n_latents,
        belief_hidden_dim,
        hidden_dim,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.n_agents = 10
        self.obs_space = obs_space.original_space
        obs_shape = self.obs_space['self'].shape
        ###########
        # action encoder
        self.encoder = AgentEncoder(n_latents=n_latents, num_embeddings=self.obs_space['self'].high[0]+1,
                                    hidden_dim=belief_hidden_dim)
        self.decoder = AgentDecoder(n_latents=n_latents, num_embeddings=self.obs_space['self'].high[0]+1,
                                    hidden_dim=belief_hidden_dim)
        ###########
        # NN main body
        self.shared = nn.Sequential(
            SlimFC(in_size=n_latents*obs_shape[0],
                   out_size=hidden_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn='tanh'),
            SlimFC(in_size=hidden_dim,
                   out_size=hidden_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn='tanh'),
            SlimFC(in_size=hidden_dim,
                   out_size=hidden_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn='tanh'),
        )
        ###########
        # Action NN head
        action_post_logits = [
            SlimFC(in_size=hidden_dim,
                   out_size=num_outputs,
                   initializer=normc_initializer(0.01),
                   activation_fn=None),
        ]
        self.action_output = nn.Sequential(*action_post_logits)
        ###########
        # Value NN head
        value_post_logits = [
            SlimFC(in_size=hidden_dim,
                   out_size=1,
                   initializer=normc_initializer(0.01),
                   activation_fn=None),
        ]
        self.value_output = nn.Sequential(*value_post_logits)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = restore_original_dimensions(input_dict["obs"], self.obs_space, "torch")
        batch_size = obs['self'].shape[0]
        device = obs['self'].device

        # action_feature_map = torch.zeros(
        #     batch_size, self.n_agents, self.encoder_out_features
        # ).to(device)
        # value_feature_map = torch.zeros(
        #     batch_size, self.n_agents, self.encoder_out_features
        # ).to(device)
        # action_feature_map[:, 0] = self.action_encoder(input_dict["obs"]['self'].float())
        # value_feature_map[:, 0] = self.value_encoder(input_dict["obs"]['self'].float())
        # input_others = input_dict["obs"]['others']
        # for i, other in enumerate(input_others):
        #     action_feature_map[:, i + 1] = self.action_encoder(other.float())
        #     value_feature_map[:, i + 1] = self.value_encoder(other.float())
        #
        # action_shared_features = self.action_shared(
        #     action_feature_map.view(batch_size, self.n_agents * self.encoder_out_features)
        # )
        # value_shared_features = self.value_shared(
        #     action_feature_map.view(batch_size, self.n_agents * self.encoder_out_features)
        # )
        _zs, _obs_recons, _mus, _logvars = [], [], [], []
        _zs_app, _obs_app, _mus_app, _logvars_app = _zs.append, _obs_recons.append, _mus.append, _logvars.append
        for i in range(obs['self'].shape[1]):
            _mu, _logvar = self.encoder(obs['self'][:, i])
            _z = reparametrize(_mu, _logvar, self.training)
            # _obs_app(self.decoder(_z))
            _zs_app(_z)
            # _mus_app(_mu)
            # _logvars_app(_logvar)
        z = torch.cat(_zs, dim=1)
        # mu = torch.cat(_mus, dim=1)
        # logvar = torch.cat(_logvars, dim=1)
        # obs_recon = torch.cat(_obs_recons, dim=1)
        shared_features = self.shared(torch.sigmoid(z))
        outputs = self.action_output(shared_features)
        values = self.value_output(shared_features).squeeze(1)

        self._cur_value = values

        return outputs, state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    # @override(ModelV2)
    # def custom_loss(self, policy_loss, loss_inputs):
    #     """Calculates a custom loss on top of the given policy_loss(es).
    #     Args:
    #         policy_loss (List[TensorType]): The list of already calculated
    #             policy losses (as many as there are optimizers).
    #         loss_inputs (TensorStruct): Struct of np.ndarrays holding the
    #             entire train batch.
    #     Returns:
    #         List[TensorType]: The altered list of policy losses. In case the
    #             custom loss should have its own optimizer, make sure the
    #             returned list is one larger than the incoming policy_loss list.
    #             In case you simply want to mix in the custom loss into the
    #             already calculated policy losses, return a list of altered
    #             policy losses (as done in this example below).
    #     """
    #     # Get the next batch from our input files.
    #     batch = self.reader.next()
    #
    #     # Define a secondary loss by building a graph copy with weight sharing.
    #     obs = restore_original_dimensions(
    #         torch.from_numpy(batch["obs"]).float().to(policy_loss[0].device),
    #         self.obs_space,
    #         tensorlib="torch")
    #     logits, _ = self.forward({"obs": obs}, [], None)
    #
    #     # You can also add self-supervised losses easily by referencing tensors
    #     # created during _build_layers_v2(). For example, an autoencoder-style
    #     # loss can be added as follows:
    #     # ae_loss = squared_diff(
    #     #     loss_inputs["obs"], Decoder(self.fcnet.last_layer))
    #     print("FYI: You can also use these tensors: {}, ".format(loss_inputs))
    #
    #     # Compute the IL loss.
    #     action_dist = TorchCategorical(logits, self.model_config)
    #     imitation_loss = torch.mean(-action_dist.logp(
    #         torch.from_numpy(batch["actions"]).to(policy_loss[0].device)))
    #     self.imitation_loss_metric = imitation_loss.item()
    #     self.policy_loss_metric = np.mean(
    #         [loss.item() for loss in policy_loss])
    #
    #     # Add the imitation loss to each already calculated policy loss term.
    #     # Alternatively (if custom loss has its own optimizer):
    #     # return policy_loss + [10 * self.imitation_loss]
    #     return [loss_ + 10 * imitation_loss for loss_ in policy_loss]
    #
    # def metrics(self):
    #     return {
    #         "policy_loss": self.policy_loss_metric,
    #         "imitation_loss": self.imitation_loss_metric,
    #     }
