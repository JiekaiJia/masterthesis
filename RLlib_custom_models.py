import copy

import numpy as np
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    SlimFC,
)
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils import try_import_torch

from model_components import (
    reparametrize,
    MLP,
    prior_expert,
    ProductOfExperts
)

torch, nn = try_import_torch()


class MaskPoEModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 silent,
                 n_latents,
                 hidden_dim,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        self.silent = silent
        self.n_latents = n_latents
        self.obs_space = obs_space.original_space
        self.reconstruct_loss = nn.CrossEntropyLoss(reduction="none")
        obs_shape = self.obs_space["self"].shape
        # observation encoder.
        self.encoder = MLP([obs_shape[0], hidden_dim, 2 * n_latents], last_activation=False)
        # communication method
        self.PoE = ProductOfExperts()
        # NN main body
        self.hidden_layers = MLP([n_latents, hidden_dim, hidden_dim])
        self.decoder_head = MLP([hidden_dim, 6 * obs_shape[0]], last_activation=False)
        self.value_layers = MLP([2 * n_latents, hidden_dim, hidden_dim])
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
        # Holds the current 'base' output (before logits layer).
        self.value_features = None

    @override(ModelV2)
    def value_function(self):
        assert self.value_features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self.value_features), [-1])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        shape = obs["self"].shape
        batch_size = shape[0]
        obs_len = shape[1]
        obs_mask = obs["obs_mask"]

        me = self.encoder(obs["self"])
        real_me = self.encoder(obs["real_obs"])
        me_mu = me[:, :self.n_latents]
        me_logvar = me[:, self.n_latents:]
        self.final_mu, self.final_logvar = me_mu, me_logvar
        if not self.silent:
            prior_mu, prior_logvar = prior_expert((shape[0], 1, self.n_latents))
            others_obs = torch.cat([o.unsqueeze(1) for o in obs["others"]], dim=1)
            digits_masks = [torch.cat([m.unsqueeze(1) for m in mask], dim=1) for mask in obs_mask]
            masked_obs_o = [others_obs * mask for mask in digits_masks]
            codes = [self.encoder(o) for o in masked_obs_o]
            msg = [self.PoE(torch.cat([prior_mu, code[:, :, :self.n_latents]], dim=1), torch.cat([prior_logvar, code[:, :, self.n_latents:]], dim=1), dim=1)
                   for code in codes]
            sprune_msg = [self.PoE(mu.view(batch_size, obs_len, -1), logvar.view(batch_size, obs_len, -1), dim=1) for mu, logvar in msg]
            sprune_mus, sprune_logvars = [], []
            sprune_mus_app, sprune_logvars_app = sprune_mus.append, sprune_logvars.append
            for sprune_mu, sprune_logvar in sprune_msg:
              sprune_mus_app(sprune_mu)
              sprune_logvars_app(sprune_logvar)
            sprune_mu_tensor = torch.cat(sprune_mus, dim=1)
            sprune_logvar_tensor = torch.cat(sprune_logvars, dim=1)
            pd_mu, pd_logvar = self.PoE(torch.cat([sprune_mu_tensor.unsqueeze(1), me_mu.unsqueeze(1)], dim=1), torch.cat([sprune_logvar_tensor.unsqueeze(1), me_logvar.unsqueeze(1)], dim=1), dim=1)
            self.final_mu, self.final_logvar = pd_mu, pd_logvar
        z = reparametrize(self.final_mu, self.final_logvar)
        self.action_features = self.hidden_layers(z)
        self.logits = self.decoder_head(self.action_features).view(batch_size, obs_len, -1)
        self.value_features = self.value_layers(torch.cat([real_me[:, :self.n_latents], z], dim=1))
        action_out = self.action_branch(self.action_features)
        return action_out, state


class AttentionPoEModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 silent,
                 n_latents,
                 hidden_dim,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        self.silent = silent
        self.n_latents = n_latents
        self.obs_space = obs_space.original_space
        self.reconstruct_loss = nn.CrossEntropyLoss(reduction="none")
        obs_shape = self.obs_space["self"].shape
        # observation encoder.
        self.encoder = MLP([obs_shape[0], hidden_dim, 2 * n_latents], last_activation=False)
        # attention module aggregates the messages between other agents.
        self.linear_k = nn.Sequential(
            SlimFC(in_size=2 * n_latents,
                   out_size=2 * n_latents,
                   initializer=normc_initializer(1.0),
                   activation_fn=None),
        )
        self.linear_q = nn.Sequential(
            SlimFC(in_size=2 * n_latents,
                   out_size=2 * n_latents,
                   initializer=normc_initializer(1.0),
                   activation_fn=None),
        )
        self.linear_v = nn.Sequential(
            SlimFC(in_size=2 * n_latents,
                   out_size=2 * n_latents,
                   initializer=normc_initializer(1.0),
                   activation_fn=None),
        )
        self.attention = nn.MultiheadAttention(embed_dim=2 * n_latents, num_heads=obs_shape[0], batch_first=True)
        self.layer_norm = nn.LayerNorm(2 * n_latents)
        # communication method
        self.PoE = ProductOfExperts()
        # NN main body
        self.hidden_layers = MLP([n_latents, hidden_dim, hidden_dim])
        self.decoder_head = MLP([hidden_dim, 6 * obs_shape[0]], last_activation=False)
        self.value_layers = MLP([2 * n_latents, hidden_dim, hidden_dim])
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
        # Holds the current 'base' output (before logits layer).
        self.value_features = None

    @override(ModelV2)
    def value_function(self):
        assert self.value_features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self.value_features), [-1])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        shape = obs["self"].shape
        batch_size = shape[0]
        obs_len = shape[1]
        num_others = len(obs["others"])
        obs_mask = obs["obs_mask"]

        # self observation encoding
        me = self.encoder(obs["self"])
        real_me = self.encoder(obs["real_obs"])
        me_mu = me[:, :self.n_latents]
        me_logvar = me[:, self.n_latents:]
        self.final_mu, self.final_logvar = me_mu, me_logvar
        if not self.silent:
            prior_mu, prior_logvar = prior_expert((shape[0], 1, self.n_latents))
            others_obs = torch.cat([o.unsqueeze(1) for o in obs["others"]], dim=1)
            others_msg = self.encoder(others_obs)
            q = self.linear_q(others_msg)
            k = self.linear_k(others_msg)
            v = self.linear_v(others_msg)
            attn_output, attn_output_weights = self.attention(q.unsqueeze, k.unsqueeze, v.unsqueeze)
            attn_output = self.layer_norm(attn_output)
            cat_mu = torch.cat([prior_mu, me_mu.unsqueeze(1), attn_output[:, :, :self.n_latents]], dim=1)
            cat_logvar = torch.cat([prior_logvar, me_logvar.unsqueeze(1), attn_output[:, :, self.n_latents:]], dim=1)
            pd_mu, pd_logvar = self.PoE(cat_mu, cat_logvar, dim=1)
            self.final_mu, self.final_logvar = pd_mu, pd_logvar
        z = reparametrize(self.final_mu, self.final_logvar)
        self.action_features = self.hidden_layers(z)
        self.logits = self.decoder_head(self.action_features).view(batch_size, obs_len, -1)
        self.value_features = self.value_layers(torch.cat([real_me[:, :self.n_latents], z], dim=1))
        action_out = self.action_branch(self.action_features)
        return action_out, state


class CommnetModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 silent,
                 n_latents,
                 hidden_dim,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        self.silent = silent
        self.n_latents = n_latents
        self.obs_space = obs_space.original_space
        self.reconstruct_loss = nn.CrossEntropyLoss(reduction="none")
        obs_shape = self.obs_space["self"].shape
        # observation encoder.
        self.encoder = MLP([obs_shape[0], hidden_dim], last_activation=False)
        # communication channel
        self.f_linear = MLP([3 * hidden_dim, hidden_dim, hidden_dim])
        # action head
        self.action_branch = nn.Sequential(
            SlimFC(in_size=hidden_dim,
                   out_size=num_outputs,
                   initializer=normc_initializer(0.01),
                   activation_fn=None))
        # value head
        self.value_branch = nn.Sequential(
            SlimFC(in_size=hidden_dim,
                   out_size=1,
                   initializer=normc_initializer(0.01),
                   activation_fn=None))
        # Holds the current 'base' output (before logits layer).
        self.value_features = None

    @override(ModelV2)
    def value_function(self):
        assert self.value_features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self.value_features), [-1])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        shape = obs["self"].shape
        batch_size = shape[0]
        obs_len = shape[1]
        num_others = len(obs["others"])
        obs_mask = obs["obs_mask"]

        h = self.encoder(obs["self"])
        others_h = torch.cat([self.encoder(o.unsqueeze(1)) for o in obs["others"]], dim=1)
        total_h = torch.cat([h.unsqueeze(1), others_h], dim=1)
        total_init_h = total_h.clone()
        # 2 communication steps
        for _ in range(2):
            h = []
            h_app = h.append
            for i in range(num_others+1):
                mask = torch.ones_like(total_h)
                mask[:, i, :] = torch.zeros_like(mask[:, i, :])
                mask = torch.ge(mask, 0.5)
                masked_out = total_h[mask].view(batch_size, num_others, -1)
                c = torch.mean(masked_out, dim=1)
                f_input = torch.cat([total_h[:, i, :], c, total_init_h[:, i, :]], dim=1)
                h_app(self.f_linear(f_input).unsqueeze(1))
            total_h = torch.cat(h, dim=1)

        features = total_h[:, 0, :]
        self.value_features = features
        action_out = self.action_branch(features)
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
        obs_shape = self.obs_space["self"].shape
        self.gru_state_size = n_latents * obs_shape[0]
        # observation encoder
        self.fc1 = nn.Linear(obs_shape[0], hidden_dim)
        self.gru = nn.GRU(hidden_dim, self.gru_state_size, batch_first=True)
        # message distribution
        self.msg_dist = nn.Sequential(
            SlimFC(in_size=self.gru_state_size,
                   out_size=hidden_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn="relu"),
            SlimFC(in_size=hidden_dim,
                   out_size=hidden_dim,
                   initializer=normc_initializer(1.0),
                   activation_fn="relu"),
            SlimFC(in_size=hidden_dim,
                   out_size=2 * self.gru_state_size,
                   initializer=normc_initializer(1.0),
                   activation_fn=None),
        )
        # communication method
        self.PoE = ProductOfExperts()
        # NN main body
        self.hidden_layers = MLP(input_dim=self.gru_state_size, hidden_dim=hidden_dim)
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
        me = torch.relu(self.fc1(obs["self"]))
        gru_out_me, h_me = self.gru(me, torch.unsqueeze(state[0], 0))
        # others" messages encoding and concat.
        mu, logvar = prior_expert((shape[0], shape[1], self.gru_state_size, 1))
        msg_mus, msg_logvars = [mu.to(device)], [logvar.to(device)]
        states = [torch.squeeze(h_me, 0)]
        states_app, msg_mu_app, msg_logvar_app = states.append, msg_mus.append, msg_logvars.append
        for i, other in enumerate(obs["others"]):
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
