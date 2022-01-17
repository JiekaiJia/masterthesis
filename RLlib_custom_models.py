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
    """The network we proposed."""
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
        self.obs_space = obs_space.original_space
        self.reconstruct_loss = nn.CrossEntropyLoss(reduction="none")
        obs_shape = self.obs_space["self"].shape
        n_latents = obs_shape[0] * n_latents
        self.n_latents = n_latents
        # observation encoder.
        encoder_in = obs_shape[0]
        self.encoder = MLP([encoder_in, hidden_dim, hidden_dim, 2 * n_latents], last_activation=False)
        self.decoder = MLP([n_latents, hidden_dim, hidden_dim, 6 * obs_shape[0]], last_activation=False)
        # communication method
        self.PoE = ProductOfExperts()
        # NN main body
        self.hidden_layers = MLP([n_latents, hidden_dim, hidden_dim])
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
        num_others = len(obs["others"])

        me = self.encoder(obs["self"])
        real_me = self.encoder(obs["real_obs"])
        me_mu = me[:, :self.n_latents]
        me_logvar = me[:, self.n_latents:]
        self.final_mu, self.final_logvar = me_mu, me_logvar
        if not self.silent:
            prior_mu, prior_logvar = prior_expert((shape[0], 1, self.n_latents))  # bx1x64
            others_obs = torch.cat([o.unsqueeze(1) for o in obs["others"]], dim=1)  # bxnx2
            digits_masks = [torch.cat([m.unsqueeze(1) for m in mask], dim=1) for mask in obs_mask]
            masked_obs_o = [others_obs * mask for mask in digits_masks]
            codes = [self.encoder(o) for o in masked_obs_o]
            mus = [code[:, :, :self.n_latents].view(batch_size, num_others, obs_len, -1) for code in codes]
            logvars = [code[:, :, self.n_latents:].view(batch_size, num_others, obs_len, -1) for code in codes]
            sum_mus = [torch.sum(mu, keepdim=True, dim=2) for mu in mus]
            sum_logvars = [torch.sum(logvar,  keepdim=True, dim=2) for logvar in logvars]
            mus_t = torch.cat(sum_mus, dim=2).view(batch_size, num_others, -1)
            logvars_t = torch.cat(sum_logvars, dim=2).view(batch_size, num_others, -1)  # bxnx64
            pd_mu, pd_logvar = self.PoE(torch.cat([prior_mu, mus_t, me_mu.unsqueeze(1)], dim=1),
                                        torch.cat([prior_logvar, logvars_t, me_logvar.unsqueeze(1)], dim=1),
                                        dim=1)
            # msg = [self.PoE(torch.cat([prior_mu, code[:, :, :self.n_latents]], dim=1), torch.cat([prior_logvar, code[:, :, self.n_latents:]], dim=1), dim=1)
            #        for code in codes]
            # sprune_msg = [self.PoE(mu.view(batch_size, obs_len, -1), logvar.view(batch_size, obs_len, -1), dim=1) for mu, logvar in msg]
            # sprune_mus, sprune_logvars = [], []
            # sprune_mus_app, sprune_logvars_app = sprune_mus.append, sprune_logvars.append
            # for sprune_mu, sprune_logvar in sprune_msg:
            #   sprune_mus_app(sprune_mu)
            #   sprune_logvars_app(sprune_logvar)
            # sprune_mu_tensor = torch.cat(sprune_mus, dim=1)
            # sprune_logvar_tensor = torch.cat(sprune_logvars, dim=1)
            # pd_mu, pd_logvar = self.PoE(torch.cat([sprune_mu_tensor.unsqueeze(1), me_mu.unsqueeze(1)], dim=1), torch.cat([sprune_logvar_tensor.unsqueeze(1), me_logvar.unsqueeze(1)], dim=1), dim=1)
            self.final_mu, self.final_logvar = pd_mu, pd_logvar
        z = reparametrize(self.final_mu, self.final_logvar)
        # self.action_features = self.hidden_layers(z)
        self.action_features = z
        self.predictions = self.decoder(z).view(batch_size, obs_len, -1)
        # self.value_features = self.value_layers(torch.cat([real_me[:, :self.n_latents], z], dim=1))
        self.value_features = real_me[:, :self.n_latents]
        action_out = self.action_branch(self.action_features)
        return action_out, state


class MaskSPoEModel(TorchModelV2, nn.Module):
    """A variant we tried, but not satisfying"""
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
        self.n_latents = n_latents  # latent variables per length
        self.obs_space = obs_space.original_space
        self.reconstruct_loss = nn.CrossEntropyLoss(reduction='none')
        obs_shape = self.obs_space["self"].shape
        num_agents = len(self.obs_space["others"]) + 1
        # Gaussian process estimation
        encoder_in = obs_shape[0] + 1  # [index, observation]
        self.h_encoder = MLP([encoder_in, hidden_dim, hidden_dim, 2 * obs_shape[0] * n_latents], last_activation=False)
        self.h_decoder = MLP([obs_shape[0] * n_latents, hidden_dim, hidden_dim, 2 * 6 * obs_shape[0]], last_activation=False)
        # communication method
        self.PoE = ProductOfExperts()
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
        num_others = len(obs["others"])

        # Gaussain process estimation
        me_in = torch.cat([obs["self"], obs["self_id"]], dim=-1)
        me_gp = self.h_encoder(me_in).view(batch_size, obs_len, -1)  # bx2x64
        me_mu = me_gp[:, :, :self.n_latents]  # bx2x32
        me_logvar = me_gp[:, :, self.n_latents:]  # bx2x32
        self.final_mu, self.final_logvar = me_mu, me_logvar

        rme_in = torch.cat([obs["real_obs"], obs["self_id"]], dim=-1)
        rme_gp = self.h_encoder(rme_in).view(batch_size, obs_len, -1)  # bx2x64
        rme_mu = rme_gp[:, :, :self.n_latents]  # bx2x32
        rme_logvar = rme_gp[:, :, self.n_latents:]  # bx2x32
        rz = reparametrize(rme_mu, rme_logvar).view(batch_size, -1)  # bx64
        if not self.silent:
            prior_mu, prior_logvar = prior_expert((batch_size, 1, 2, self.n_latents))  # bx1x2x32

            others_obs = torch.cat([o.unsqueeze(1) for o in obs["others"]], dim=1)  # bxnx2
            # Gaussain process estimation
            o_in = torch.cat([others_obs, obs["other_ids"].unsqueeze(-1)], dim=-1)  # bxnx3
            o_gp = self.h_encoder(o_in).view(batch_size, num_others, obs_len, -1)  # bxnx2x64
            # mask out the irrelevant variables
            digits_masks = [torch.cat([m.unsqueeze(1) for m in mask], dim=1).detach() for mask in obs_mask]  # [bxnx2]x2
            masked_gps = [o_gp * m.unsqueeze(-1) for m in digits_masks]  # [bxnx2x64]x2
            sum_masked_gps = [torch.sum(gp, keepdim=True, dim=2) for gp in masked_gps]  # [bxnx1x64]x2
            cat_gps = torch.cat(sum_masked_gps, dim=2)  # bxnx2x64
            gps_mu = cat_gps[:, :, :, :self.n_latents]  # bxnx2x32
            gps_logvar = cat_gps[:, :, :, self.n_latents:]  # bxnx2x32
            # PoE
            to_pd_mu = torch.cat([prior_mu, gps_mu, me_mu.unsqueeze(1)], dim=1)  # bxn+2x2x32
            to_pd_lovar = torch.cat([prior_logvar, gps_logvar, me_logvar.unsqueeze(1)], dim=1)  # bxn+2x2x32
            pd_mu, pd_logvar = self.PoE(to_pd_mu, to_pd_lovar, dim=1)  # bx2x32
            self.final_mu, self.final_logvar = pd_mu, pd_logvar
        z = reparametrize(self.final_mu, self.final_logvar)  # bx2x32
        # reconstruction
        p_gp = self.h_decoder(z.view(batch_size, -1)).view(batch_size, obs_len, -1)  # bx2x12
        self.predictions = reparametrize(p_gp[:, :, :6], p_gp[:, :, 6:])  # bx2x6
        # action and value
        self.value_features = rme_mu.contiguous().view(batch_size, -1)
        action_out = self.action_branch(z.view(batch_size, -1))

        self.final_mu = self.final_mu.view(batch_size, -1)
        self.final_logvar = self.final_logvar.view(batch_size, -1)

        return action_out, state


class MaskRNNPoEModel(TorchRNN, nn.Module):
    """Tried to use RNN for inference, increasing computational complexity without improvement."""
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
        self.hidden_dim = hidden_dim
        self.reconstruct_loss = nn.CrossEntropyLoss(reduction="none")
        obs_shape = self.obs_space["self"].shape
        num_agents = len(self.obs_space["others"]) + 1
        # observation encoder.
        self.id_embedding = nn.Embedding(num_agents, 8)
        self.obs_embedding = MLP([obs_shape[0], hidden_dim])

        self.h_encoder = MLP([1, hidden_dim, hidden_dim, 2 * n_latents], last_activation=False)
        self.h_decoder = MLP([n_latents, hidden_dim, hidden_dim, 2 * 6], last_activation=False)

        rnn_in = 8 + hidden_dim
        self.rnn_embedding = nn.Linear(rnn_in, rnn_in)
        self.rnn_encoder = nn.GRU(rnn_in, hidden_dim, batch_first=True)
        # communication method
        self.PoE = ProductOfExperts()
        # NN main body
        main_in = hidden_dim + 2 * n_latents
        self.hidden_layers = MLP([main_in, hidden_dim, hidden_dim])
        self.value_layers = MLP([main_in, hidden_dim, hidden_dim])
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
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [self.rnn_embedding.weight.new(1, self.hidden_dim).zero_().squeeze(0)] * 2
        return h

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        obs = restore_original_dimensions(inputs, self.obs_space, "torch")
        shape = obs["self"].shape
        batch_size = shape[0]
        seq_len = shape[1]
        obs_len = shape[2]
        obs_mask = obs["obs_mask"]
        eps = 1e-8

        # index embedding
        me_id = torch.tanh(self.id_embedding(obs["self_id"].long())).squeeze(2)  # bxsx8
        # observation embedding
        me_obs = self.obs_embedding(obs["self"])  # bxsx64
        r_me_obs = self.obs_embedding(obs["real_obs"])  # bxsx64
        # Gaussian estimation
        me_gp = self.h_encoder(obs["self"].unsqueeze(-1))  # bxsx2x64
        me_mu = me_gp[:, :, :, :self.n_latents]  # bxsx2x32
        me_logvar = me_gp[:, :, :, self.n_latents:]  # bxsx2x32
        self.final_mu, self.final_logvar = me_mu, me_logvar

        r_me_gp = self.h_encoder(obs["real_obs"].unsqueeze(-1))  # bxsx2x64
        r_me_mu = r_me_gp[:, :, :, :self.n_latents]  # bxsx2x32
        r_me_logvar = r_me_gp[:, :, :, self.n_latents:]  # bxsx2x32
        r_me_z = reparametrize(r_me_mu, r_me_logvar)  # bxsx2x32

        if not self.silent:
            prior_mu, prior_logvar = prior_expert((batch_size, seq_len, 1, obs_len, self.n_latents))  # bxsx1x2x32
            # obs embedding
            others_obs = torch.cat([o.unsqueeze(2) for o in obs["others"]], dim=2)  # bxsxnx2
            # index embedding
            others_id = self.id_embedding(obs["other_ids"].long())  # bxsxnx8
            # Gaussain process estimation
            o_gp = self.h_encoder(others_obs.unsqueeze(-1))  # bxsxnx2x64
            # mask out the irrelevant variables
            digits_masks = [torch.cat([m.unsqueeze(2) for m in mask], dim=2).detach() for mask in obs_mask]  # [bxsxnx2]x2
            masked_gps = [o_gp * m.unsqueeze(-1) for m in digits_masks]  # [bxsxnx2x64]x2
            sum_masked_gps = [torch.sum(gp, keepdim=True, dim=3) for gp in masked_gps]  # [bxsxnx1x64]x2
            cat_gps = torch.cat(sum_masked_gps, dim=3)  # bxsxnx2x64
            gps_mu = cat_gps[:, :, :, :, :self.n_latents]  # bxsxnx2x32
            gps_logvar = cat_gps[:, :, :, :, self.n_latents:]  # bxsxnx2x32
            # PoE
            to_pd_mu = torch.cat([prior_mu, gps_mu, me_mu.unsqueeze(2)], dim=2)  # bxsxn+2x2x32
            to_pd_lovar = torch.cat([prior_logvar, gps_logvar, me_logvar.unsqueeze(2)], dim=2)  # bxsxn+2x2x32
            pd_mu, pd_logvar = self.PoE(to_pd_mu, to_pd_lovar, dim=2)  # bxsx2x32
            self.final_mu, self.final_logvar = pd_mu, pd_logvar

        z = reparametrize(self.final_mu, self.final_logvar)  # bxsx2x32
        # histories computing
        rnn_in = torch.cat([me_id, me_obs], dim=-1)  # bxsx72
        rnn_out, h = self.rnn_encoder(rnn_in, state[0].unsqueeze(0))  # rnn_out: bxsx64  h: bx64
        # action head
        act_in = torch.cat([z.view(batch_size, seq_len, -1), rnn_out], dim=-1)  # bxsx128
        action_features = self.hidden_layers(act_in)  # bxsx64
        action_out = self.action_branch(action_features)
        # value head
        r_rnn_in = torch.cat([me_id, r_me_obs], dim=-1)  # bxsx72
        r_rnn_out, rh = self.rnn_encoder(r_rnn_in, state[1].unsqueeze(0))  # rnn_out: bxsx64  h: bx64
        value_in = torch.cat([r_me_z.view(batch_size, seq_len, -1), r_rnn_out], dim=-1)  # bxsx128
        self.value_features = self.value_layers(value_in)

        self.final_mu = self.final_mu.contiguous().view(batch_size*seq_len, -1)
        self.final_logvar = self.final_logvar.contiguous().view(batch_size*seq_len, -1)
        p_gp = self.h_decoder(z)  # bxsx2x12
        self.predictions = reparametrize(p_gp[:, :, :, :6], p_gp[:, :, :, 6:]).view(batch_size*seq_len, obs_len, -1)

        state = [h.squeeze(0), rh.squeeze(0)]
        
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
        obs_shape = self.obs_space["self"].shape
        num_agents = len(self.obs_space["others"]) + 1

        # observation encoder.
        encoder_in = obs_shape[0]
        self.encoder = MLP([encoder_in, obs_shape[0] * n_latents])
        # communication channel
        self.f_linear = MLP([3 * obs_shape[0] * n_latents, hidden_dim, obs_shape[0] * n_latents], activation="tanh")
        # action head
        action_in = obs_shape[0] * n_latents
        self.action_branch = nn.Sequential(
            SlimFC(in_size=action_in,
                   out_size=num_outputs,
                   initializer=normc_initializer(0.01),
                   activation_fn=None))
        # value head
        value_in = obs_shape[0] * n_latents
        self.value_layer = MLP([value_in, hidden_dim, hidden_dim])
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

        h = self.encoder(obs["self"])  # bx64

        others_obs = torch.cat([o.unsqueeze(1) for o in obs["others"]], dim=1)  # bxnx2
        digits_masks = [torch.cat([m.unsqueeze(1) for m in mask], dim=1) for mask in obs_mask]
        masked_obs_o = [others_obs * mask for mask in digits_masks]  # [bxnx2]x2
        codes = [self.encoder(o).view(batch_size, num_others, obs_len, -1) for o in masked_obs_o]  # [bxnx2x32]x2
        sum_codes = [torch.sum(code, keepdim=True, dim=2) for code in codes]
        codes_t = torch.cat(sum_codes, dim=2).view(batch_size, num_others, -1)  # bxnx64

        total_h = torch.cat([h.unsqueeze(1), codes_t], dim=1)  # bxn+1x64
        total_init_h = total_h.clone()  # bxn+1x64
        # 2 communication steps
        for _ in range(2):
            h = []
            h_app = h.append
            for i in range(num_others+1):
                mask = torch.ones_like(total_h)
                mask[:, i, :] = torch.zeros_like(mask[:, i, :])
                mask = torch.ge(mask, 0.5)
                masked_out = total_h[mask].view(batch_size, num_others, -1)
                c = torch.mean(masked_out, dim=1)  # bx64
                f_input = torch.cat([total_h[:, i, :], c, total_init_h[:, i, :]], dim=-1)  # bx192
                h_app(self.f_linear(f_input).unsqueeze(1))  # bx1x64
            total_h = torch.cat(h, dim=1)  # bxn+1x64

        # value head
        rh = self.encoder(obs["real_obs"])  # bx64
        self.value_features = self.value_layer(rh)  # bx64
        # action head
        action_out = self.action_branch(total_h[:, 0, :])
        return action_out, state


class BicnetModel(TorchModelV2, nn.Module):
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
        obs_shape = self.obs_space["self"].shape
        num_agents = len(self.obs_space["others"]) + 1

        # observation encoder.
        encoder_in = obs_shape[0]
        self.encoder = MLP([encoder_in, obs_shape[0] * n_latents])
        # communication channel
        self.gru = nn.GRU(obs_shape[0] * n_latents, hidden_dim, bidirectional=True, batch_first=True)
        # action head
        action_in = 3 * hidden_dim
        self.action_branch = nn.Sequential(
            SlimFC(in_size=action_in,
                   out_size=num_outputs,
                   initializer=normc_initializer(0.01),
                   activation_fn=None))
        # value head
        value_in = hidden_dim
        self.value_layer = MLP([value_in, hidden_dim, hidden_dim])
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
        num_others = len(obs["others"])
        batch_size = shape[0]
        obs_len = shape[1]
        obs_mask = obs["obs_mask"]

        me_obs = self.encoder(obs["self"])  # bx64
        r_me_obs = self.encoder(obs["real_obs"])  # bx64

        others_obs = torch.cat([o.unsqueeze(1) for o in obs["others"]], dim=1)  # bxnx2
        digits_masks = [torch.cat([m.unsqueeze(1) for m in mask], dim=1) for mask in obs_mask]
        masked_obs_o = [others_obs * mask for mask in digits_masks]  # [bxnx2]x2
        codes = [self.encoder(o).view(batch_size, num_others, obs_len, -1) for o in masked_obs_o]  # [bxnx2x32]x2
        sum_codes = [torch.sum(code, keepdim=True, dim=2) for code in codes]
        codes_t = torch.cat(sum_codes, dim=2).view(batch_size, num_others, -1)  # bxnx64

        total_obs = torch.cat([me_obs.unsqueeze(1), codes_t], dim=1)  # bxn+1x64
        comm_features, _ = self.gru(total_obs)  # bxn+1x128

        features = torch.cat([total_obs[:, 0, :], comm_features[:, 0, :]], dim=-1)  # bx192
        action_out = self.action_branch(features)  # bx4
        self.value_features = self.value_layer(r_me_obs)  # bx64

        return action_out, state
