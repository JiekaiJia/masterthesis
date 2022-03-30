from abc import ABC

from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    SlimFC,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
from tensorboardX import SummaryWriter
from torch._C import is_grad_enabled

from model_components import (
    reparametrize,
    MLP,
    prior_expert,
    ProductOfExperts
)

torch, nn = try_import_torch()


class CommNetModel(TorchModelV2, nn.Module, ABC):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 silent,
                 attention,
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

        # observation encoder.
        encoder_in = obs_shape[0] + 1
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
        num_others = len(obs["others"])
        obs_mask = obs["obs_mask"]
        cur_obs = obs["self"] + 1

        h = self.encoder(torch.cat([cur_obs, obs["self_id"]], dim=1))  # bx64

        others_obs = torch.cat([o.unsqueeze(1)+1 for o in obs["others"]], dim=1)  # bxn-1x2
        # Filtering out the queue length that current agent doesn't have access to.
        digits_masks = [torch.cat([m.unsqueeze(1) for m in mask], dim=1) for mask in obs_mask]
        masked_obs_o = [others_obs * mask for mask in digits_masks]
        # Modified the others observations so that they have the same oder as the current agent.
        mask_other = [torch.zeros_like(o) for o in masked_obs_o]
        for i, x in enumerate(mask_other):
            x[:, :, i] += torch.ones_like(x[:, :, i])
        masked_obs_o = [torch.sum(o, dim=2, keepdim=True)*ms for ms, o in zip(mask_other, masked_obs_o)]
        modified_obs_o = sum(masked_obs_o)  # bxn-1x2
        codes_t = self.encoder(torch.cat([modified_obs_o, obs["other_ids"].view(batch_size, num_others, 1)], dim=2))

        total_h = torch.cat([h.unsqueeze(1), codes_t], dim=1)  # bxnx64
        total_init_h = total_h.clone()  # bxnx64
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
            total_h = torch.cat(h, dim=1)  # bxnx64

        # value head
        rh = self.encoder(torch.cat([obs["real_obs"]+1, obs["self_id"]], dim=1))  # bx64
        self.value_features = self.value_layer(rh)  # bx64
        # action head
        action_out = self.action_branch(total_h[:, 0, :])
        return action_out, state


class BicNetModel(TorchModelV2, nn.Module, ABC):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 silent,
                 attention,
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

        # observation encoder.
        encoder_in = obs_shape[0] + 1
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
        batch_size = shape[0]
        num_others = len(obs["others"])
        obs_mask = obs["obs_mask"]
        cur_obs = obs["self"] + 1

        me_obs = self.encoder(torch.cat([cur_obs, obs["self_id"]], dim=1))  # bx64
        r_me_obs = self.encoder(torch.cat([obs["real_obs"]+1, obs["self_id"]], dim=1))  # bx64

        others_obs = torch.cat([o.unsqueeze(1)+1 for o in obs["others"]], dim=1)  # bxn-1x2
        # Filtering out the queue length that current agent doesn't have access to.
        digits_masks = [torch.cat([m.unsqueeze(1) for m in mask], dim=1) for mask in obs_mask]
        masked_obs_o = [others_obs * mask for mask in digits_masks]
        # Modified the others observations so that they have the same oder as the current agent.
        mask_other = [torch.zeros_like(o) for o in masked_obs_o]
        for i, x in enumerate(mask_other):
            x[:, :, i] += torch.ones_like(x[:, :, i])
        masked_obs_o = [torch.sum(o, dim=2, keepdim=True)*ms for ms, o in zip(mask_other, masked_obs_o)]
        modified_obs_o = sum(masked_obs_o)  # bxn-1x2
        codes_t = self.encoder(torch.cat([modified_obs_o, obs["other_ids"].view(batch_size, num_others, 1)], dim=2))

        total_obs = torch.cat([me_obs.unsqueeze(1), codes_t], dim=1)  # bxnx64
        comm_features, _ = self.gru(total_obs)  # bxnx128

        features = torch.cat([total_obs[:, 0, :], comm_features[:, 0, :]], dim=-1)  # bx192
        action_out = self.action_branch(features)  # bx4
        self.value_features = self.value_layer(r_me_obs)  # bx64

        return action_out, state


class ATVCModel(TorchModelV2, nn.Module, ABC):
    """The network we proposed."""
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 silent,
                 attention,
                 n_latents,
                 hidden_dim,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        self.step = 0
        self.silent = silent
        self.attention = attention
        self.obs_space = obs_space.original_space
        self.reconstruct_loss = nn.CrossEntropyLoss(reduction="none")
        # self.writer = SummaryWriter()
        obs_shape = self.obs_space["self"].shape
        self.n_latents = n_latents * obs_shape[0]
        embedding_dim = 8
        # Index embedding
        self.id_embedding = MLP([1, embedding_dim], last_activation="tanh")
        # observation encoder.
        encode_in = obs_shape[0] + 1
        self.encoder = MLP([encode_in, hidden_dim, hidden_dim, 2 * self.n_latents], last_activation=False)
        self.decoder = MLP([self.n_latents, hidden_dim, hidden_dim, 6 * obs_shape[0]], last_activation=False)
        # communication method
        self.PoE = ProductOfExperts()
        # attention network
        att_in = 2 * self.n_latents + embedding_dim
        self.att_k = MLP([att_in, hidden_dim], last_activation="tanh")
        self.att_q = nn.Linear(hidden_dim, 1, bias=False)
        # NN main body
        self.action_branch = nn.Sequential(
            SlimFC(in_size=self.n_latents,
                   out_size=num_outputs,
                   initializer=normc_initializer(0.01),
                   activation_fn=None))
        self.value_branch = nn.Sequential(
            SlimFC(in_size=self.n_latents,
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
        n_others = len(obs["others"])
        cur_obs = obs["self"] + 1

        id_embedding = self.id_embedding(obs["self_id"]).unsqueeze(1)
        cur_id = id_embedding.repeat(1, n_others, 1)
        cur_gp = self.encoder(torch.cat([cur_obs, obs["self_id"]], dim=1))  # bx128
        cur_mu = cur_gp[:, :self.n_latents]
        cur_logvar = cur_gp[:, self.n_latents:]
        self.final_mu, self.final_logvar = cur_mu, cur_logvar
        if not self.silent:
            prior_mu, prior_logvar = prior_expert((batch_size, 1, self.n_latents), use_cuda=torch.cuda.is_available())  # bx1x64
            # Collecting others observations.
            others_obs = torch.cat([o.unsqueeze(1)+1 for o in obs["others"]], dim=1)  # bxn-1x2
            # Filtering out the queue length that current agent doesn't have access to.
            digits_masks = [torch.cat([m.unsqueeze(1) for m in mask], dim=1) for mask in obs_mask]
            masked_obs_o = [others_obs * mask for mask in digits_masks]
            # Modified the others observations so that they have the same oder as the current agent.
            mask_other = [torch.zeros_like(o) for o in masked_obs_o]
            for i, x in enumerate(mask_other):
                x[:, :, i] += torch.ones_like(x[:, :, i])
            masked_obs_o = [torch.sum(o, dim=2, keepdim=True)*ms for ms, o in zip(mask_other, masked_obs_o)]
            modified_obs_o = sum(masked_obs_o)  # bxn-1x2
            # Gaussian process inference
            others_gp = self.encoder(torch.cat([modified_obs_o, obs["other_ids"].view(batch_size, n_others, -1)], dim=2))  # bxn-1x128
            others_mu = others_gp[:, :, :self.n_latents]
            others_logvar = others_gp[:, :, self.n_latents:]
            if self.attention:
                total_k = self.att_k(torch.cat([others_gp, cur_id], dim=2))  # bxn-1x64
                total_q = self.att_q(total_k)  # bxn-1
                att_weight = torch.softmax(total_q, dim=1)
                # att_weight = torch.where(att_weights>0.3, torch.ones_like(att_weights), torch.zeros_like(att_weights))
                # communication_f = torch.sum(att_weights, dim=1)
                # comm_ave = torch.mean(communication_f)/2
                # self.writer.add_scalar('comm_number', comm_ave.item(), global_step=self.step)
                log_att_weight = torch.log(att_weight+1e-8)
                # Multiply weight to Gaussian
                others_logvar = others_logvar - log_att_weight
            # Making PoE
            pd_mu, pd_logvar = self.PoE(torch.cat([prior_mu, others_mu, cur_mu.unsqueeze(1)], dim=1),
                                        torch.cat([prior_logvar, others_logvar, cur_logvar.unsqueeze(1)], dim=1),
                                        dim=1)
            self.final_mu, self.final_logvar = pd_mu, pd_logvar
        z = reparametrize(self.final_mu, self.final_logvar)
        self.predictions = self.decoder(z).view(batch_size, obs_len, -1)
        self.value_features = z
        action_out = self.action_branch(z)

        self.step += 1
        return action_out, state
