import torch
import torch.nn as nn


class GRUtrigger(nn.Module):

    def __init__(self, obs_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        """
        obs_size: observation length
        embedding_dim: length of embedding vector
        hidden_dim: Number of GRU's neuron
        layer_dim: Number of GRU's layer
        output_dim: length of output
        """
        super().__init__()
        # embedding
        self.embedding = nn.Embedding(obs_size, embedding_dim)
        # GRU
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x : [bacth, time_step, vocab_size]
        embeds = self.embedding(x)
        # embeds : [batch, time_step, embedding_dim]
        r_out, h_n = self.gru(embeds, None)
        # r_out : [batch, time_step, hidden_dim]
        out = self.fc1(r_out[:, -1, :])
        # out : [batch, time_step, output_dim]
        return out


if __name__ == '__main__':
    from torchsummary import summary
    model = GRUtrigger(obs_size=5, embedding_dim=128, hidden_dim=128, layer_dim=2, output_dim=5)
    summary(model, input_size=(5,), batch_size=-1)
