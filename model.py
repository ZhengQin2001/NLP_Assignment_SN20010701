import torch
from torch import nn

class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, vocab_size):
        super(CVAE, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.hidden2mean = nn.Linear(hidden_dim, latent_dim)
        self.hidden2logv = nn.Linear(hidden_dim, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim + input_dim, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_embeddings, condition_embeddings):
        projected_input = self.input_projection(input_embeddings)
        encoder_output = self.encoder(projected_input)
        mu = self.hidden2mean(encoder_output)
        logvar = self.hidden2logv(encoder_output)
        z = self.reparameterize(mu, logvar)
        hidden = self.latent2hidden(z)
        combined = torch.cat((hidden, condition_embeddings), dim=-1)
        decoder_output = self.output_projection(combined)
        return decoder_output, mu, logvar
