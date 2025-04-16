import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# ZINB LOSS FUNCTION
# ----------------------------
class ZINBLoss(nn.Module):
    def __init__(self, ridge_lambda=0.0):
        super(ZINBLoss, self).__init__()
        self.eps = 1e-10
        self.ridge_lambda = ridge_lambda

    def forward(self, x, mean, dispersion, pi, scale_factor=1.0):
        # Scale the mean using the scale factor
        mean = mean * scale_factor

        nb_case = (
            torch.lgamma(dispersion + self.eps)
            + torch.lgamma(x + 1.0)
            - torch.lgamma(x + dispersion + self.eps)
            - dispersion * torch.log(dispersion + self.eps)
            - x * torch.log(mean + self.eps)
            + (dispersion + x) * torch.log(dispersion + mean + self.eps)
        )

        zero_case = -torch.log(pi + ((1.0 - pi) * torch.exp(-nb_case)) + self.eps)
        result = torch.where(torch.lt(x, 1e-8), zero_case, -torch.log(1.0 - pi + self.eps) + nb_case)
        ridge = self.ridge_lambda * (pi ** 2).sum()

        return result.mean() + ridge

# ----------------------------
# CONDITIONAL ENCODER
# ----------------------------
class ConditionalEncoder(nn.Module):
    def __init__(self, input_dim, s_dim, hidden_dim, latent_dim):
        """
        input_dim: Number of genes/features.
        s_dim: Dimensionality of the batch label (e.g. one-hot vector length).
        hidden_dim: Number of hidden units.
        latent_dim: Dimensionality of the latent space.
        """
        super(ConditionalEncoder, self).__init__()
        # Concatenate x and s so the input dimension is increased by s_dim
        self.fc1 = nn.Linear(input_dim + s_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, s):
        # Concatenate gene expression data and batch label
        xs = torch.cat((x, s), dim=1)
        h = F.relu(self.fc1(xs))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# ----------------------------
# CONDITIONAL DECODER
# ----------------------------
class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, s_dim, hidden_dim, output_dim):
        """
        latent_dim: Dimensionality of the latent space.
        s_dim: Dimensionality of the batch label.
        hidden_dim: Number of hidden units.
        output_dim: Number of genes/features (should match input_dim).
        """
        super(ConditionalDecoder, self).__init__()
        # Concatenate z and s so the decoder knows which batch to condition on
        self.fc1 = nn.Linear(latent_dim + s_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_disp = nn.Linear(hidden_dim, output_dim)
        self.fc_pi = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, s):
        # Concatenate latent variable and batch label
        zs = torch.cat((z, s), dim=1)
        h = F.relu(self.fc1(zs))
        mean = F.relu(self.fc_mean(h))
        dispersion = F.relu(self.fc_disp(h))
        pi = torch.sigmoid(self.fc_pi(h))
        return mean, dispersion, pi

# ----------------------------
# CONDITIONAL VAE MODULE
# ----------------------------
class ConditionalZINBVAE(nn.Module):
    def __init__(self, input_dim, s_dim, hidden_dim, latent_dim):
        """
        input_dim: Number of genes/features.
        s_dim: Dimensionality of the batch label.
        hidden_dim: Number of hidden units.
        latent_dim: Dimensionality of the latent space.
        """
        super(ConditionalZINBVAE, self).__init__()
        self.encoder = ConditionalEncoder(input_dim, s_dim, hidden_dim, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, s_dim, hidden_dim, input_dim)
        self.zinb_loss_fn = ZINBLoss()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, s):
        # s should be provided as the batch label (e.g. one-hot encoded vector)
        mu, logvar = self.encoder(x, s)
        z = self.reparameterize(mu, logvar)
        mean, disp, pi = self.decoder(z, s)
        return mean, disp, pi, mu, logvar, z

    def loss_function(self, x, mean, disp, pi, mu, logvar):
        zinb_loss = self.zinb_loss_fn(x, mean, disp, pi)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return zinb_loss + 1e-3 * kl_div
