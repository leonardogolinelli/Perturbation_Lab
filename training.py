import torch
import torch.nn.functional as F
import numpy as np
from utils import set_seed


def train(vae, optimizer, x, s, n_epochs, n_cells, mu_values, theta_values, pi_values, device, seed=11):
    """
    Train the Conditional VAE model.

    Parameters:
        vae (VAE): The ConditionalZINBVAE model to train.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        x (torch.Tensor): The input data (cells x genes).
        s (torch.Tensor): One-hot encoded batch/state labels (cells x num_states).
        n_epochs (int): Number of training epochs.
        n_cells (int): Number of original (non-perturbed) cells.
        mu_values (np.ndarray): Ground truth means.
        theta_values (np.ndarray): Ground truth dispersions.
        pi_values (np.ndarray): Ground truth dropout probabilities.
        device (str or torch.device): Computation device ('cuda' or 'cpu').
        seed (int): Random seed for reproducibility.
    """
    set_seed(seed)
    
    # Ensure tensors are on the right device
    x = torch.tensor(x, dtype=torch.float32).to(device)
    s = s.to(device)
    true_mu_tensor = torch.tensor(mu_values, dtype=torch.float32).to(device)
    true_theta_tensor = torch.tensor(theta_values, dtype=torch.float32).to(device)
    true_pi_tensor = torch.tensor(pi_values, dtype=torch.float32).to(device)

    vae.to(device)
    vae.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass with conditioning
        mean, disp, pi, mu, logvar, z = vae(x, s)

        # Compute loss
        loss = vae.loss_function(x, mean, disp, pi, mu, logvar)
        loss.backward()
        optimizer.step()

        # Evaluation
        with torch.no_grad():
            mean_mse = F.mse_loss(mean, true_mu_tensor).item()
            theta_mse = F.mse_loss(disp, true_theta_tensor).item()
            pi_mse = F.mse_loss(pi, true_pi_tensor).item()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                  f"Mean MSE: {mean_mse:.4f}, Theta MSE: {theta_mse:.4f}, Pi MSE: {pi_mse:.4f}")

    # Return final tensors
    return mean, disp, pi, mu, logvar, z
