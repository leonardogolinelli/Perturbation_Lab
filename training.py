import torch
import torch.nn.functional as F
import numpy as np
from utils import set_seed

def train(vae, optimizer, x, n_epochs, n_cells, mu_values, theta_values, pi_values, device, seed=11):
    """
    Train the VAE model.
    
    Parameters:
        vae (VAE): The VAE model to train.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        x (torch.Tensor): The input data.
        true_mu_tensor (torch.Tensor): The true mean tensor.
        true_theta_tensor (torch.Tensor): The true theta tensor.
        true_pi_tensor (torch.Tensor): The true pi tensor.
    """

    true_mu_tensor = torch.tensor(np.tile(mu_values, (n_cells, 1)) , dtype=torch.float32).to(device)
    true_theta_tensor = torch.tensor(np.tile(theta_values, (n_cells, 1)), dtype=torch.float32).to(device)
    true_pi_tensor = torch.tensor(np.tile(pi_values, (n_cells, 1)), dtype=torch.float32).to(device)

    #to_device = lambda x : x.to(device)

    
    

    x = torch.tensor(x, dtype=torch.float32).to(device)
    vae.to(device)


    set_seed(seed)
    vae.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        mean, disp, pi, mu, logvar, z = vae(x)
        loss = vae.loss_function(x, mean, disp, pi, mu, logvar)
        loss.backward()
        optimizer.step()
        #print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            mean_mse = F.mse_loss(mean, true_mu_tensor).item()
            theta_mse = F.mse_loss(disp, true_theta_tensor).item()
            pi_mse = F.mse_loss(pi, true_pi_tensor).item()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Mean MSE: {mean_mse:.4f}, Theta MSE: {theta_mse:.4f}, Pi MSE: {pi_mse:.4f}")

    return mean, disp, pi, mu, logvar, z