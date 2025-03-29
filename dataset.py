import numpy as np
from scipy.stats import nbinom

def base_genes(n_cells, 
               n_genes, 
               target_zero_prob, 
               mu_bot=1.0, 
               mu_top=10, 
               theta_bot=0.5, 
               theta_top=2.0,
               seed=42):
    
    # Simulate gene-specific parameters
    np.random.seed(seed)
    mu_values = np.random.uniform(mu_bot, mu_top, n_genes)         # Mean of NB
    theta_values = np.random.uniform(theta_bot, theta_top, n_genes)   # Dispersion 

    # Initialize gene expression matrix
    x = np.zeros((n_cells, n_genes))

    pi_values = []
    # Fill the matrix with ZINB simulated data, solving for pi to match target zero probability
    for j in range(n_genes):
        mu = mu_values[j]
        r = 1 / theta_values[j]  

        # Scipy-compatible NB probability
        p = r / (r + mu)

        # Compute probability of zero from NB
        p_nb_zero = nbinom.pmf(0, r, p)

        # Solve for pi to match target overall zero probability
        pi = (target_zero_prob - p_nb_zero) / (1 - p_nb_zero)
        pi = np.clip(pi, 0, 1)  # Ensure it's within [0,1]
        pi_values.append(pi)

        # Simulate NB samples
        nb_samples = nbinom.rvs(r, p, size=n_cells)

        # Apply zero-inflation
        zero_mask = np.random.binomial(1, pi, size=n_cells)
        nb_samples[zero_mask == 1] = 0

        x[:, j] = nb_samples

    pi_values = np.array(pi_values)

    # Check actual zero proportion
    actual_zero_rate = (x == 0).sum() / x.size
    print(f"Actual overall zero proportion: {actual_zero_rate:.4f}")

    tile_up = lambda x : np.tile(x, (n_cells, 1))
    mu_values, theta_values, pi_values = [tile_up(arr) for arr in [mu_values, theta_values, pi_values]]

    return x, mu_values, theta_values, pi_values

def get_mech_dataset(x, 
                     mu_values, 
                     theta_values, 
                     pi_values, 
                     n_cells=10000, 
                     n_genes=5, 
                     target_zero_prob=.8,
                     seeds=[42,11],
                     theta_fixed=True):
    
    seed1,seed2 = seeds
    np.random.seed(seed1)
    mean_weights = np.random.uniform(-2, +2, n_genes)
    mu = np.sum(x * mean_weights, axis=1) #mean of ZINB
    mu = np.clip(mu, 0.1, None)

    np.random.seed(seed2)
    if theta_fixed:
        theta = np.ones_like(mu) * 1e-3
        dispersion_weights = None
    else:
        dispersion_weights = np.random.uniform(-2, +2, n_genes)
        theta = np.average(x, axis=1, weights=dispersion_weights) #dispersion of ZINB
        theta = np.clip(theta, 0.1, None)

    r = 1 / theta  

    # Scipy-compatible NB probability
    p = r / (r + mu)

    # Compute probability of zero from NB
    p_nb_zero = nbinom.pmf(0, r, p)

    # Solve for pi to match target overall zero probability
    pi = (target_zero_prob - p_nb_zero) / (1 - p_nb_zero)
    pi = np.clip(pi, 0, 1)  # Ensure it's within [0,1]

    # Simulate NB samples
    nb_samples = nbinom.rvs(r, p, size=n_cells)

    # Apply zero-inflation
    zero_mask = np.random.binomial(1, pi, size=n_cells)
    nb_samples[zero_mask == 1] = 0

    x_comb = nb_samples

    #tile_up = lambda x : np.tile(x, (n_cells, 1))
    #mu_values, theta_values, pi_values = [tile_up(arr) for arr in [mu_values, theta_values, pi_values]]

    stack_up = lambda x,y : np.hstack((x,y.reshape((n_cells,1))))
    pairs = [(x, x_comb), (mu_values, mu), (theta_values, theta), (pi_values, pi)]
    x, mu_values, theta_values, pi_values = [stack_up(x,y) for x,y in pairs]

    n_genes += 1

    return x, mu_values, theta_values, pi_values, mean_weights, dispersion_weights, n_genes