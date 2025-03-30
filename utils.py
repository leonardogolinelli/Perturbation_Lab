import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_zinb(mean, disp, pi, eps=1e-8, seed=11):
    """
    Sample counts from a Zero-Inflated Negative Binomial distribution using NumPy.

    Parameters:
        mean: np.ndarray [n_cells, n_genes]
        disp: np.ndarray [n_cells, n_genes]
        pi:   np.ndarray [n_cells, n_genes]
        eps:  Small constant to prevent division by zero.
        seed: Random seed for reproducibility.

    Returns:
        zinb_sample: np.ndarray [n_cells, n_genes]
    """
    np.random.seed(seed)
    mean = mean.astype(np.float32)
    disp = disp.astype(np.float32)
    pi = pi.astype(np.float32)

    # Compute NB parameters
    p = mean / (mean + disp + eps)  # success probability
    r = disp  # number of failures

    # Gamma-Poisson trick for NB sampling
    gamma_sample = np.random.gamma(shape=r, scale=1.0, size=mean.shape)
    nb_sample = np.random.poisson(gamma_sample * (p / (1 - p + eps)))

    # Apply zero inflation
    keep_mask = np.random.binomial(1, 1.0 - pi)
    zinb_sample = nb_sample * keep_mask

    return zinb_sample

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_column_densities(matrix1, matrix2, column_index, label1='Matrix 1', label2='Matrix 2'):
    """
    Plots the KDE (density) of the specified column from two numpy matrices.
    
    Parameters:
        matrix1 (np.ndarray): First numpy matrix.
        matrix2 (np.ndarray): Second numpy matrix.
        column_index (int): Index of the column to compare.
        label1 (str): Label for the first matrix in the plot legend.
        label2 (str): Label for the second matrix in the plot legend.
    """
    # Extract the specific column from both matrices
    col1 = matrix1[:, column_index]
    col2 = matrix2[:, column_index]

    # Plot density
    plt.figure(figsize=(10, 6))
    sns.kdeplot(col1, color="blue", label=label1, linewidth=2)
    sns.kdeplot(col2, color="red", label=label2, linewidth=2)

    plt.title(f'Density Plot of Column {column_index}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_umap_pca(adata, color, random_state = 0):
    sc.pp.neighbors(adata, random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)
    sc.tl.pca(adata, random_state=random_state)
    sc.pl.umap(adata, color=color)
    sc.pl.pca(adata, color=color)

    # Extract variance ratios
    var_ratio = adata.uns['pca']['variance_ratio']
    pcs = np.arange(1, len(var_ratio) + 1)

    # Scree plot
    plt.figure(figsize=(6, 4))
    plt.plot(pcs, var_ratio, marker='o')
    plt.xticks(pcs)  # Set x-axis ticks to integers
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.title('Scree Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.show()