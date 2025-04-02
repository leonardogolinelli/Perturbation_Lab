**Warning**: this repo is a proof of concept. Minimal code was used. An environment with torch, numpy, cuda, matplotlib and scanpy installed should be sufficient.
To install scanpy, check this out https://scanpy.readthedocs.io/en/stable/installation.html

The idea of this repo is to illustrate a way of benchmarking of perturbation prediction experiments by using simulated data and artificially creating known, predictable and complex dependencies between different genes. For example, we can force the expression of one gene (here called "mechanicistic" gene) to be e.g. a linear or polinomial function of the expression of the other genes. We can tell whether a generative model has learned these dependencies by using perturbed, out-of-distribution observations for which we know a priori the effect on the mechanicistic gene. I have no idea whether this approach is already used \ to which extent in the literature. If it's already used -- well -- it was a good excercise.

This repo also includes useful functions that you might want to use yourself or extend. 

**Notebook 1)** Simulate single-cell gene expression data using known ZINB parameters and train a VAE to predict such parameters by minimizing negative ELBO.

**Notebook 2)** Simulate a mechanicistic gene - a gene for which its ZINB distribution depends directly on the expression of the other genes in a quantifiable and predictable way. Train the VAE on all genes (normal independent genes + mechanicistic gene), to learn the generative process underlying the distribution -- especially the dependency between the mechanicistic gene and the other genes.

**Notebook 3)** Simulate multiple perturbations in one (or potentially more) input gene, and predict the effect on the mechanicistic gene using the VAE trained in notebook 2. Compare the curve of predicted vs observed distribution parameters of the mechanicistic gene under the aforementioned perturbations. Quantitative metrics (e.g. MSE between predicted and observed parameters) are obviously recommended as well.

**Notebook 4)** Simulate drug effects on the simulated cells and predict a generalizable effect of such perturbation on cells. As a simple example, here I achieved this by computing the mean difference vector between perturbed and unperturbed cells in the latent space (scGen style).

In the code, you will find these functions and objects, which may be useful to you to speed up simulations or custom implementation.

1) _base_genes_ function in dataset.py to simulate cells and statistically independent genes, each with its own mean and dispersion values, and with a desired % sparsity. The parameters of each gene are sampled from a uniform distribution to ensure gene diversity. Notice that, to simulate more than one cell population \ cell type, we can simply use the same function twice with different parameters, and concatenate the results. The idea behind this is that within an individual cell population, cells have similar gene distributions.

4) _get_mech_dataset_ function in dataset.py that takes the simulated genes in input, and outputs the same dataset in input with an extra gene or column, called mechanicistic gene, for which the mean (and potentially the dispersion) are linear combinations of the expression of the other genes. Instead of using linear combinations, one could use other functions to simulate more complex effects.

5) produce_perturbed_observations_1gene function in perturb.py takes the simulated gene expression data (base genes + mechanicistic gene) in input, and outputs a desired number of perturbed samples, where a desired base gene is perturbed from 0 (suppressed) up to a value equal to the number of perturbed samples.

6) simple VAE in model.py, to learn the underlying distribution of all genes

7) simple VAE training code in training.py
