**Warning**: this repo is a proof of concept. Minimal code was used. An environment with torch, numpy, cuda, matplotlib and scanpy installed should be sufficient.
To install scanpy, check this out https://scanpy.readthedocs.io/en/stable/installation.html

The idea of this repo is to illustrate a way of benchmarking of perturbation prediction experiments by using simulated data and artificially creating known, predictable and complex dependencies between different genes. For example, we can force the expression of one gene (here called "mechanicistic" gene) to be a linear or polinomial function of the expression of the other genes. We can tell whether a generative model has learned these dependencies by using perturbed, out-of-distribution observations for which we know a priori the effect on the mechanicistic gene.

**Notebook 1)** Simulate single-cell gene expression data using known ZINB parameters and train a VAE to predict such parameters by minimizing negative ELBO.

**Notebook 2)** Simulate a mechanicistic gene - a gene for which its ZINB distribution depends directly on the expression of the other genes in a quantifiable and predictable way. Train the VAE on all genes (normal independent genes + mechanicistic gene), to learn the generative process underlying the distribution -- especially the dependency between the mechanicistic gene and the other genes.

**Notebook 3)** Simulate multiple perturbations in one (or potentially more) input gene, and predict the effect on the mechanicistic gene using the VAE trained in notebook 2. Compare the curve of predicted vs observed distribution parameters of the mechanicistic gene under the aforementioned perturbations.

**Notebook 4)** Simulate drug effects on the simulated cells and predict a generalizable effect of such perturbation on cells. As a simple example, here I achieved this by computing the mean difference vector between perturbed and unperturbed cells in the latent space (scGen style).
