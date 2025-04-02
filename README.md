Notebook 1) Simulate gene expression data using known ZINB parameters and train a VAE to predict such parameters by minimizing ELBO.

Notebook 2) Generate a mechanicistic gene - a gene for which its ZINB distribution depends directly on the expression of the other genes in a quantifiable and predictable way. Train the VAE on all genes (normal genes + mechanicistic gene).

Notebook 3) Simulate multiple perturbations in one (or potentially more) input gene, and predict the effect on the mechanicistic gene using the VAE trained in notebook 2. Compare the curve of predicted vs observed distribution parameters of the mechanicistic gene under the aforementioned perturbations.

Notebook 4) Simulate drug effects on the simulated cells and predict a generalizable effect of such perturbation on cells, by computing the mean difference vector between perturbed and unperturbed cells in the latent space (scGen style).
