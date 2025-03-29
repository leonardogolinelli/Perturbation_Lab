import numpy as np

def produce_perturbed_observations_1gene(
        x,
        mean_weights,
        n_perturbed_samples=30,
        reference_sample_idx=0,
        suppressing_perturbation=True,
        perturbed_gene_idx=None,
        perturbation_increment=1
        ):
    
        min_weight_idx = np.argmin(mean_weights) 
        max_weight_idx = np.argmax(mean_weights)
        min_weight = mean_weights[min_weight_idx]
        max_weight = mean_weights[max_weight_idx]

        print(f"min weight: {min_weight}, max weight: {max_weight}")

        reference_sample = x[reference_sample_idx].copy()
        perturbed_samples = np.tile(reference_sample, (n_perturbed_samples,1))

        if not perturbed_gene_idx:
                perturbed_gene_idx = min_weight_idx if suppressing_perturbation else max_weight_idx
                
        else:
                print(f"A perturbed_gene_idx was provided. The suppressing_perturbation flag will be ignored.")

        perturbed_samples[0][perturbed_gene_idx] = 0
        for i in range(1,n_perturbed_samples):
                perturbed_samples[i][perturbed_gene_idx] = perturbed_samples[i-1][perturbed_gene_idx] + perturbation_increment

        return perturbed_samples, perturbed_gene_idx
