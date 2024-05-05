import random
import numpy as np

def bootstrap_conf_interval(data, num_samples, conf_interval):
    # Perform bootstrap sampling
    bootstrap_samples = [random.choices(data, k=len(data)) for _ in range(num_samples)]
    
    # Calculate the means of the bootstrap samples
    sample_means = sorted(np.mean(sample) for sample in bootstrap_samples)
    
    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = np.percentile(sample_means, ((1-conf_interval)/2) * 100)
    upper_bound = np.percentile(sample_means, (1 - (1-conf_interval)/2) * 100)
    
    return round(lower_bound, 1), round(upper_bound, 1)
