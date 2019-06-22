#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:44:36 2019

@author: bbonik
"""

from scipy import stats
import numpy as np
import random
import matplotlib.pyplot as plt
from distributional_undersampling import undersample_dataset


def generate_random_data(total_data, seed, redundancy=0.1):
    

    redundancy_data = round(total_data * redundancy)
    distribution_data = total_data - redundancy_data
    
    data_distr = np.zeros(distribution_data, dtype=float)
    data_redun = np.zeros(redundancy_data, dtype=float)
    
    
    rnd = random.randint(0,6)
    
    if rnd == 0:
        data_distr = stats.norm.rvs(loc=0, 
                                    scale=1, 
                                    size=distribution_data, 
                                    random_state=seed)
    elif rnd == 1:
        data_distr = stats.genpareto.rvs(c=-random.uniform(0,1), 
                                         loc=0, scale=1, 
                                         size=distribution_data, 
                                         random_state=seed)
    elif rnd == 2:
        data_distr = stats.triang.rvs(c=random.uniform(0,1), 
                                      loc=0, 
                                      scale=1, 
                                      size=distribution_data, 
                                      random_state=seed)
    elif rnd == 3:
        data_distr = stats.anglit.rvs(loc=0, 
                                      scale=1, 
                                      size=distribution_data, 
                                      random_state=seed)
    elif rnd == 4:
        data_distr = stats.nakagami.rvs(nu=random.uniform(0.1,5), 
                                        loc=0, 
                                        scale=1, 
                                        size=distribution_data, 
                                        random_state=seed)
    elif rnd == 5:
        data_distr = stats.arcsine.rvs(loc=0, 
                                       scale=1, 
                                       size=distribution_data, 
                                       random_state=seed)
    elif rnd == 6:
        data_distr = stats.argus.rvs(chi=random.uniform(0.1,5), 
                                     loc=0, 
                                     scale=1, 
                                     size=distribution_data, 
                                     random_state=seed)
    
    
    # min max normalization
    data_distr = (data_distr - data_distr.min()) / (data_distr.max() - data_distr.min())
    
    
    # adding some redundancy data
    data_redun = stats.uniform.rvs(loc=0, 
                                   scale=1, 
                                   size=redundancy_data, 
                                   random_state=seed)
    
    data = np.concatenate((data_distr, data_redun))

    
    return data
    
    


def main():

    plt.close('all')
    

    random_seed = 2
    
    # generating random dataset of 10 dimensions and different distributions
    
    data_observations = 3500
    data_dimensions = 4
    A = np.zeros([data_observations, data_dimensions], dtype=float)
    
    
    
    for i in range(data_dimensions):
        A[:,i] = generate_random_data(total_data=data_observations, seed=i)
    
    

    
    indices_to_keep = undersample_dataset(data=A,
                                          data_to_keep=200,
                                          target_distribution='uniform',
                                          bins=10,
                                          lamda=0.5,
                                          verbose=True,
                                          scatterplot_matrix=True)
    
    
    A_undersampled = A[indices_to_keep]
    
    print ('Original dataset size:', str(A.shape))
    print ('Undersampled dataset size:', str(A_undersampled.shape))



if __name__ == '__main__':
  main()

