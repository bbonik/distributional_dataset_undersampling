#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:44:36 2019
@author: bbonik

Example script to demonstrate the use of the distributional undersampling
technique. A N-dimensional dataset is randomly created. Different dimensions,
distributions and datapoints can be selected. Then the undersampling function 
is called, in order to create a balanced subset across all given dimensions. 
Different target distributions can be achieved by using the correct input
string.
"""



from scipy import stats
import numpy as np
import random
import matplotlib.pyplot as plt
from distributional_undersampling import undersample_dataset



def generate_random_data(total_data, seed, redundancy=0.1):
    
    '''
    ---------------------------------------------------------------------------
         Function to generate a random dataset with random distribution
    ---------------------------------------------------------------------------
    

    INPUTS
    ------
        total_data: int
            The total number of datapoints (observations) that the generated
            dataset will have.
        seed: int
            Random seed to be passed for controling the random generator.
        redundancy: float in the interval [0,1]
            Percentage of datapoints that will follow a uniform distribution. 
            If for example, redundancy=0.1, 90% of the generated dataset will
            have a random distribution and 10% will follow a uniform 
            distribution. This is done in order to ensure that some datapoints
            will cover all possible range of values. Otherwise, some random 
            distributions may have values concentrated in a smaller range, 
            without covering all the possible values.

        
    OUTPUT
    ------
        data: numpy float array [total_data,]
            Array with datapoints drawn from a random distribution.
    
    '''
    
    # estimate redundant data size
    redundancy_data = round(total_data * redundancy)
    distribution_data = total_data - redundancy_data
    
    # prepare matrices
    data_distr = np.zeros(distribution_data, dtype=float)
    data_redun = np.zeros(redundancy_data, dtype=float)
    
    
    rnd = random.randint(0,6)  # get a random number between 0-6
    
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
    data_distr = ((data_distr - data_distr.min()) / 
                 (data_distr.max() - data_distr.min()))
    
    
    # adding some redundancy data
    data_redun = stats.uniform.rvs(loc=0, 
                                   scale=1, 
                                   size=redundancy_data, 
                                   random_state=seed)
    
    data = np.concatenate((data_distr, data_redun))  # merge the 2 matrices

    
    return data
    
    


def main():

    plt.close('all')
    
    # generating a random dataset of different dimensions and distributions
    
    data_observations = 5000  # change accordingly
    data_dimensions = 5  # change accordingly
    A = np.zeros([data_observations, data_dimensions], dtype=float)
    
    # create a random distribution for each dimension
    for i in range(data_dimensions):
        A[:,i] = generate_random_data(total_data=data_observations, seed=i)
    
    # run the undersmapling optimization function
    indices_to_keep = undersample_dataset(data=A,
                                          data_to_keep=1000, 
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

