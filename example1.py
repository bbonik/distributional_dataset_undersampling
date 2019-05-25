#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:44:36 2019

@author: bbonik
"""

import scipy.io
import matplotlib.pyplot as plt
from distributional_undersampling import undersample_dataset




def main():

    plt.close('all')
    
    # loading precomputed 6-dimensional data
    data = scipy.io.loadmat('data/DATA_random_6D.mat')['A']
  
    indices_to_keep = undersample_dataset(data=data,
                                          data_to_keep=10,
                                          target_distribution='uniform',
                                          bins=10,
                                          lamda=0.5,
                                          verbose=True,
                                          scatterplot_matrix='auto')
    
    data_undersampled = data[indices_to_keep]
    
    print ('Original dataset size:', str(data.shape))
    print ('Undersampled dataset size:', str(data_undersampled.shape))
    



if __name__ == '__main__':
  main()

