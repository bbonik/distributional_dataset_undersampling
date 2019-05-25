#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:44:36 2019

@author: bbonik
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from distributional_undersampling import undersample_dataset







def main():

    plt.close('all')
    

    random_seed = 2
    
    # generating random dataset of 10 dimensions and different distributions
    
    data_observations=10000
    
    A=np.zeros([data_observations+1000,10], dtype=float)
    
    q=stats.norm.rvs(loc=0, scale=1, size=data_observations, random_state=0)
    q=(q-q.min())/q.max()
    A[:data_observations,0]=q
    
    q=stats.genpareto.rvs(c=-0.5, loc=0, scale=1, size=data_observations, random_state=1)
    q=(q-q.min())/q.max()
    A[:data_observations,1]=q
    
    q=stats.triang.rvs(c=0.5, loc=0, scale=1, size=data_observations, random_state=2)
    q=(q-q.min())/q.max()
    A[:data_observations,2]=q
    
    q=stats.uniform.rvs(loc=0, scale=1, size=data_observations, random_state=3)
    q=(q-q.min())/q.max()
    A[:data_observations,3]=q
    
    q=stats.nakagami.rvs(nu=1, loc=0, scale=1, size=data_observations, random_state=4)
    q=(q-q.min())/q.max()
    A[:data_observations,4]=q
    
    A[:data_observations,5] = (A[:data_observations,1] + A[:data_observations,3])/2
    
    q=stats.norm.rvs(loc=0, scale=1, size=data_observations, random_state=5)
    q=(q-q.min())/q.max()
    A[:data_observations,6]=q
    
    q=stats.genpareto.rvs(c=-0.5, loc=0, scale=1, size=data_observations, random_state=6)
    q=(q-q.min())/q.max()
    A[:data_observations,7]=q
    
    q=stats.triang.rvs(c=0.5, loc=0, scale=1, size=data_observations, random_state=7)
    q=(q-q.min())/q.max()
    A[:data_observations,8]=q
    
    q=stats.nakagami.rvs(nu=1, loc=0, scale=1, size=data_observations, random_state=8)
    q=(q-q.min())/q.max()
    A[:data_observations,9]=q
    
    
    for i in range(10):
        
        q=stats.uniform.rvs(loc=0, scale=1, size=1000, random_state=i+10)
        q=(q-q.min())/q.max()
        A[data_observations:,i]=q

    
    
    indices_to_keep = undersample_dataset(data=A,
                                          data_to_keep=1000,
                                          target_distribution='uniform',
                                          bins=10,
                                          lamda=0.5,
                                          verbose=True,
                                          scatterplot_matrix='auto')
    
    
    A_undersampled = A[indices_to_keep]
    
    print ('Original dataset size:', str(A.shape))
    print ('Undersampled dataset size:', str(A_undersampled.shape))



if __name__ == '__main__':
  main()

