#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 09:50:36 2019
@author: bbonik

Simple script to explore different free datasets for distributional 
undersampling
"""
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from distributional_undersampling import undersample_dataset
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes



plt.close('all')
plt.style.use('ggplot')


X=load_diabetes(return_X_y=False)
#X=load_iris(return_X_y=False)
#X=load_breast_cancer(return_X_y=False)

data = X.data
df_A=pd.DataFrame(data, columns=X.feature_names)
axes = pd.plotting.scatter_matrix(df_A, alpha=0.5, figsize=(8, 8), diagonal='hist')
corr = df_A.corr().as_matrix()
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("r=%.3f" %corr[i,j], (0.7, 0.9), xycoords='axes fraction', ha='center', va='center')
plt.suptitle('Original Dataset')
plt.show()



indices_to_keep = undersample_dataset(data=data,
                                      data_to_keep=20,
                                      target_distribution='uniform',
                                      bins=10,
                                      lamda=0.5,
                                      verbose=True,
                                      scatterplot_matrix='auto')