#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:47:48 2019
@author: Vasileios Vonikakis
"""

from __future__ import print_function
from ortools.linear_solver import pywraplp
from scipy import stats
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb






def undersample_dataset(data,
                        data_to_keep=1000,
                        target_distribution='uniform',
                        bins=10,
                        lamda=0.5,
                        verbose=True,
                        scatterplot_matrix='auto'):
    
    '''
    ---------------------------------------------------------------------------
    Function to undersample a dataset by imposing distributional and 
    correlational constraints across its dimenions. The function runs a mixed 
    integer linear program to estimate which is the optimal combination of a 
    datapoints that results closer to the given target distribution. It then 
    returns the indices of this optimal combination of datapoints to be kept.
    The functions makes use of the ortools mixed integer linear solver.
    ---------------------------------------------------------------------------
    
    DEPENDECES
    ----------
    The function makes use of the ortools library from Google.
    Please install the ortools package as described in the following link:
    https://developers.google.com/optimization/install/

    INPUTS
    ------
        data: numpy.array [N,M]
            Array of datapoints of N observations and M dimensions. All 
            datapoints should be within the interval [0,1].
        target_distribution : 'uniform', 'gaussian', 'weibull', 'triangular'
            String defining the type of distribution to be enforced on the
            resulting subsampled dataset, by selecting the appropriate 
            datapoints that will create this distribution. Selecting 'uniform' 
            will result in a balanced dataset.
        data_to_keep: int
            The number of datapoints to keep from the original dataset, in the
            interval of [1,N].
        bins: int
            The number of bins in which, the dataset will be quantized in order
            to run the integer programming.
        lamda: float
            Number to balance the 2 objectives: distribution vs correlation. 
            lamda=0 implies only distribution contstraints. Lamda>0 enforces
            correlation minimization constraints also.
        verbose: bool
            Wether to show the stages of the proceedure.
        scatterplot_matrix: True / False / 'auto'
            Depict or not a scatterplot matrix of the dataset distributions
            across all dimensions for the input and the undersampled datasets. 
            If 'auto' the scatterplot matrix is depicted only for datasets of 
            10 or less dimensions.
        
    OUTPUT
    ------
        indx_selected: bool numpy.array [N]
            Vector with the same size as the original data observations, with
            True for each observation kept in the downsampled dataset. If no
            solution can be find, a vector with zero entries is returned.
    
    CITATION
    --------
    If you use this code for research puproses please cite the following paper:
    Vonikakis, V., Subramanian, R., Arnfred, J., & Winkler, S. 
    A Probabilistic Approach to People-Centric Photo Selection and Sequencing. 
    IEEE Transactions in Multimedia, 11(19), pp.2609-2624, 2017.
    
    '''
    
    # TODO: make target distribution different adjustable across dimensions. 
    # TODO: Also, wildcard dimensions?
    # TODO: auto normalize input data
    
    #------------------------------------------------------ internal parameters
    
    max_solver_time = 10000  # msec
    data_observations = data.shape[0]  # total number of observations
    data_dimensions = data.shape[1]  # total number of dimensions
    
    if scatterplot_matrix is 'auto':
        if data_dimensions > 10:
            scatterplot_matrix = False
        else:
            scatterplot_matrix = True

    result_status = ['optimal', 
                     'feasible', 
                     'infeasible', 
                     'abnormal', 
                     'not solved', 
                     'unbounded']
    
    plt.style.use('ggplot')

    #--------------------------------------------- defining target distribution

    x = np.arange(1,bins+1) - 0.5
    
    if target_distribution is 'uniform':
        target_pdf = stats.uniform.pdf(x, loc=0, scale=bins)
    elif target_distribution is 'gaussian':
        target_pdf = stats.norm.pdf(x, loc=bins/2, scale=1)
    elif target_distribution is 'weibull':
        target_pdf = stats.weibull_min.pdf(x, c=5, loc=2, scale=1)
    elif target_distribution is 'triangular':
        target_pdf = stats.triang.pdf(x, c=0.75, loc=0, scale=bins)

    #------------------------------------------------ quantizing data into bins
    
    if verbose is True:
        print('\nQuantizing dataset...')
    
    data_quantized = np.digitize(data.copy(),
                                 bins=np.linspace(0,1,bins+1),
                                 right=False)
    data_quantized -= 1
    data_quantized[data_quantized == bins] = bins-1
    
    #------------------------------------- displaying the initial distributions
    
    
    plot_scatter_matrix(data,
                        column_names=None,
                        show_correlation=True,
                        alpha=None,
                        title=('Original dataset (' + 
                               str(data_observations) + 
                               ' datapoints)')
                        )
  
    #-------------------------------------------------------- MILP optimization

    if verbose is True:
        print('Filling problem matrices...')

    solver = pywraplp.Solver('SolveIntegerProblem',
                               pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    solver.set_time_limit(max_solver_time)
    
    #------- constructing the data for correlation minimization (2nd objective)
    
    #estimating the final distribution in each dimension
    avg=np.dot(((np.arange(1,bins+1) - 0.5)/bins), target_pdf)
    
    qq = np.zeros([data_observations, int(comb(data_dimensions,2))],
                   dtype=float)
    v = np.zeros([data_observations], dtype=float)
    
    for k in range(data_observations):
        kk=0
        for i in range(data_dimensions-1):  
            for j in range(i+1, data_dimensions):
    
                qq[k,kk] = np.abs(data[k,i] - avg) * np.abs(data[k,j] - avg)
    
                kk=kk+1
    
        v[k] = qq[k,:].sum()

    #---------- constructing the data distribution minimization (1st objective)
    
    # constructing the bin matrix
    
    B=[]
    
    for j in range(data_quantized.shape[1]):  # accross all dimensions
      
        b = np.zeros([bins, data_quantized.shape[0]], dtype=bool)
    
        for i in range(data_quantized.shape[0]): # accross all observations
            
            b[data_quantized[i,j], i] = True
    
        B.append(b)   

    # Objective function
    f = lamda*v #2nd objective: minimization of correlation
    ff = np.ones(data_dimensions*bins, dtype=float) #indexes of slack variables
    c = np.hstack([f, ff])

    
    # creating problem variables
    x = {}
    for i in range(c.shape[0]):
        if i < f.shape[0]:
            x[i] = solver.BoolVar('x[%i]' % i)  # binary selection variables
        else:
            x[i] = solver.NumVar(-solver.infinity(), 
                                 solver.infinity(), 
                                 'x[%i]' % float(i))  # real slack variables
    
    # define the objective function
    solver.Minimize(solver.Sum([c[i] * x[i] for i in range(c.shape[0])]))
    

    # constraints
    
    
    # equality constraint: strictly data_to_keep datapoints should be selected
    q = np.hstack([np.ones(data_observations), np.zeros(data_dimensions*bins)])
    solver.Add(solver.Sum(q[j]*x[j] for j in range(len(x))) == data_to_keep)
    
    total_constraints = data_dimensions * bins
    
    if verbose is True:
        print('Adding contstraints [0%]\r', end='')
    
    # distribution constraints
    k=0
    progress=0
    for m in range(data_dimensions):   #accross all dimensions
        
        ATR = B[m].astype(dtype=float)
        
        for n in range(bins):  #across all quantization bins
    
            
            b = np.ceil(target_pdf[n] * data_to_keep)
    
            a = np.zeros(data_dimensions*bins, dtype=float)
            z = m * bins + n #2D to 1D
            a[z] = -1
    
            # upper slack bound
            A = np.hstack( [ATR[n,:], a] )
            solver.Add(solver.Sum([A[j]*x[j] for j in range(A.shape[0])]) <= b)
    
            # lower slack bound
            A = np.hstack( [-ATR[n,:], a] )
            solver.Add(solver.Sum([A[j]*x[j] for j in range(A.shape[0])]) <= -b)
            
            k += 1
            progress = round((k*100)/total_constraints)
            
            if verbose is True:
                if progress >= 10:
                    print('\b\b\b\b\b[%d%%]\r'%progress, end='')
                else:
                    print('\b\b\b\b[%d%%]\r'%progress, end='')
            
      
    if verbose is True:
        print('\nNumber of variables =', solver.NumVariables())
        print('Number of constraints =', solver.NumConstraints())
    
    
    #----------------------------------------------------- solving optimization
    
    if verbose is True:
        print('Solving...')
    
    result_status_code = solver.Solve()  # solve problem
    
    if verbose is True:
        print('Result status =', result_status[result_status_code])
        print('Total cost = ', solver.Objective().Value())
        print('Problem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
        print()

    indx_selected = np.zeros(len(x), dtype=bool)
    
    for i in range(len(x)):
        if x[i].solution_value() > 0:
            indx_selected[i] = True
    
    #spliting vector x into the slack variables and selection variables
    xslack = indx_selected[data_observations:]  # slack variables
    indx_selected = indx_selected[:data_observations]  # selection variables
    indx_selected = indx_selected.astype(bool)
     
    if indx_selected.sum()>0:
        
        if scatterplot_matrix is True:
            
            
            plot_scatter_matrix(data[indx_selected,:],
                                column_names=None,
                                show_correlation=True,
                                alpha=None,
                                title=('Undersampled dataset (' + 
                                       str(indx_selected.sum()) + 
                                       ' datapoints) - ' +
                                       target_distribution)
                                )
                               
            plot_scatter_matrix(data_quantized[indx_selected,:],
                                column_names=None,
                                show_correlation=True,
                                alpha=None,
                                title=('Undersampled dataset quantized (' + 
                                       str(indx_selected.sum()) + 
                                       ' datapoints) - ' +
                                       target_distribution)
                                )

        
    else:
        if verbose is True:
            print('No solution was found')
    
    
    return indx_selected




def plot_scatter_matrix(data,
                       column_names=None,
                       show_correlation=True,
                       alpha=None,
                       title=None):
    
    '''
    ---------------------------------------------------------------------------
         Function to plot a customized scatterplot matrix (based on Pandas)
    ---------------------------------------------------------------------------
    

    INPUTS
    ------
        data: numpy.array [N,M]
            Array of datapoints of N observations and M dimensions.
        column_names: list of strings [M] or None
            List of strings containing the names of each data dimension. If 
            None, then simple dimension labels will be auto generated.
        show_correlation: boolean
            Whether to depict the Pearson correlation coefficient for each pair
            of dimensions on the upper triangle of the scatterplot matrix. 
        alpha: float in [0,1] or None
            The transparency level of each datapoint. 0 = totally transparent,
            1 = totally opaque. If None, then value is automatically 
            adjusted in order to be more transparent for large datasets and 
            less transparent for smaller datasets. This is done because for 
            very large datasets there is lots of overlapping between datapoints
            and it is very difficult to understand the underlying distribution.
        title: string
            The title that will be displayed on the scatterplot matrix.
        
    OUTPUT
    ------
        Plots a customized scatterplot matrix of the input data array.
    
    '''
    
    # define names for each dimension
    if column_names is None:
        column_names = ['D'+str(i) for i in range(data.shape[1])]
    
    # auto set of alpha according to dataset size in the interval [0.1,0.7]
    if alpha is None:   
        alpha = (5000 - data.shape[0]) / 5000
        if alpha > 0.7: alpha = 0.7
        elif alpha < 0.1: alpha = 0.1
        
    # create a dataframe and plot the basic scatterplot matrix
    df_A = pd.DataFrame(data, columns=column_names)
    axes = pd.plotting.scatter_matrix(df_A, 
                                      alpha=alpha, 
                                      figsize=(8, 8), 
                                      diagonal='hist')
    
    # plot Pearson correlation coefficient for pairs of dimensions
    if show_correlation is True:
        corr = df_A.corr().as_matrix()
        for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
            axes[i, j].annotate("r=%.3f" %corr[i,j], 
                (0.7, 0.9), 
                xycoords='axes fraction', 
                ha='center', 
                va='center')
    
    # add title     
    if title is not None:        
        plt.suptitle(title)
        
    plt.show()



    
    