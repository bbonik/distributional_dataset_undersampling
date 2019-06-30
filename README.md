# distributional_dataset_undersampling
Python function to undersample a dataset while enforcing a particular distribution across its dimensions

# Introduction
Datasets can be highly unbalanced: some values/categories may be over-represented, while others may be under-represented. Such imbalance may have a negative impact on many machine learning techniques: the learning algorithm may be very accurate for the over-represented classes, while exhibiting a very high error for the under-represented ones. **Oversampling** (replicating the under-represented classes) or **undersampling** (reducing the over-represented classes) are two typical approaches to address this problem. 

When the main attributes of the dataset


We introduce a new undersampling MILP-based dataset shaping technique. The proposed optimization leverages on the (possible) redundancies in a large dataset to generate a more compact version of the original dataset with a specified target distribution across each dimension, while simultaneously minimizing linear correlations among dimensions. 

In summary, given a large dataset and a required target distribution, our MILP optimisation method creates a compact subset of the original dataset by finding the optimal combination of datapoints that:
Enforces the target distribution across all dimensions.
Minimizes linear correlations between dimensions.
As such, our technique can be seen as complementary to dimensionality reduction: instead of reducing feature dimensions while maintaining the number of observations, we reduce the number of observations while imposing distributional constraints on the dimensions.

The following figure depicts covariance scatter plots for a 6-dimensional dataset with 11K data points. Distribution for each dimension is given by a histogram, while Pearson correlation rho between dimensions and corresponding p-value (in parentheses) are mentioned for each scatter plot. Dimension 6 (D6) is a linear combination of D1 and D4. Three subsets of 1K datapoints are generated with our data shaping technique, so as to have Uniform, Gaussian and Weibull distributions, while minimising correlations between different dimensions.




[[https://github.com/bbonik/distributional_dataset_undersampling/tree/master/data/example.png]]
