# distributional_dataset_undersampling()
A Mixed Integer Linear Programming (MILP) Python function for undersampling a dataset while enforcing a particular distribution across multiple dimensions.



<div style="width: 500px; height: 250px; overflow: hidden;">
  <img src="https://github.com/bbonik/distributional_dataset_undersampling/blob/master/data/example.png" width="800" height="800">
</div>

# Introduction
Datasets can be highly unbalanced: some values/categories may be over-represented, while others may be under-represented. Such imbalance may have a negative impact on many machine learning techniques: the learning algorithm may be very accurate for the over-represented classes, while exhibiting a very high error for the under-represented ones. **Oversampling** (replicating the under-represented classes) or **undersampling** (reducing the over-represented classes) are two typical approaches to address this problem. 

Balancing a dataset across only 1 attribute is straight forward. For exmpale, building a face gender classifier (M/F) from an imbalanced dataset should be easy: just undersample the majority class or oversample the minority class. 

Things get really tricky if more than one attributes are involved. For example, assume that we would like to build the same face gender classifier (M/F), but achieve a balanced performance also across different ages, races and facial expressions. In this case, we have to balance the dataset across 4 attributes (gander, age, race, expressions). Even more, some of these attributes are not categorical, for example age, and require balancing across different age ranges. 

Undersampling across multiple dimensions is a difficult combinatorial problem. For example, a datapoint from a majority attribute, may also be a minotiry in another attribute. It is very difficult to know which datapoints to drop in order to achive a target distribution across all attributes. 

# Description
We introduce a new undersampling MILP-based dataset shaping technique. The proposed optimization leverages on the (possible) redundancies in a large dataset to generate a more compact version of the original dataset with a specified target distribution across each attribute/dimension, while simultaneously minimizing linear correlations among them. 

In summary, given a large dataset and a required target distribution, our MILP optimisation method creates a compact subset of the original dataset by finding the optimal combination of datapoints that:
1. Enforces the target distribution across all dimensions.
2. Minimizes linear correlations between dimensions.
As such, our technique can be seen as complementary to dimensionality reduction: instead of reducing feature dimensions while maintaining the number of observations, we reduce the number of observations while imposing distributional constraints on the dimensions.

The following figure depicts covariance scatter plots for a 6-dimensional dataset with 11K data points. Distribution for each dimension is given by a histogram, while Pearson correlation rho between dimensions and corresponding p-value (in parentheses) are mentioned for each scatter plot. Dimension 6 (D6) is a linear combination of D1 and D4. Three subsets of 1K datapoints are generated with our data shaping technique, so as to have Uniform, Gaussian and Weibull distributions, while minimising correlations between different dimensions.





<img src="https://github.com/bbonik/distributional_dataset_undersampling/blob/master/data/example.png" width="500">
