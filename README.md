# distributional_undersampling()
A Mixed Integer Linear Programming (**MILP**) Python function for **undersampling a dataset** while enforcing a particular **target distribution** across multiple dimensions. The function leverages on the (possible) **redundancies** in a large dataset to generate a more **compact** version of it with a specified target distribution across each attribute/dimension, while simultaneously minimizing linear correlations among them. 

<img src="https://github.com/bbonik/distributional_dataset_undersampling/blob/master/data/example.png" width="900" height="900">

# Introduction
Datasets can be highly unbalanced: some values/categories may be over-represented, while others may be under-represented. Such imbalance may have a negative impact on many machine learning techniques: the learning algorithm may be very accurate for the over-represented classes, while exhibiting a very high error for the under-represented ones. **Oversampling** (replicating the under-represented classes) or **undersampling** (reducing the over-represented classes) are two typical approaches to address this problem. 

**Balancing a dataset across only 1 attribute is straight forward**. For exmpale, building a face gender classifier (M/F) from an imbalanced dataset should be easy: just undersample the majority class or oversample the minority class. 

**Things get really tricky if more than one attributes are involved**. For example, assume that we would like to build the same face gender classifier (M/F), but also achieve a balanced performance across different ages, races and facial expressions. In this case, we have to balance the dataset across 4 attributes (gander, age, race, expressions). Even more, some of these attributes are not categorical, for example age, requiring balancing across different age ranges. 

**Undersampling across multiple dimensions is a difficult combinatorial problem**. A datapoint may be majority for attribute A, but minority for attribute B. In the previous example, assume that Male training examples are over-represented, but age ranging from 10-20 years is under-represented. Should you delete a Male datapoint of age 10-20? The answer is not straight forward. It is very difficult to know which datapoints to drop in order to achive a target distribution *across all attributes*. 

# Description
Here, a new **undersampling MILP-based dataset shaping** technique is introduced. The proposed optimization leverages on the (possible) *redundancies* in a large dataset to generate a more *compact* version of the original dataset with a specified target distribution across each attribute/dimension, while simultaneously minimizing linear correlations among them. 

In summary, given a large dataset and a required target distribution, the MILP optimisation method creates a compact subset of the original dataset by finding the optimal combination of datapoints that:
1. **Enforces the target distribution across all dimensions**.
2. **Minimizes linear correlations between dimensions**.
As such, this technique can be seen as *complementary to dimensionality reduction*: instead of reducing feature dimensions while maintaining the number of observations, we reduce the number of observations while imposing distributional constraints on the dimensions.

The above figure depicts covariance scatter plots for a 6-dimensional dataset with 11K data points. Distribution for each dimension is given by a histogram, while Pearson correlation rho between dimensions and corresponding p-value (in parentheses) are mentioned for each scatter plot. Dimension 5 (D5) is a linear combination of D0 and D3. Three subsets of 1K datapoints are generated with our data shaping technique, so as to have Uniform, Gaussian and Triangular distributions, while minimising correlations between different dimensions.

# Dependences
The function makes use of the **ortools** library from Google. Please install the ortools package as described in the following link: https://developers.google.com/optimization/install/

# Citations
If you use this code in your research please cite the following papers:   
1. [Vonikakis, V., Subramanian, R., Arnfred, J., & Winkler, S. A Probabilistic Approach to People-Centric Photo Selection and Sequencing.  IEEE Transactions in Multimedia, 11(19), pp.2609-2624, 2017.](https://www.researchgate.net/publication/316569587_A_Probabilistic_Approach_to_People-Centric_Photo_Selection_and_Sequencing)
2. [V. Vonikakis, R. Subramanian, S. Winkler. Shaping Datasets: Optimal Data Selection for Specific Target Distributions. Proc. ICIP2016, Phoenix, USA, Sept. 25-28, 2016.](http://vintage.winklerbros.net/Publications/icip2016a.pdf)
