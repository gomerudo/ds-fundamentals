# K Nearest Neighbors

- [K Nearest Neighbors](#k-nearest-neighbors)
  - [The algorithm (classification and regression)](#the-algorithm-classification-and-regression)
    - [Training](#training)
    - [Prediction](#prediction)
  - [Notes](#notes)

A simple but powerful method for classification or regression is KNN (K Nearest Neighbors). **The idea behind KNN is that points that are close to each other** (i.e., neighbors) **should have similar target values**. The criterion to consider an observation as _neighbor_ depends on _distances_. All details are described next.

## The algorithm (classification and regression)

Assume a **dataset** $(X, Y)$ with $n$ observations and $m$ features. Let $D$ be a distance function defined on $\mathbb{R}^m$, and $k \in \mathbb{R}$ the number of neighbors to consider. Assume an observation $x_j=(x_{j0}, x_{j1}, ... , x_{jm}) \notin X$ (i.e., a previously unseen observation).

### Training 

1. Store the dataset (i.e., no training at all).

### Prediction

For a new observation $x_j$, the prediction is as follows:
1. Compute $D(x_i, x_j)$ $\forall$ $x_i \in X$ and keep the $k$ observations with the smallest distances to $x_j$
2. Store the target value of the $k$ observations in a set $C$.
3. Return the result as follows:
   1. For **classification**: The class $\hat{y_j}$ of $x_j$ is the majority class in $C$.
   2. For  **regression**:  The value is an aggregate (e.g., average) of the values in $C$.

## Notes

- KNN is a bad algorithm for high-dimensional datasets, because of the curse of dimensionality (i.e., distances become too big - and thus meaningless - in high dimensions)
- Considering few neighbors results in overfitting (e.g., for $k=1$, the prediction for new points will depend on a single point everytime).
- Considering many neighbors results in underfitting (e.g., for $k=n$, the decision will always be the average of the whole dataset $X$).