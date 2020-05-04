# Trees and Ensembles

- [Trees and Ensembles](#trees-and-ensembles)
  - [Decision Trees](#decision-trees)
    - [Classification](#classification)
      - [Impurity functions](#impurity-functions)
    - [Regression](#regression)
  - [Random Forest](#random-forest)
    - [Algorithm](#algorithm)
      - [Training](#training)
      - [Prediction](#prediction)
  - [Gradient Boosting](#gradient-boosting)

For all algorithms here, assume a **dataset** $(X, Y)$ with $n$ observations and $m$ features. For simplicity, assume that all features are categorical.

## Decision Trees

Decision trees are nested decision rules learned from the data features (e.g., `if` $f_{1} > 5$ `then` `if` $f_{2} < 100$  `then` `...`). At prediction time, the observation passes through the tree and the resulting leaf is returned. The more the rules, the deeper the tree and it tends to overfit the data. On the contrary, few rules tend to make it underfit.

There are many algorithms to build decision trees. Scikit-learn 0.22.2 uses CART. We will not cover CART but only the generic algorithm.

### Classification

Let $Q = X$. Let $d \in \mathbb{R}+$ be the maximum depth allowed for the decision tree. Let $i = 0$ a counter. Let $I = \empty$ be an empty list. Assume you know a function $H(Q)$ that measures the impurity $\rho \in \mathbb{R}$ of $Q$.

`build_decision_tree(`$Q, T, d$`)`:
1. `if` $Q = \empty$ `or` $d = 0$:
    1. `return` $T$
2. `else`:
   1. Create a list $S$ of all possible splits of the form $\theta = (f, t_{i})$ where $f$ is a f-th feature and $t_i$ a threshold (remember we are assuming categorical features for simplicity).
   2. For each $s \in S$:
       1. Split $Q$ into $Q_{L} = \{ (x_i, y_i) | x_{if} \leq t_i \}$ and $Q_{R} = \{ (x_i, y_i) | x_{if} > t_i \}$.
       2. Compute the joint impurity $G$ of $Q_{L}$ and $Q_{R}$, which is defined as $G = G(Q, s) = \frac{|Q_{L}|}{|Q|} H(Q_{L}) + \frac{|Q_{R}|}{|Q|} H(Q_{R})$
       3. Add $(s, G)$ to $I$.
   3. Select the pair $(s^*, G^*)$ from $I$ that minimizes the impurity; i.e., $s^* = \argmin_{s \in S} G(Q, s)$
   4. Add $s^*$ to $T$
   5. Call `build_decision_tree(`$Q_L, T, d - 1$`)` and `build_decision_tree(`$Q_R, T, d - 1$`)`

#### Impurity functions

There are several impurity functions. Most of them rely on the frequency per class. Let $k$ be a class in $Y$. Assume you are at a node with the set $Q \subset X$ as observations and their respective target values $Y^Q$. The proportion of class $k$ is:

$\hat{p}_k = \frac{1}{|Q|} \sum_{x_i \in Q} \mathbb{1}\{Y_i^Q = k\}$

Some examples of impurity functions relying on that frequency are:

| Missclasfication error        | Gini-index                                   | Entropy                                             | 
|:-----------------------------:|:--------------------------------------------:|:---------------------------------------------------:|
| $1 - \argmax_{k} \hat{p}_{k}$ | $\sum_{k = 1}^{k} \hat{p}_k (1 - \hat{p}_k)$ | $E(X)= - \sum_{k=1}^{k} \hat{p}_k \log_2 \hat{p}_k$ |


### Regression


## Random Forest

Random forest refers to _a **forest of decision trees** generated from **random data**_. The _forest_ is an example of what it is called an **ensemble**: a set of models that evaluate the dataset individually but combine their output to make a prediction. To make sense, each model has to be different from the others (e.g., in the hyperparameters, rules, depth, etc.).

### Algorithm

#### Training 
Suppose that you have a `decision_tree()` function. The general algorithm applies for both regression and classification, because the particularities rely on the decision tree procedure. Let $n \in \mathbb{R}^+$ be the number of trees in the forest, $d \in \mathbb{R}^+$ the maximum depth of a tree, and $f \in \mathbb{R}^+$ a number satisfying $0 < f < m$. Let $i=0$ be a counter and $F = \empty$, $T = \empty$ lists.

> Remember that $m$ is the number of features in the dataset.

`build_random_forest(`$n, f$`)`:
1. `while` $i < n$:
   1. Create subset $B \subset X$ sampling at random from $X$ with replacement ($B$ is known as a _bootstrap_).
   2. Keep $f$ features from $B$ only (for simplicity, assume selection at random).
   3. T = `decision_treee(`$B, T, d$`)`
   4. Add $T$ to the forest $F$.
2. `return` $F$

> The cardinality of the subset $B$ (step 1.i) is, in principle, the same as $X$'s, however you could try with smaller subsets.

#### Prediction

At prediction time, data is evaluated on every $T \in F$ and each result is added to an array $r$. Then:
- For **classification**: Majority voting decides the outcome. If required, probability per class can be returned with softmax over the frequency per class in $r$.
- For **regression**: Aggregate all values in $r$ (e.g. average)

## Gradient Boosting