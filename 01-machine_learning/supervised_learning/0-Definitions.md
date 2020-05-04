# Definitions for Supervised Learning


## Dataset

A **dataset** is a pair of matrices ($X, Y$) . $X \in \mathbb{R}^{n} \times \mathbb{R}^{m}$ is a matrix of $n$ **observations** (the rows) characterized by $m$ **features** (the columns). An **observation** $x_i \in X$ is a vector (the row) $x_i = (x_{i1}, x_{i2}, ... , x_{im})$. $Y \in \mathbb{R}^{n}$ is a matrix of $n$ **target values** (a.k.a. **labels**) associated to each $x_i$; i.e., for any $1 \leq i \leq n$, there is a pair ($x_i, y_i$).

### Example

|       | Feature $f_1$ | Feature $f_2$ | Feature $f_3$   | Target values |
|:-----:|:-------------:|:-------------:|:---------------:|:-------------:|
| $x_1$ | $x_{11} = 2$  | $x_{12} = 15$ | $x_{13} = 73.8$ | $y_{0} = 10$  |
| $x_2$ | $x_{21} = 1$  | $x_{22} = 13$ | $x_{23} = 65.2$ | $y_{1} = 8$   |
| $x_3$ | $x_{31} = 3$  | $x_{32} = 14$ | $x_{33} = 70.1$ | $y_{2} = 9$   |
| $x_4$ | $x_{41} = 5$  | $x_{42} = 15$ | $x_{43} = 75.3$ | $y_{3} = 10$  |
| $x_5$ | $x_{51} = 4$  | $x_{52} = 16$ | $x_{53} = 76.0$ | $y_{4} = 6$   |

$m=3$ (Number of features) ; $n=5$ (Number of observations)

## Supervised Learning

Supervised Learnring is a subarea of Machine Learning that deals with the problem of capturing the relation between observations and target values. In supervised learning, a **learner** is **trained** for that purpose. The learner can be seen as a function $L: (X, Y) \mapsto Y' \in \mathbb{R}^n$. The codomain is known as _the prediction set_ $Y'$. 

The term supervision is given because we know the expected output (predictions) of $L$, and thus we can **supervise** whether or not the output is correct.

 <!-- ## Train and test datasets

Typically, there exist at least two datasets: $(X_{train}, Y_{train})$ and $(X_{test}, Y_{test})$. They have $k$, and $l$ observations, respectively and should satisfy $X_{train} \cap X_{test} = \empty$.

The idea of this datasets

> When explaining the models we will not use the train/test datasets. However, this definition is necessary to understand supervised learning. -->

<!-- The supervision happens after the training of the learner, when we **evaluate** its performance on the dataset using a function $R: (Y, Y') \mapsto \mathbb{R}$. Without labels, the evaluation (and thus, the supervision) cannot happen. -->