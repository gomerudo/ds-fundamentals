# Bayesian Learning

- [Bayesian Learning](#bayesian-learning)
  - [The Bayes theorem](#the-bayes-theorem)
  - [Classification](#classification)
    - [Vanilla Naive Bayes](#vanilla-naive-bayes)
      - [Training](#training)
      - [Prediction](#prediction)
    - [Gaussian Naive Bayes](#gaussian-naive-bayes)
      - [Training](#training-1)
      - [Prediction](#prediction-1)
    - [Bernoulli Naive Bayes](#bernoulli-naive-bayes)
      - [Training](#training-2)
      - [Prediction](#prediction-2)
  - [Regression](#regression)
    - [Gaussian Processes](#gaussian-processes)

## The Bayes theorem

Assume a **dataset** $(X, Y)$ with $n$ observations and $m$ features. Knowing this dataset, the Bayes theorem allows us to compute the probability of any observation $x$ (not necessarely in $X$) having a target value $y$. Such probability is given by the so-called **Bayes rule**:

$P(y \mid x ) = \frac{P(y) P(x \mid y)}{ P(x) }$

The elements of the theorem are:
- **The priors**. Priors are probabilities that we know _a priori_; i.e., knowledge that we do not have to infer because it is already there. For the Bayes theorem we have two priors:
  - **The predictor's prior probability $P(x)$**. It is the probability of the observation $x_i$ happening. We can compute it from the dataset.
  - **The target's prior probability $P(y)$**. Similary, we now the probability of any target value $y$ in the dataset.
- **The likelihood $P(x \mid y)$ of the features**. It can be estimated from data too.
- **The posterior probability $P(y \mid x )$**. We do not know it beforehand, but we can derive after (thus posterior) we compute the priors.

Keep in mind that $x$ is a vector of $m$ features so that $x = (x_1, x_2, ... , x_m)$, and $P(x)$ is the joint probability of its features: $P(x_1, x_2, ... , x_m)$. This holds for the likelihood $P(x \mid y)$ too.

Putting all together, **the Bayes rule in our setting is**:

$
P(y \mid x_{1}, x_{2}, ... , x_{m} ) = \frac{P(y) P(x_{1}, x_{2}, ... , x_{m} \mid y)}{ P(x_{1}, x_{2}, ... , x_{m}) }
$

## Classification

### Vanilla Naive Bayes

This classifier is named in this way because it relies on a _naive_ (but strong) assumption: **all features $f_j$ are independent**, meaning that:
- $P(x \mid y) = P(x_{1}\mid y) \times P(x_{2}\mid y) \times ... \times P(x_{m}\mid y) = \prod_{j=1}^{m} P(x_j \mid y)$
- Similarly, $P(x) = \prod_{j=1}^{m} P(x_j)$.

This results in the next Bayes rule:

$
P(y \mid x_{1}, x_{2}, ... , x_{m} ) = \frac{P(y) \prod_{j=1}^{m} P(x_j \mid y) }{ \prod_{j=1}^{m} P(x_j) }
$

> The main assumption does not hold in many cases, thus it should be treated carefully.

This vanilla classifier assumes that all data is categorical.

#### Training

Training a Naive Bayes classifier is just computing a likelihood table per feature (because they are assumed independent) that will contain information of the priors. Take for example the next dataset $X_{\text{e.g.}}$.

<table>

<th>Dataset</th>
<th>Likelihood table</th>

<tr>

<td>

| Person (observation) | Weather ($f_1$) | Played golf ($Y$) |
|:--------------------:|:---------------:|:-----------------:|
| $x_1$                | Sunny           | Yes               |
| $x_2$                | Sunny           | Yes               |
| $x_3$                | Sunny           | Yes               |
| $x_4$                | Sunny           | No                |
| $x_5$                | Sunny           | No                |
| $x_6$                | Overcast        | Yes               |
| $x_7$                | Overcast        | Yes               |
| $x_8$                | Overcast                | Yes               |
| $x_9$                | Overcast                | Yes               |
| $x_{10}$             | Rainy                   | Yes               |
| $x_{11}$             | Rainy                   | Yes               |
| $x_{12}$             | Rainy                   | No                |
| $x_{13}$             | Rainy                   | No                |
| $x_{14}$             | Rainy                   | No                |

</td>
<td>

| $f_1$          | $y$ = Yes | $y$ = No | Proportion |
|:--------------:|:---------:|:--------:|:----------:|
|   Sunny        |      3/9  |  2/5     | **5/14**   |
|  Overcast      |      4/9  |  0/5     | **4/14**   |
|   Rainy        |      2/9  |  3/5     | **5/14**   |
| **Proportion** | **9/14**  | **5/14** |            |

</td>
</tr>
</table>

#### Prediction

Prediction is done directly with the Bayes rule. We have to predict the probability for each of the classes. The final prediction is the class with the highest probability, i.e.:

$
\hat{y} = \argmax_{y} \frac{P(y) \prod_{j=1}^{m} P(x_j \mid y) }{ \prod_{j=1}^{m} P(x_j) }
$

> It is possible to make one small improvement by considering that the denominator is the same for all classes (i.e., constant). Thus, the predictor is: $\hat{y} = \argmax_{y} P(y) \prod_{j=1}^{m} P(x_j \mid y)$

You should take into consideration that Naive Bayes classifiers are known to be bad estimators, and thus the output probabilities are usually not to be considered.

### Gaussian Naive Bayes

Gaussian Naive Bayes is a modification of the vanilla classifier to make it suitable for numerical data. As most of the subfamily of Naive Bayes classifiers, the only difference is the way the likelihood of the features $P(x \mid y)$ is estimated. As the name suggests, this classifiers uses a Gaussian distribution:

$
P(x \mid y) = P(x_1, x_2, ... , x_m \mid y) = \prod_{j=1}^{j=m} \frac{1}{\sqrt{2 \pi \sigma_y^2}} \exp(- \frac{(x_j - \mu_y)^2}{2 \sigma_y^2})
$

#### Training

Consists of estimating the median $\mu_y$ and the variance $\sigma_y^2$ per class $y$.

#### Prediction

Same as in the vanilla Naive Bayes, but with the gaussian likelihood.

### Bernoulli Naive Bayes

This classifier assumes features in a binary format, and thus the likelihood of features $P(x \mid y)$ is modeled with a Bernoulli distribution as follows:

$
P(x \mid y) = P(x_1, x_2, ... , x_m \mid y) = \prod_{j=1}^{j=m} P(j \mid y)x_j + (1 - P(i \mid y))( 1 - x_j)
$

Here, $P(j \mid y)$ must be read as _the probability of feature $i$ ocurring (i.e., having a value of 1) when class y "occurs"_. Since the features are binary, then $x_j$ and $(1 - x_j)$ are just flags to either consider $P(j \mid y)$ or its complement.

#### Training

Consists of computing the probabilities $P(i \mid j)$ per feature.

#### Prediction

Same as in the vanilla Naive Bayes, but with the Bernoulli likelihood.
<!-- ### Multimonial Naive Bayes -->

## Regression

### Gaussian Processes

TODO