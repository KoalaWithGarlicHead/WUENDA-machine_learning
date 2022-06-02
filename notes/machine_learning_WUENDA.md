# Introduction

Field of study that gives computers the ability to learn without being explicitly programmed.

## Supervised Learning

GIve the dataset with the "right answers".

* Classification Problem: discrete valued output (0, 1, 2, ...)
* Regression: Predict continous valued output

## Unsupervised Learning

The dataset have the same label or no labels.

鸡尾酒问题：两个人同时讲话，两个不同距离的麦克风录到了声音的不同组合。-->奇异值分解

# 单变量线性回归 Linear Regression with One Variable

## Model Description

$$m$$: number of training examples

$$x$$'s: "input" variable/features

$$y$$'s: "output" variable /"target" training example

$$(x,y)$$: one triaing example

$$(x^{(i)}, y^{(i)})$$: $$i$$th training example

Training Set -> Learning Algorithm -> $$h$$

* $$h$$: hypothesis. $$h$$ maps from $$x$$'s to $$y$$'s.
* **linear regression with one variable**: $$h_{\theta}(x)=\theta_0+\theta_1x$$, also called **univariate linear regression**.

## Cost Function

Idea: Choose $$\theta_0, \theta_1$$ so that $$h_{\theta}(x)$$ is close to $$y$$ for our training examples $$(x, y)$$.

Object: $$minimize_{\theta_0, \theta_1}\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$$, $$m$$ is the number of training examples.

We definite a cost function $$J(\theta_0, \theta_1)=\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$$, and we are to: $$minimize_{\theta_0, \theta_1} J(\theta_0, \theta_1)$$. It is also called **square error cost function**.

**Simplified**:

$$h_\theta(x)=\theta_1x$$, meaning that $$\theta_0 = 0$$. For fixed $$\theta_1$$, this is a function of $$x$$.

now we have $$J(\theta_1)=\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$$. This is a function of $$\theta_1$$.

Each value of $$\theta_1$$ corresponds to a different hypothesis, and derive a different value of $$J(\theta_1)$$.

<img src="machine_learning_WUENDA.assets/image-20220514155128830.png" alt="image-20220514155128830" style="zoom:50%;" />

**Original:**

<img src="machine_learning_WUENDA.assets/image-20220514155435774.png" alt="image-20220514155435774" style="zoom:50%;" />

Use contour plots/contour figures below:

<img src="machine_learning_WUENDA.assets/image-20220514155627977.png" alt="image-20220514155627977" style="zoom:50%;" />

## Gradient Descent

We have function $$J(\theta_0,\theta_1)$$, and we want $$min_{\theta_0, \theta_1}J(\theta_0, \theta_1)$$

**Outline:**

* start with some $$\theta_0, \theta_1$$
* keep changing $$\theta_0, \theta_1$$ to reduce $$J(\theta_0, \theta_1)$$ until we hopefully end up at a minimum.

Starting from different starting point, you may end up with different local optimum(局部最优解).

**Gradient decent algorithm:**

repeat until convergence(收敛){

​    $$\theta_j := \theta_j-\alpha \frac{\partial}{\partial \theta_j}J(\theta_0, \theta_1)$$ (for $$j=0$$ and $$j=1$$)

}

* $$\alpha$$: learning rate, controls how big the step downhill
* $$\frac{\partial}{\partial \theta_j}J(\theta_0, \theta_1)$$: derivative term.

We need to **simultaneously** update $$\theta_0$$ and $$\theta_1$$.

 $$temp_0 := \theta_0-\alpha \frac{\partial}{\partial \theta_0}J(\theta_0, \theta_1)$$

 $$temp_1 := \theta_1-\alpha \frac{\partial}{\partial \theta_1}J(\theta_0, \theta_1)$$

$$\theta_0 := temp_0$$ 

$$\theta_1 := temp-1$$

Gradient descendant can converge to a local minimum, even with the learning rate $$\alpha$$ fixed.

* As we approach a local minnimum, gradient descent will **automatically take smaller steps** (the dirivative is coming close to 0). So, no need ro decrease $$\alpha$$ over time. 



**Into the linear regression:**

$$\frac{\partial}{\partial \theta_j}J(\theta_0, \theta_1)=\frac{\partial}{\partial \theta_j}*\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i))}-y^{(i)})^2=\frac{\partial}{\partial \theta_j}*\frac{1}{2m}\sum_{i=1}^m(\theta_0+\theta_1x^{(i)}-y^{(i)})^2$$

* $$j=0: \frac{\partial}{\partial \theta_0}J(\theta_0, \theta_1)=\frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i))}-y^{(i)})$$
* $$j=1: \frac{\partial}{\partial \theta_1}J(\theta_0, \theta_1)=\frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i))}-y^{(i)})*x^{(i)}$$

Gradinet descent algorithm:

repeat until convergence(收敛){

​    $$\theta_0 := \theta_0-\alpha \frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i)}-y^{(i)})$$

​    $$\theta_1 := \theta_1-\alpha \frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i)}-y^{(i)})*x^{(i)}$$

} remember to update $$\theta_0, \theta_1$$ simultaneously.

The cost funtion of a linear regression is always a convex function(凸函数). It does not have a local optimum, it only has the global optimum.

**"Batch" Gradient Descent:** each step of gradient descent uses all the training examples.($$\sum_{i=1}^m(h_{\theta}x^{(i))}-y^{(i)})$$)



# Linear Algebra Review

## Matrix and Vectors

Dimension of matrix: number of rows X number of columns. e.g. $$2\times 3$$

$$A_{i,j}$$: "$$i,j$$ entry" in the $$i^{th}$$ row, $$j^{th}$$ column

Vector: an $$n \times 1$$ Matrix

## Matrix-vector multiplication

$$A$$: $$m \times n$$ Matrix 

$$x$$: $$n\times 1$$ Matrix($$n$$-dimensional vector)

$$A \times x=y$$, $$y$$: $$m$$-dimensional vector

To get $$y_i$$,multiply $$A$$'s $$i^{th}$$ row with elemnets of vector $$x$$, and add them up.

## Matrix-Matrix multiplication

$$A$$: $$m \times n$$ Matrix 

$$B$$: $$n \times o$$ Matrix 

$$A \times B = C$$, $$C$$ is a $$m\times o$$ matrix

The $$i^{th}$$ column of the matrix $$C$$ is obtained by multiplying $$A$$ with the $$i^{th}$$ colimn of $$B$$ (For $$i=1,2,...o$$).

* $$A \times B \neq B \times A$$: not commutative 不可交换
* $$A \times (B \times C)=(A\times B)\times C$$: enjoy the associative property 服从结合律
* **Identity Matrix**: Denoted as $$I$$, or $$I_{n\times n}$$. 对角线元素为1，其余都为0
  * For any matrix $$A_{m\times n}$$, $$A\cdot I = I \cdot A = A$$

## Inverse and Transpose

**Matrix Inverse**: if $$A$$ is an $$m \times m$$ matrix(**square matrix**), and if has an inverse, $$AA^{-1}=A^{-1}A=I$$

* Matrices that do not have an inverse are **"singular"** or **"degenerate"** 奇异矩阵/退化矩阵

**Matrix Transpose**:

Let $$A$$ be an $$m\times n$$ matrix, and let $$B=A^T$$. Then $$B$$ is an $$n \times m$$ matrix, and $$B_{ij}=A_{ji}$$



# Linear Regresiion with multiple variables

## Multiple Features

$$n$$: the number of features

$$x^{(i)}$$: input(features) of $$i^{th}$$ training example

$$x_j^{(i)}$$: value of feature $$j$$ in $$i^{th}$$ training example

$$h_{\theta}(x)=\theta_0+\theta_1x_1+\theta_2x_2+...+\theta_xx_n$$

For convenience of notation, define $$x_0=1$$ ($$x_0^{(i)}=1$$)

$$X=[x_0, x_1, x_2, ..., x_n]^T \in R^{n+1}$$

$$\Theta=[\theta_0, \theta_1, \theta_2,...,\theta_n]^T\in R^{n+1}$$

Then: $$h_\theta(x)=\Theta^TX$$ -> **Multivariate Linear Regression**

## Multivariate Gradient Descent

Hypothetis: $$h_{\theta}(x)=\theta_0+\theta_1x_1+\theta_2x_2+...+\theta_xx_n$$

Parameters: $$\Theta=[\theta_0, \theta_1, \theta_2,...,\theta_n]^T\in R^{n+1}$$

Cost function: $$J(\Theta)=\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$$

Gradient Descent:

repeat{

​    $$\theta_j := \theta_j-\alpha \frac{\partial}{\partial \theta_j}J(\theta_0,..., \theta_n)$$ (simultaneously update for every $$j=0, ..., n$$)

​    ( $$\frac{\partial}{\partial \theta_j}J(\Theta)=\frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i)}-y^{(i)})*x^{(i)}_j$$)

}

* $$\theta_0 := \theta_0-\alpha\frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i)}-y^{(i)})*x^{(i)}_0$$
* $$\theta_1 := \theta_0-\alpha\frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i)}-y^{(i)})*x^{(i)}_1$$
* $$\theta_2 := \theta_0-\alpha\frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i)}-y^{(i)})*x^{(i)}_2$$
* ......

## Feature Scaling

**Idea: make sure features are on a similar scale.**

e.g: $$x_1$$: size (0-2000 feet2), $$x_2$$: number of bedrooms(1-5)

**After Scaling**:

* $$x_1$$: size/2000
* $$x_2$$: number of bedrooms/5

**Feature Scaling: Get every feature into approxiamtely a $$-1 \leq x_i \leq 1$$ range.**

**Mean Normalization:** Replace $$x_i$$ with $$x_i-u_i$$ to make features have approximatelt zero mean **(Do not apply to $$x_0=1$$)**

* $$x_1=\frac{size-1000}{2000}$$
* $$x_2=\frac{No.bedrooms-2}{5}$$

More generally, $$x_1 = \frac{x_1-u_1}{s_1}$$

* $$u_1$$: avarage value of $$x_1$$
* $$s_1$$: range value (max-min), also called standard deviation

## Learning rate 

$$J(\Theta)$$ should decrease after every iteration.

Example automatic convergence test:

* Declare convergence if $$J(\Theta)$$ decreases by less than $$10^{-3}$$ in one iteration.

But if $$\alpha$$ is too small, gradient descent can be very slow.

Summary:

* If $$\alpha$$ is too small: slow convergence.
* if $$\alpha$$ is too large: $$J(\Theta)$$ may not decrease on every iteration, may not converge.
* To choose $$\alpha$$, try $$..., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, ...$$

## Features and Polynomial Regression

Polynomial Regression:

![image-20220516182123529](machine_learning_WUENDA.assets/image-20220516182123529.png)

$$h_{\theta}(x)=\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3=\theta_0+\theta_1(size)+\theta_2(size)^2+\theta_3(size)^3$$  

* $$x_1=(size)$$
* $$x_2=(size)^2$$
* $$x_3=(size)^3$$

**It's important to do feature scaling.**



## Normal Equation

Method to solve for $$\Theta$$ analytically.

$$\theta \in R^{n+1}, $$ $$J(\Theta)=\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$$, $$\frac{\partial}{\partial \theta_j}J(\Theta)=...=0$$, for every$$j$$.

* Solve for $$\theta_0, \theta_1, ..., \theta_n$$
* $$\Theta = (X^TX)^{-1}X^Ty$$

Generally, we have $$(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})$$ and $$n$$ features.

$$x^{(i)}=[x_0^{(i)}, x_1^{(i)}, ...,x_n^{(i)}] \in R^{n+1}$$

$$X:m\times (n+1), y:$$$$m$$-dimensional vector

| Gradient Descent                     | Normal Equation                                              |
| ------------------------------------ | ------------------------------------------------------------ |
| Need to choose $$\alpha$$.           | No need to choose $$\alpha$$.                                |
| Needs many iterations.               | Don't need to iterate.                                       |
| Works well even when $$n$$ is large. | Need to compute $$ (X^TX)^{-1}$$, $$O(n^3)$$, slow if $$n$$ is large |

**For the specific model of linear regression, normal equation is an alternative of gradient descent.**

Question: What if $X^TX$ is non-invertible? Computer can still handle the questions.

*   Redundant Features (linearly dependent)
    *   $x_1$: size in feet2
    *   $x_2$: size in m2
*   Too many features ($m\leq n$)
    *   delete some features or use regularization.



# Classification

$y \in \{0,1\}$

*   0: Negative class
*   1: Positive class

Threshold classifier output $h_\theta(x)$ at 0.5:

*   if $h_\theta(x) \geq 0.5$, predict "$y=1$"
*   if $h_\theta(x) <0.5$, predict "$y=0$"

**Logistic Regression**: $0 \leq h_\theta(x)\leq 1$ actually a classification problem.



## Hypothesis Representation

$h_\theta(x) = g(\theta^Tx)$

Sigmoid Function / Logistic Function: $g(z) = \frac{1}{1+e^{-z}}$

Then, $h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}$

![551652855689_.pic](machine_learning_WUENDA.assets/551652855689_.pic.jpg)

**Interpretation of Hypothesis Output:**

$h_\theta(x)$: estimated probability that $y=1$ on input $x$.

*   If $x=[x_0, x_1]^T=[1, tumorSize]^T, h_\theta(x)=0.7$, means the 70% chance of tumor being malignant
*   $h_\theta(x)=p(y=1|x;\theta)$: probability that $y=1$, given $x$, parameterized by $\theta$.
*   $y=$ 0 or 1
    *   $P(y=0|x;\theta)+P(y=1|x;\theta)=1$
    *   $P(y=0|x;\theta) = 1-P(y=1|x;\theta)$



## Decision Boundary

Suppose predict "$y=1$" if $h_\theta(x) \geq 0.5$

*   That is, when $\theta^Tx \geq 0$

Predict "$y=0$" if $h_\theta(x) <0.5$

*   That is, when $\theta^Tx<0$

![561652855689_.pic](machine_learning_WUENDA.assets/561652855689_.pic.jpg)

$h_\theta(x)=g(\theta_0+\theta_1x_1+\theta_2x_2)$

When $\theta=[-3,1,1]^T$, predict "$y=1$" if $-3+x_1+x_2\geq0$

**Non-linear decision boundaries**:

![571652855689_.pic](machine_learning_WUENDA.assets/571652855689_.pic.jpg)

$h_\theta(x)=g(\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_1^2+\theta_4x_2^2)$

When $\theta=[-1, 0, 0, 1, 1]^T$, predict "$y=1$" if $-1+x_1^2+x_2^2 \geq 0$.

We use $\theta$ **but not the training set** to difine decision boundary.

*   The training set may be used to fit the parameter $\theta$.



## Cost Function

Training set: $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}),..., (x^{(m)}, y^{(m)})\}$

$m$ examples $x \in [x_0, x_1, ..., x_n]^T, x_0=1, y \in \{0,1\}$ 

$h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}$

How to choose parameter $\theta$?

**Logistic Regression cost function:**
$$
cost(h_\theta(x), y)= \begin{cases}
-log(h_\theta(x)),\quad &y=1 \\
-log(1-h_\theta(x)),\quad &y=0
\end{cases}
$$
![image-20220517113831944](machine_learning_WUENDA.assets/image-20220517113831944.png)

![image-20220517114352638](machine_learning_WUENDA.assets/image-20220517114352638.png)

$cost=0$ if $y=1. h_\theta(x)=1$.

* But as $h_\theta(x) \rightarrow 0$, $cost \rightarrow \infty$
* Captures intuition that if $h_\theta(x)=0$ (predict $P(y=1|x;\theta)=0$), but actually $y=1$, we will penalize learning algorithm by a very large cost.

$cost=0$ if $y=0. h_\theta(x)=0$.

* But as $h_\theta(x) \rightarrow 1$, $cost \rightarrow \infty$
* Captures intuition that if $h_\theta(x)=1$ (predict $P(y=1|x;\theta)=1$), but actually $y=0$, we will penalize learning algorithm by a very large cost.



## Simplified cost function and gradient descent

$J(\theta)=\frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}), y^{(i)})$

$cost(h_\theta(x), y)= \begin{cases}
-log(h_\theta(x)),\quad &y=1 \\
-log(1-h_\theta(x)),\quad &y=0
\end{cases}$

Note: $y\in \{0,1\}$ always

**Equivalent**:$cost(h_\theta(x), y)=-ylog(h_\theta(x))-(1-y)log(1-h_\theta(x))$

so, $J(\theta)=\frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}), y^{(i)})=-\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(h_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$

To fit $\theta$, we should $min_{\theta}J(\theta)$

To make a prediction given new $x$, output: $h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}$, which means $p(y=1|x;\theta)$



**Gradient Descent:**

$J(\theta)=-\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(h_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$

Want $min_{\theta}J(\theta)$

Repeat {

​    $\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)$ (simultaneously update all $\theta_j$)

}

In that, $\frac{\partial}{\partial\theta_j}J(\theta) = \frac{1}{m}\sum_{i=1}^m[h_\theta(x^{(i)})-y^{(i)}]x^{(i)}_j$, **looks identical to linear regression**.



## Advanced Optimization

Cost Function $J(\theta)$, want $min_\theta J(\theta)$

Given $\theta$, we have code that can compute:

* $J(\theta)$
* $\frac{\partial}{\partial\theta_j}J(\theta)$   (for $j=0,1,...,m$)

Optimazation algorithms:

* Gradient Descent
* Conjugate gradient
* BFGS
* L-BFGS

The last three algorithms:

* Advantages:
  * No need to manually pick $\alpha$
  * Often faster than gradient descent
* Disadvantages:
  * More complex



## Multi-class classification: one-vs-all

Example:

* Email folding/tagging: Work; Friends; Family; Hobby
* Medical diagrams: Not ill; Cold; Flu
* Weather: Sunny; Cloudy; Rainy; Snowy

![image-20220517150225645](machine_learning_WUENDA.assets/image-20220517150225645.png)

**One-vs-all / One-vs-rest**

![image-20220517150432310](machine_learning_WUENDA.assets/image-20220517150432310.png)

$h_\theta^{(i)}(x)=P(y=i|x;\theta)  (i=1,2,3)$

* $h_\theta^{(1)}(x)$ to classify triangle
* $h_\theta^{(2)}(x)$ to classify square
* $h_\theta^{(3)}(x)$ to classify cross

**One-vs-all**: 

* Train a logistic regression classifier $h_\theta^{(i)}(x)$ for each class $i$ to predict the probablity that $y=i$.
* On a new input $x$, to make a prediction, pick the class $i$ that $max_ih_\theta^{(i)}(x)$.

# Regularization

## The problem of overfitting

<img src="machine_learning_WUENDA.assets/image-20220518165617638.png" alt="image-20220518165617638" style="zoom:50%;" />

"Underfitting" "highbias"

"overfitting" "high variance"

**Overfitting**:If we have too many features, the learned hypothesis may fit the training set very well ($j(\theta)\approx0$), but fail to generalize to new examples.

Addressing overfitting:

* Reduce number of features
  * Manually select what features to keep
  * model selection algorithm
* Regularization
  * Keep all the features, but reduce magnitude/valus of parameter $\theta$
  * Works well when we have a lot of features, each of which contributes a bit to predicting $y$.

## Cost Function

<img src="machine_learning_WUENDA.assets/image-20220518165617638.png" alt="image-20220518165617638" style="zoom:50%;" />

Suppose we penalize and make $\theta_3, \theta_4$ very small:

$min_\theta\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2+1000\theta_3^2+1000\theta_4^2$ --> $\theta_3 \approx 0, \theta_4 \approx 0$

**Regularization**

Small values parameters $\theta_0, \theta_1, ..., \theta_n$

* "Simpler" hypothesis
* Less prone to overfitting

e.g.: Housing:

* Features: $x_1, x_2, ..., x_{100}$
* Parameters: $\theta_1, \theta_2, ..., \theta_{100}$

$J(\theta)=\frac{1}{2m}[\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2+\lambda \sum_{j=1}^n\theta_j^2]$

* $\lambda \sum_{j=1}^n\theta_j^2$: regularization term
  * $\lambda$: regularization parameter, controls the trade-off between 2 different goals
    * fit the training data well
    * keep the parameter small
  * if $\lambda$ is set to an extremely large value, like $10^{10}$, then the penalizing on the parameters would be too heavy, so all the $\theta_1, \theta_2,... \approx 0$
    * In this situation, $h_\theta(x)=\theta_0$ --> **underfitting**



## Regularize Linear Regression

**regularized linear regression**

$J(\theta)=\frac{1}{2m}[\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2+\lambda \sum_{j=1}^n\theta_j^2]$

**Gradient Descent:**

repeat until convergence(收敛){

​    $$\theta_0 := \theta_0-\alpha \frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i)}-y^{(i)})$$

​    $$\theta_j := \theta_j-\alpha [\frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i)}-y^{(i)})x^{(i)}+\frac{\lambda}{m}\theta_j], j=1,2,3...,n$$

} 

 $$\theta_j := \theta_j(1-\alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i)}-y^{(i)})x^{(i)}, j=1,2,3...,n$$

* $1-\alpha\frac{\lambda}{m}<1$: slightly making $\theta_j$ smaller 略小于1

**Normal Equation**

$X=[(x^{(1)})^T, ..., (x^{(m)})^T]^T \in R^{m\times(n+1)}$

$y=[y^{(1)}, ..., y^{(m)}]^T \in R^m$

$\theta = (X^TX+\lambda \begin{bmatrix}0&0&0...0\\0&1&0...0\\0&0&1...0\\...\\0&0&0...1 \end{bmatrix})^{-1}X^Ty$, where $M$ is a $(n+1)\times(n+1)$ sqaure matrix，**same for non-invertible $X$**



## Regularized Logistic Regression

$J(\theta)=-\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(h_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$

**Gradient Descent:**

repeat until convergence(收敛){

​    $$\theta_0 := \theta_0-\alpha \frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i)}-y^{(i)})x_0^{{i}}$$

​    $$\theta_j := \theta_j-\alpha [\frac{1}{m}\sum_{i=1}^m(h_{\theta}x^{(i)}-y^{(i)})x^{(i)}+\frac{\lambda}{m}\theta_j], j=1,2,3...,n$$

} 



# Neural Network

## Model Repretation

Neuron model: logistic unit

![image-20220520161013851](machine_learning_WUENDA.assets/image-20220520161013851.png)

$h_\theta(x) = \frac{1}{1+e^{-\theta^tx}}$

$x=[x1, x2, x3, x4]^T, \theta=[\theta_1,\theta_2, \theta_3, \theta_4]^T$. $\theta$ is called "weights" or "parameters".

$x_0$: bias neuron / bias unit. $x_0=1$. Sometimes draw, sometimes nor draw for convenience.

Sigmoid (logistic) activation function: $g(z)=\frac{1}{1+e^{-z}}$

**Neural Network**:

![image-20220520161458353](machine_learning_WUENDA.assets/image-20220520161458353.png)

* not draw: $x_0 = 0$, $a_0^{(2)}=0$, bias unit
* Layer1: Input layer
* Layer2: Hidden Layer (every layer that is either not input layer nor not output layer)
* Layer3: Output Layer

$a_i^{(j)}$: "activation" of unit $i$ in layer $j$.

$\Theta^{(j)}$: matrix of weights controlling funtion mapping from layer $j$ to layer $j+1$.

* $a_1^{(2)}=g(\Theta_{10}^{(1)}x_0+\Theta_{11}^{(1)}x_1+\Theta_{12}^{(1)}x_2+\Theta_{13}^{(1)}x_3)$
* $a_2^{(2)}=g(\Theta_{20}^{(1)}x_0+\Theta_{21}^{(1)}x_1+\Theta_{22}^{(1)}x_2+\Theta_{23}^{(1)}x_3)$
* $a_3^{(2)}=g(\Theta_{30}^{(1)}x_0+\Theta_{31}^{(1)}x_1+\Theta_{32}^{(1)}x_2+\Theta_{33}^{(1)}x_3)$
* $h_\Theta(x)=a_1^{(3)}=g(\Theta_{10}^{(2)}a_0^{(2)}+\Theta_{11}^{(2)}a_1^{(2)}+\Theta_{12}^{(2)}a_2^{(2)}+\Theta_{13}^{(2)}a_3^{(2)})$

If network has $s_j$ units in layer $j$, $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ would be of dimention $s_{j+1}\times(s_j+1)$

* $\Theta^{(1)} \in R^{3\times 4}$
* $\Theta^{(2)}\in R^{1 \times 4}$

**Forward propagation: vectorized implememtation**

Define:

* $z_1^{(2)}=\Theta_{10}^{(1)}x_0+\Theta_{11}^{(1)}x_1+\Theta_{12}^{(1)}x_2+\Theta_{13}^{(1)}x_3$
* $z_2^{(2)}=\Theta_{20}^{(1)}x_0+\Theta_{21}^{(1)}x_1+\Theta_{22}^{(1)}x_2+\Theta_{23}^{(1)}x_3$
* $z_3^{(2)}=\Theta_{30}^{(1)}x_0+\Theta_{31}^{(1)}x_1+\Theta_{32}^{(1)}x_2+\Theta_{33}^{(1)}x_3$

In this way, we have:

* $a_1^{(2)} = g(z_1^{(2)})$
* $a_2^{(2)} = g(z_2^{(2)})$
* $a_3^{(2)} = g(z_3^{(2)})$

$x = [x_0, x_1, x_2, x_3]^T, z^{(2)} = [z_1^{(2)}, z_2^{(2)}, z_3^{(2)}]^T$

So: 

* $z^{(2)}=\Theta^{(1)}x$, $z^{(2)} \in R^3$, and if denote $x$ as $a^{(1)}$, then $z^{(2)}=\Theta^{(1)}a^{(1)}$
* $a^{(2)}=g(z^{(2)})$, $a^{(2)} \in R^3$

**Add** $a_0^{(2)}=1$, (in this way, $a^{(2)} \in R^4$), then:

* $z^{(3)} = \Theta^{(2)}a^{(2)}$
  * $z^{(3)}=\Theta_{10}^{(2)}a_0^{(2)}+\Theta_{11}^{(2)}a_1^{(2)}+\Theta_{12}^{(2)}a_2^{(2)}+\Theta_{13}^{(2)}a_3^{(2)}$
* $h_\Theta(x) = a^{(3)}=g(z^{(3)})$

**What the neural network is doing is just like logstic regression, except that rather then using the original features $x_1, x_2, x_3$, is using the NEW features $a_1, a_2, a_3$.**

* $a_1^{(2)},a_2^{(2)},a_3^{(2)}$ are leanred as function mapping layer 1 to layer 2 of the input
  * determined by other parameters: $\Theta^{(1)}$.

**architecture**: how the neurons are connected to each other.



## Examples and intuition

**Simple example: AND**

$x_1, x_2 \in \{0,1\}, y= x_1 \&\& x_2$

![image-20220520172946341](machine_learning_WUENDA.assets/image-20220520172946341.png)

With $\Theta_{10}^{(1)}=-30, \Theta_{11}^{(1)}=20, \Theta_{12}^{(1)}=20$, $h_\Theta(x) = \frac{1}{1+e^{-\theta^Tx}}$

| $x_1$ | $x_2$ | $h_\Theta(x)$    |
| ----- | ----- | ---------------- |
| 0     | 0     | $g(-30)\approx0$ |
| 0     | 1     | $g(-10)\approx0$ |
| 1     | 0     | $g(-10)\approx0$ |
| 1     | 1     | $g(10)\approx1$  |

**Simple example: OR**

$x_1, x_2 \in \{0,1\}, y= x_1 || x_2$

![image-20220520173340848](machine_learning_WUENDA.assets/image-20220520173340848.png)

With $\Theta_{10}^{(1)}=-10, \Theta_{11}^{(1)}=20, \Theta_{12}^{(1)}=20$, $h_\Theta(x) = \frac{1}{1+e^{-\theta^Tx}}$

| $x_1$ | $x_2$ | $h_\Theta(x)$    |
| ----- | ----- | ---------------- |
| 0     | 0     | $g(-10)\approx0$ |
| 0     | 1     | $g(10)\approx1$  |
| 1     | 0     | $g(10)\approx1$  |
| 1     | 1     | $g(30)\approx1$  |

**Simple Example: Negation**

$x_1 \in \{0,1\}, y = NOT x_1$

![image-20220520173614913](machine_learning_WUENDA.assets/image-20220520173614913.png)

With $\Theta_{10}^{(1)}=10, \Theta_{11}^{(1)}=-20$, $h_\Theta(x) = \frac{1}{1+e^{-\theta^Tx}}$

| $x_1$ | $h_\Theta(x)$      |
| ----- | ------------------ |
| 0     | $g(10) \approx 0$  |
| 1     | $g(-10) \approx 0$ |



**How to compute (NOT $x_1$) AND (NOT $x_2$)?**

<img src="machine_learning_WUENDA.assets/image-20220520174502485.png" alt="image-20220520174502485" style="zoom:50%;" />

With $\Theta_{10}^{(1)}=10, \Theta_{11}^{(1)}=-20, \Theta_{12}^{(1)}=-20$, $h_\Theta(x) = \frac{1}{1+e^{-\theta^Tx}}$

| $x_1$ | $x_2$ | $h_\Theta(x)$    |
| ----- | ----- | ---------------- |
| 0     | 0     | $g(10)\approx1$  |
| 0     | 1     | $g(-10)\approx0$ |
| 1     | 0     | $g(-10)\approx0$ |
| 1     | 1     | $g(-20)\approx0$ |



**How to compute $x_1$ XNOR $x_2$?**

* XNOR: 同或，相同为1，不同为0
* $x_1$ XNOR $x_2$ = ($x_1$ and $x_2$) or (NOT $x_1$ and NOT $x_2$)

<img src="machine_learning_WUENDA.assets/image-20220520174726936.png" alt="image-20220520174726936" style="zoom:50%;" />

<img src="machine_learning_WUENDA.assets/流程图 (3).jpg" alt="流程图 (3)" style="zoom:50%;" />



## Multi-class classification

![image-20220520181319830](machine_learning_WUENDA.assets/image-20220520181319830.png)

$h_\Theta(x) \in R^4$

* $h\Theta(x) \approx [1,0,0,0]^T$, predict as **class1**
* $h\Theta(x) \approx [0,1,0,0]^T$, predict as **class2**
* $h\Theta(x) \approx [0,0,1,0]^T$, predict as **class3**
* ...

Training set: $(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)})$

* $y^{(i)} \in \{[1,0,0,0]^T, [0,1,0,0]^T, [0,0,1,0]^T, [0,0,0,1]^T\}$

## Cost Function

Training set: $(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)})$

$L$: total no. of layers in network

$s_l$: no. of units (not including bias unit) in the layer $l$

* Binary classification $y \in \{0,1\}$
  * 1 output unit
  * $h_\Theta(x) \in R$
  * $s_L = 1$
  * $K=1$
* Multi-class classification($K$ classes)
  * $y \in R^k$
  * $k$ output units

Logstic regression cost function:

$J(\theta)=-\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(h_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$

**Neural Network cost function:**

$h_\Theta(x) \in R^k$, $(h_\Theta(x))_i = i^{th}$output

$J(\Theta) = -\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K[y^{(i)}_klog(h_\Theta(x^{(i)}))_k+(1-y^{(i)}_k)log(1-(h_\Theta(x^{(i)}))_k)]+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\Theta_{ji}^{(l)})^2$

* We don't compute $\Theta_{j0}$ in regularized term.



## Backpropogation algorithm

$J(\Theta) = -\frac{1}{m}[\sum_{i=1}^m\sum_{k=1}^Ky^{(i)}_klog(h_\Theta(x^{(i)}))_k+(1-y^{(i)}_k)log(1-(h_\Theta(x^{(i)}))_k)]+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\Theta_{ji}^{(l)})^2$

Specially, 1 output unit: $J(\Theta) = -\frac{1}{m}[\sum_{i=1}^my^{(i)}log(h_\Theta(x^{(i)}))+(1-y^{(i)})log(1-(h_\Theta(x^{(i)})))]+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\Theta_{jl}^{(l)})^2$

Want: $min_\Theta (\Theta)$

Need to compute:

* $J(\Theta)$
* $\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)$

Given one training example $(x,y)$:

Forward propagation:

<img src="machine_learning_WUENDA.assets/image-20220520235845577.png" alt="image-20220520235845577" style="zoom:50%;" />

* $a^{(1)} = x$
* $z^{(2)} = \Theta^{(1)}a^{(1)}$
* $a^{(2)} = g(z^{(2)})$, (add $a_0^{(2)}$)
* $z^{(3)} = \Theta^{(2)}a^{(2)}$
* $a^{(3)} = g(z^{(3)})$, (add $a_0^{(3)}$)
* $z^{(4)} = \Theta^{(3)}a^{(3)}$
* $a^{4} = h_\Theta(x)=g(z^{(4)})$

**Gradient computation: Backpropogatin algorithm**

Intuition: $\delta_j^{(l)}$ = "error" of node $j$ in layer $l$

* $a_j^{(l)}$: the activation of node $j$ in layer $l$

For each output unit (layer $L$=4)

<img src="machine_learning_WUENDA.assets/image-20220521160033334.png" alt="image-20220521160033334" style="zoom:50%;" />

* $\delta_j^{(4)}=a_j^{(4)}-y_j$
  * $a_j^{(4)} = (h_\Theta(x))_j$
  * also can be written as: $\delta^{(4)}=a^{(4)}-y$
* $\delta^{(3)}= (\Theta^{(3)})^T\delta^{(4)}.*g'(z^{(3)})$
  * $g'(z^{(3)}) = a^{(3)}.*(1-a^{(3)})$
  * $.*$: 两个矩阵中的各个对应元素相乘，得到一个新的矩阵
*  $\delta^{(2)}= (\Theta^{(2)})^T\delta^{(3)}.*g'(z^{(2)})$
* no $\delta^{(1)}$
* 推导过程：
  * $\delta^{(l)}=\frac{\partial J(\theta)}{\partial z^{(l)}}$
  * 推导$\delta^{(3)},\delta^{(2)}$ (链式求导法则)：
    * $\delta^{(3)} = \frac{\partial J(\theta)}{\partial z^{(l)}}$
    * $ = \frac{\partial J}{\partial a^{(4)}}\cdot \frac{\partial a^{(4)}}{\partial z^{(4)}}\cdot \frac{\partial z^{(4)}}{\partial a^{(3)}}\cdot \frac{\partial a^{(3)}}{\partial z^{(3)}}$
    * $ =\left(\frac{-y}{a^{(4)}} + \frac{(1-y)}{1-a^{(4)}}\right)\cdot \frac{\partial g(z^{(4)})}{\partial z^{(4)}}\cdot \theta^{(3)}\cdot \frac{\partial g(z^{(3)})}{\partial z^{(3)}}$
    * =$\left(\frac{-y}{a^{(4)}} + \frac{(1-y)}{1-a^{(4)}}\right)\cdot a^{(4)}\cdot(1-a^{(4)})\cdot \theta^{(3)}\cdot g'(z^{(3)})$
    * $=(a^{(4)}-y)\cdot \theta^{(3)}\cdot a^{(3)} \cdot(1-a^{(3)})$
    * $=(\theta^{(3)})^T\cdot \delta^{(4)}\cdot g'(z^{(3)})$ (考虑维度问题)


**Backpropogation algorithm**:

Training set: $(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)})$

Set $\Delta_{ij}^{(l)} = 0$, for all $i,j,l$, used to compute $\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)$

For $i=1,2,...,m$

* Set $a^{(1)} = x^i$
* Perform forward propogation to compute $a^{(l)}$ for $l=2,3,...L$
* using $y^{(i)}$, compute $\delta^{(L)}=a^{(L)}-y^{(i)}$
* Compute: $\delta^{(L-1)}, \delta^{(L-2)}, ... \delta^{(2)}$
* $\Delta^{(l)}_{ij}:=\Delta^{(l)}_{ij}+a_j^{(l)}\delta_i^{(l+1)}$

$D_{ij}^{(l)} = \frac{1}{m}\Delta_{ij}^{(l)}+\frac{\lambda}{m}\Theta_{ij}^{(l)}$, if $j \neq 0$

$D_{ij}^{(l)} = \frac{1}{m}\Delta_{ij}^{(l)}$, if $j=0$

Then, $\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta) = D_{ij}^{(l)}$

<img src="machine_learning_WUENDA.assets/image-20220521163127754.png" alt="image-20220521163127754" style="zoom:50%;" />

**What is backpropogation doing?**

$J(\Theta) = -\frac{1}{m}[\sum_{i=1}^my^{(i)}log(h_\Theta(x^{(i)}))+(1-y^{(i)})log(1-(h_\Theta(x^{(i)})))]+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\Theta_{jl}^{(l)})^2$

Focusing on a single example $x^{(i)}, y^{(i)}$, the case of 1 output unit, and ignoring regularization ($\lambda = 0$)

Then, $cost(i) = y^{(i)}log(h_\Theta(x^{(i)}))+(1-y^{(i)})log(1-(h_\Theta(x^{(i)})))$

* Thinking is as $cost(i) \approx (h_\Theta(x^{(i)})-y^{(i)})^2$
* i.e. How well the network is doing on example $i$

<img src="machine_learning_WUENDA.assets/image-20220521164938213.png" alt="image-20220521164938213" style="zoom:50%;" />

In the backpropogation: 

* $\delta_j^{(l)}$ = "error" of cost for $a_j^{(l)}$ (unit $j$ in layer $l$)
* Formally, $\delta_j^{(l)} = \frac{\partial}{\partial z_j^{(l)}}$, for $j \geq 0$, where $cost(i) = y^{(i)}log(h_\Theta(x^{(i)}))+(1-y^{(i)})log(1-(h_\Theta(x^{(i)})))$



## Gradient checking

**Numerical estimation of gradients:**

<img src="machine_learning_WUENDA.assets/image-20220521181441124.png" alt="image-20220521181441124" style="zoom:50%;" />

$\frac{d}{d\theta}J(\theta) \approx \frac{J(\theta+\epsilon)- J(\theta-\epsilon)}{2\epsilon}$, $\epsilon = 10^{-4}$

**Paramerter vector $\theta$**

$\theta \in R^n$, e.g. $\theta$ is "unrolled" version of $\Theta^{(1)}, \Theta^{(2)}, \Theta^{(3)}$

$\theta = [\theta_1, \theta_2, \theta_3, ..., \theta_n]$

Then:

* $\frac{\partial}{\partial \theta_1}J(\theta)\approx \frac{J(\theta_1+\epsilon, \theta_2, ..., \theta_n)-J(\theta_1-\epsilon, \theta_2, ..., \theta_n)}{2\epsilon}$
* $\frac{\partial}{\partial \theta_2}J(\theta)\approx \frac{J(\theta_1, \theta_2+\epsilon, ..., \theta_n)-J(\theta_1, \theta_2-\epsilon, ..., \theta_n)}{2\epsilon}$
* ...
* $\frac{\partial}{\partial \theta_n}J(\theta)\approx \frac{J(\theta_1, \theta_2, ..., \theta_n+\epsilon)-J(\theta_1, \theta_2, ..., \theta_n-\epsilon)}{2\epsilon}$

**Remember to check that this numerical gradient $\approx$ Backpropogation**



## Ramdom Initialization

**Zero initialization**

$\Theta_{ij}^{(l)} = 0$ for all $i,j,l$

In this way, $a_1^{(2)} = a_2^{(2)}, \delta_1^{(2)} = \delta_2^{(2)}$--> $\frac{\partial}{\partial\Theta^{(1)}_{01}}J(\Theta)=\frac{\partial}{\partial\Theta^{(1)}_{02}}J(\Theta)$-->$\Theta^{(1)}_{01}=\Theta^{(1)}_{02}$

After each update, parameters correponding to inputs going into each hidden units are identical

<img src="machine_learning_WUENDA.assets/image-20220521184200103.png" alt="image-20220521184200103" style="zoom:50%;" />

The two dark blue lines, red lines, green lines and light blue lines are both identical.

**Ramdom initializartion: symmetry breaking**

initialize each $\Theta_{ij}^{(l)}$ to a ramdom value in $[-\epsilon, \epsilon]$



## Putting it together

**Training a neural network**

pick a network architecture (connectivity pattern between nuerons)

* No. of input units: dimension of features $x^{(i)}$
* No. of output units: number of classes
* Resonable default: 1 hidden layer, or ud > 1 hidden layer, have same no. of hidden units in every layer (usually the more the better)

1. Ramdomly initialize weights

2. Implement forward propagation to get $h_\Theta(x^{(i)})$ for any $x^{(i)}$

3. Implement code to compute cost function $J(\Theta)$

4. Implement backprapogation to compute partial derivatives $\frac{\partial}{\partial\Theta^{(l)}_{jk}}J(\Theta)$

   * for $i=1,...m$
     * Perform forward propagation and backpropagation using example $(x^{(i)}, y^{(i)})$
     * Get activation $a^{(l)}$ and delta terms $\delta^{(l)}$ for $l = 2,3,...,L$
     * compute $\Delta^{(l)}$
   * compute $\frac{\partial}{\partial\Theta^{(l)}_{jk}}J(\Theta)$

5. Use gradient checking to compare $\frac{\partial}{\partial\Theta^{(l)}_{jk}}J(\Theta)$ computed using backpropagation v.s. using numerical estimate of gradient of $J(\Theta)$.

   Then disable gradient checking code.

6. Use gradient descent or advanced optimization method with backpropagation to try to minimize $J(\Theta)$ as a function of parameters $\Theta$.



# Advice for applying machine learning

## Deciding what to do next

How to debug a learning algorithm?

* Get more training examples
* Try smaller sets of features
* Try getting additional features
* Try adding polynomial features ($x_1^2, x_2^2, x_1x_2...$)
* Try decreasing $\lambda$
* Try increasing $\lambda$

**Machine learning diagnostic:**

Diagnostic: A test that you can run to gain insight what is / is not working with a learning algorithm, and gain guidances as to how best to improve its performance.

Diagnostics can take time to implement, but doing so can ve a very good use of your time.

## Evaluating a hypothesis

**Ramdomly** choose 70% as training data: $(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})$

left 30% data as testing data: $(x^{(1)}_{test}, y^{(1)}_{test}), ..., (x^{(m_{test})}_{test}, y^{(m_{test})}_{test})$, here, $m_{test}=$ No. of test example

**Training/testing procedure for linear regression:**

* Learn parameter $\theta$ from training data (minimizing training error $J(\theta)$)
* Compute test set error: $J_{test}(\theta) = \frac{1}{2m_{test}}\sum_{i=1}^{m_{test}}(h_{\theta}(x_{test}^{(i)}-y_{test}^{(i)})^2$

**Training/testing procedure for logistic regression:**

* Learn parameter $\theta$ from training data (minimizing training error $J(\theta)$)
* Compute test set error: $J_{test}(\theta) = -\frac{1}{m_{test}}\sum_{i=1}^{m_{test}}[y_{test}^{(i)}log(h_{\theta}(x_{test}^{(i)}))+(1-y_{test})^{(i)}log(1-h_{\theta}(x_{test}^{(i)}))]$
* misclassification error (0/1 misclassification):
  * $err(h_\theta(x), y)= \begin{cases}
    1,\quad &if\ h_\theta(x) \geq 0.5\ and \ y =0, or\ if\ h_\theta(x)<0.5 \ and \ y=1 \\
    0,\quad &otherwise
    \end{cases}$
  * $Test_{error} = \frac{1}{m_{test}}\sum_{i=1}^{m_{test}}err(h_{\theta}(x_{test}^{(i)}),y_{test}^{(i)})$



## Model selection and training/validation/test sets

**Overfitting example**:

![image-20220601104900632](machine_learning_WUENDA.assets/image-20220601104900632.png)

Once parameters $\theta_0, ..., \theta_4$ were to fit some set of data (training set), the error of the parameters as measured on that data (the training error $J(\theta)$) is likely to be lower than the actual generalizetion error.

**Model selection**:

$d$ = degree of polybnomial

1. $h_\theta(x) = \theta_0+\theta_1x$  -> $\theta^{(1)}$ -> $J_{test}(\theta^{(1)})$

2. $h_\theta(x) = \theta_0+\theta_1x+\theta_2x^2$   -> $\theta^{(2)}$ -> $J_{test}(\theta^{(2)})$

3. $h_\theta(x) = \theta_0+\theta_1x+...+\theta_3x^3$   -> $\theta^{(53)}$ -> $J_{test}(\theta^{(3)})$

   ....

10.  $h_\theta(x) = \theta_0+\theta_1x+...+\theta_10x^10$   -> $\theta^{(10)}$ -> $J_{test}(\theta^{(10)})$

Choose the least $J_{test}$, -> choose $\theta_0+...+\theta_5x_5$

* How well does the model generalize?
  * report test set error $J_{test}(\theta^{(5)})$
* Problem: $J_{test}(\theta^{(5)})$ is likely to be an optimistic estimate of generalization error. i.e, our extra parameter $d$ is fir to test set. (Use test set to get the optimal $d$, and use $d$ on the test set to get the performance)

**Evaluating hypothesis**:

* Training set - 60%
  *  $(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})$
  * Training error: $J_{train}(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)}-y^{(i)})^2$
* Cross-validatiion set - 20%
  * $(x^{(1)}_{cv}, y^{(1)}_{cv}), ..., (x^{(m_cv)}_{cv}, y^{(m_{cv})}_{cv})$, here, $m_{cv}=$ No. of cross validation example
  * Cross validation error: $J_{cv}(\theta) = \frac{1}{2m_{cv}}\sum_{i=1}^{m_{cv}}(h_{\theta}(x_{cv}^{(i)}-y_{cv}^{(i)})^2$
* test set - 20%
  * $(x^{(1)}_{test}, y^{(1)}_{test}), ..., (x^{(m_{test})}_{test}, y^{(m_{test})}_{test})$, here, $m_{test}=$ No. of test example
  * Test error: $J_{test}(\theta) = \frac{1}{2m_{test}}\sum_{i=1}^{m_{test}}(h_{\theta}(x_{test}^{(i)}-y_{test}^{(i)})^2$

1. $h_\theta(x) = \theta_0+\theta_1x$  -> $\theta^{(1)}$ -> $J_{cv}(\theta^{(1)})$

2. $h_\theta(x) = \theta_0+\theta_1x+\theta_2x^2$   -> $\theta^{(2)}$ -> $J_{cv}(\theta^{(2)})$

3. $h_\theta(x) = \theta_0+\theta_1x+...+\theta_3x^3$   -> $\theta^{(53)}$ -> $J_{cv}(\theta^{(3)})$

   ....

10.  $h_\theta(x) = \theta_0+\theta_1x+...+\theta_10x^10$   -> $\theta^{(10)}$ -> $J_{cv}(\theta^{(10)})$

Pick $\theta_0+\theta_1x+...+\theta_4x^4$

Estimate generalization error for test set $J_{test}(\theta^{(4)})$



## Diagnosing bias v.s. variance

bias偏差 --> underfitting

variance方差 --> overfitting

![image-20220601112951612](machine_learning_WUENDA.assets/image-20220601112951612.png)

**Bias/variance**:

Training error: $J_{train}(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)}-y^{(i)})^2$

Cross validation error: $J_{cv}(\theta) = \frac{1}{2m_{cv}}\sum_{i=1}^{m_{cv}}(h_{\theta}(x_{cv}^{(i)}-y_{cv}^{(i)})^2$

Suppose your learning algorithm is performing less well than you were hoping. ($J_{cv}(\theta)$ or $J_{test}(\theta)$ is high). Is it a bias problem or a variance problem?

![image-20220601113728136](machine_learning_WUENDA.assets/image-20220601113728136.png)

* Bias(underfitting): 
  * $J_{train}(\theta)$ will be high
  * $J_{cv}(\theta) \approx J_{train}(\theta)$
* Variance(overfitting):
  * $J_{train}(\theta)$ will be low
  * $J_{cv}(\theta) >> J_{train}(\theta)$



## Regularization and bias/variance

**Linear regression with regularization**:

Model: $h_\theta(x) =\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4$

$J(\theta)=\frac{1}{2m}[\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2+\lambda \sum_{j=1}^n\theta_j^2]$

![image-20220601114335776](machine_learning_WUENDA.assets/image-20220601114335776.png)

**Choosing the regularization parameter $\lambda$**

 $h_\theta(x) =\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4$

$J(\theta)=\frac{1}{2m}[\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2+\lambda \sum_{j=1}^n\theta_j^2]$

$J_{train}(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)}-y^{(i)})^2$

$J_{cv}(\theta) = \frac{1}{2m_{cv}}\sum_{i=1}^{m_{cv}}(h_{\theta}(x_{cv}^{(i)}-y_{cv}^{(i)})^2$

$J_{test}(\theta) = \frac{1}{2m_{test}}\sum_{i=1}^{m_{test}}(h_{\theta}(x_{test}^{(i)}-y_{test}^{(i)})^2$

1. Try $\lambda = 0$ -> $min_\theta J(\theta)$ -> $\theta^{(1)}$ -> $J_{cv}(\theta^{(1)})$
2. Try $\lambda = 0.01$ -> $min_\theta J(\theta)$ -> $\theta^{(2)}$ -> $J_{cv}(\theta^{(2)})$
3. Try $\lambda = 0.02$ -> $min_\theta J(\theta)$ -> $\theta^{(3)}$ -> $J_{cv}(\theta^{(3)})$
4. Try $\lambda = 0.04$ -> $min_\theta J(\theta)$ -> $\theta^{(4)}$ -> $J_{cv}(\theta^{(4)})$

...

12.  Try $\lambda = 10.24$ -> $min_\theta J(\theta)$ -> $\theta^{(12)}$ -> $J_{cv}(\theta^{(12)})$

Pick $\theta^{(5)}$, test error: $J_{test}(\theta^{(5)})$

![image-20220601115625813](machine_learning_WUENDA.assets/image-20220601115625813.png)



## Learning curves

![image-20220601141818600](machine_learning_WUENDA.assets/image-20220601141818600.png)

![image-20220601142033952](machine_learning_WUENDA.assets/image-20220601142033952.png)

If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.

![image-20220601142509418](machine_learning_WUENDA.assets/image-20220601142509418.png)

If a learning algorithm is suffering from high variance, getting more training data is likely to help.

## Deciding what to do next

How to debug a learning algorithm?

* Get more training examples -> **fix high variance**
* Try smaller sets of features -> **fix high variance**
* Try getting additional features -> **fix high bias**
* Try adding polynomial features ($x_1^2, x_2^2, x_1x_2...$) -> **fix high bias**
* Try decreasing $\lambda$  -> **fix high bias**
* Try increasing $\lambda$ > **fix high variance**

**Neural Networks and overfitting**

"Small" neural network (fewer parameters, more prone to underditting) -> computationally cheaper

"Large" neural network (more parameters; more prone to overfitting) -> computationally more expensive

* Use regularization ($\lambda$) to address overfitting



# Machine Learning System Design

## Priorizing what to work on: Spam classification problem

**Building a spam classifier:**

Supervised learning. $x$ = features of email. $y$ = spam (1) or not spam(0).

* Feature $x$: Choose 100 words indicative of spam/nor spam.

* Note: In practice, take most frequently occuring $n$ words, (10000 to 50000), in training set, rather than manually picl 100 words.

How to spend your time to make it have low error?

* collect lots of data
* develop sophisticated features based on email routing information (from emial head)
* Develop sophisticated features for message body e.g. should "discount" and "discounts" be treated as the same work? 
* Develop sophisticased algorithm to detect misspellings (e.g. med1cine, w4tches)

## Error analysis

**recommended approach**:

* Start with a simple algorithm that you can implememt quickly. Inplement it and test it on your cross-validation data.
* Plot learning curves to decide if more data, more features, etc. are likely to help
* Error analysis: Manually examine the axemples (in cross validation set) that your algorithm made errors on. See if you set any systematic trend in what type of examples it is making errors on.

Email example:

* $m_{CV}=500$ examples in cross validation set
* Algorirhm misclassifies 100 emails.
* Manually examine the 100 errors, and categorize them based on:
  * what type of email it is 
  * What cues(features) you think would have helped the algorithm classify them correcrly.

**The importance of numerical evaluation**

Should discount/discounts/discounted/discounting be treated as the same word?

* Can use "stemming" software (e.g. porter stemmer)
  * Error: universe/university.

Error analysis may not be helpful for deciding if this is likely to improve performance. Only solutions is to try it and see if it works.

Need numerical evaluation (e.g. cross validation error) of algorithms's performance with and without stemming.

## Error metrics for skewed classes

**Cancer classification example**

Train logistic regression model $h_\theta(x)$, ($y=1$ if cancer, $y=0$ otherwise)

Find that you got 1% error on test set (99% correct diagnosis)

Only 0.5% of patients have cancer -> **skewed classes**, the positive and negative examples are not equal

**Precision/recall**

$y=1$ in presence of rare class that we want to detect.

|             | Actual 1       | Actual 0       |
| ----------- | -------------- | -------------- |
| Predicted 1 | True positive  | False positive |
| Predicted 0 | false negative | True negative  |

**Precision**: (of all patients where we predicted $y=1$, what fraction actually have cancer?)

$Precision = \frac{True\ positives}{True\ positives + False\ positives}$

**Recall**: (of all the patients that actually have cancer, what fraction did we correctly detect as having cancer?)

$Recall = \frac{True\ positives}{True\ positive+False\ negatives}$

## Trading off precision and recall

Logistic regression: $0 \leq h_\theta(x) \leq 1$

* predict 1 if $h_\theta(x) \geq threshold$
* predict 1 if $h_\theta(x) < threshold$

Suppose we want to predict $y=1$ only if very confident ($threshold = 0.7$) -> High precision, low recall

Suppose we want to avoid missing too many cases of cancer (avoid false negatives) ($threshold = 0.3$) -> High recall, low precision.

**F1 score (F score)**

How to compare precision/recall numbers?

$F1 = 2\frac{PR}{P+R}$



## data for machine learning

**Large data rationale**

* Use a learning algorithm with many parameters (e.g. logistic regression/linear regression with many features; nueral network with many hidden units) 
  * low bias algorithms
  * $J_{train}(\Theta)$ will be small.
* Use a very large training set (unlikely to overfit)
  * $J_{train}(\Theta) \approx J_{test}(\Theta)$
  * With the above condition, $J_{test}(\Theta)$ will be small.



