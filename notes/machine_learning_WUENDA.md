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



