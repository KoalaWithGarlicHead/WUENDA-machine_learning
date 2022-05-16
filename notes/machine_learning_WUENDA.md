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

