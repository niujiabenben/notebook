## 拉格朗日乘子, 距离, 以及SVM

SVM的核心是要最大化support vector到分类平面的距离.

## 拉格朗日乘子

拉格朗日乘子法解决**等式约束**时函数极值问题, 该方法保证
**所得的极点会包含原问题的所有极值点，但并不保证每个极值点都是原问题的极值点.**

拉格朗日乘子法可以用到所有的只含等式约束的最优化问题, 与目标函数是否是凸函数没有关系.

参考: [拉格朗日乘数wiki](
https://zh.wikipedia.org/wiki/%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E4%B9%98%E6%95%B0)

## 对偶问题

考虑一般的最优化问题:

1. $\min f(x) \\ s.t. \ g_i(x) \leq 0 \qquad i=1,...,m \\ \qquad h_j(x)=0 \qquad j=1,...,n$

其拉格朗日函数为:

2. $L(x,\lambda,\beta) = f(x)+\sum_{i}\lambda_ig_i(x)+\sum_{j}\beta_jh_j(x)$

其对偶函数为:

3. $g(\lambda,\beta) = inf_xL(x,\lambda,\beta)$, 其中, $\lambda \geq 0$

注意公式3的变量为 $\lambda$ 和 $\beta$, $x$ 为已知, $inf\{*\}$ 为下确界(与最小值类似,
但函数有可能只能无限接近于某个值, 这时候称为下确界). 公式3的含义为:一族关于
$\lambda,\beta$ 的函数的逐点的下确界, 即使原问题不是凸的, 对偶函数也是凹函数
(即: $-g(\lambda,\beta)$ 为凸函数).

这里再详细描述一下函数 $g(\lambda,\beta)$ 的求值过程: 对每一个 $x$ 而言, 首先将 $g$
看成是 $\lambda,\beta$ 的函数, 求出 $\lambda,\beta$ 使得 $g$ 最小, 此时,
$\lambda,\beta$ 为自变量的值, 得到的函数的最小值为函数值. 那有没有可能对于不同的 $x$
求得相同的 $\lambda,\beta$, 且 $g$ 的值不相同? 没有这种可能, 因为 $g$ 是关于
$\lambda,\beta$ 的线性函数, 是凹函数 (至于具体为什么, 这个我也不清楚).

对偶问题为对偶函数的极大值:

4. $\max g(\lambda,\beta) = \max_{\lambda, \beta} inf_xL(x,\lambda,\beta)$,
   其中, $\lambda \geq 0$

可以证明, 对偶问题构成了原问题的下界, 即, 设原问题的解为 $p^*$, 有:

5. $g(\lambda,\beta) \leq p^*, \quad \forall \lambda \geq 0, \forall \beta \in R^n$

由此可知, 对偶问题比原问题的条件更宽松. 可以通过求解对偶问题来确定原问题的下界.
当原问题为凸优化时, 对偶问题和原问题等价 (公式5中等号成立), 此时, 强对偶性成立.

参考:

1. [凸优化（八）——Lagrange对偶问题](https://www.jianshu.com/p/96db9a1d16e9)
2. [拉格朗日乘子法 - KKT条件 - 对偶问题](https://www.cnblogs.com/massquantity/p/10807311.html)
3. [知乎-如何通俗地讲解对偶问题](https://www.zhihu.com/question/58584814)

## 点到超平面的距离

#### 优化方法

高维空间一点$X'$ 到平面$WX+b=0$的距离可以表示为求解如下约束问题:

1. $\min \frac{1}{2} \|X-X'\|^2 \ s.t. \ WX+b=0$

这实际上是一个等式约束的二次规划问题, 可以用拉格朗日乘子法求解:

2. $\min f(X) = \frac{1}{2} \|X-X'\|^2 + \lambda (WX+b)$

3. $\frac{\partial f}{\partial X} = (X-X') + \lambda W = 0$

4. $\frac{\partial f}{\partial \lambda} = W X+b = 0$

注意公式3实际上是向量表示, 将X解出, 带入到公式4, 可得:

5. $W(X'-\lambda W)+b = 0 \Rightarrow \lambda = \frac{WX'+b}{\|W\|^2}$

将公式5代入到公式3可得:

6. $X-X' = \frac{WX'+b}{\|W\|^2} W$

注意公式6也是向量表示, 公式6两边平方开个根号即是距离(这里为了简便省掉根号,
直接计算距离的平方):

7. $\|X-X'\|^2 = \frac{(WX'+b)^2}{\|W\|^2}$

#### 几何方法

设高维空间点$X'$ 到平面$WX+b=0$的投影为$X^0$, 由于$W$为平面的法线(
任取平面上两点, 两点组成的向量与W的内积为0), 而$X'-X^0$与法线平行, 则有:

1. $\|(X'-X^0)*W\| = \|X'-X^0\|\|W\|=d\|W\|$

其中$d$即为点$X'$ 到给定平面的距离. 另一方面, 有:

2. $\|(X'-X^0)*W\| = \|X'W-X^0W\| = \|X'W+b\|$

结合公式1和2, 有: $d = \frac{\|WX'+b\|}{\|W\|}$

以上推导参考: [空间任一点到超平面的距离公式的推导过程](
https://blog.csdn.net/yutao03081/article/details/76652943)

## SVM

给定一组样本 $\{(X_i, y_i), \ i=1,...,n\}$, 其中 $y_i \in \{-1, 1\}$.
SVM的核心是要最大化support vector到分类平面的距离. 设分类平面为 $WX+b=0$,
注意到参数$W$和$b$等比例放大或缩小并不改变分类平面的的位置, 所以可以设分类平面平移到
support vector上时的方程为 $WX+b=1$ 和 $WX+b=-1$, 分别对应正例和负例.
这时两个support vector上的平面的距离为 $\frac{2}{\|W\|}$ (在一个平面上取一点,
利用点到平面的距离公式化简即得). 要最大化这个距离, 相当于最小化 $\frac{1}{2}\|W\|^2$.

所以SVM的基本的数学模型为:

1. $\min \frac{1}{2}\|W\|^2 \\ s.t. \quad 1-y_i(WX_i+b) \leq 0, \quad i={1,..,n}$

其对偶问题为:

2. $max_{\lambda} \ inf_{W,b}(\frac{1}{2}\|W\|^2+\sum_i{\lambda_i(1-y_i(WX_i+b))})$,
   其中, $\lambda_i \geq 0$.

因为内层优化问题是关于$W$和$b$的, 这时把 $\lambda$ 看成是常数, 所以是一个无约束优化问题,
直接令偏导数为0, 即可得到内层优化问题的解:

3. $W=\sum_i{\lambda_iy_iX_i}, \quad \sum_i{\lambda_i y_i}=0$

将公式3代入到公式2, 可得:

4. $min_{\lambda} \ \frac{1}{2}\sum_i\sum_j\lambda_i\lambda_jX_iX_j - \sum_i\lambda_i$
   其中, $\lambda_i \geq 0$, $\sum_i\lambda_iy_i=0$.

这个是一个带约束的二次规划问题, 是一个凸优化问题. 由于 $\sum_i\lambda_iy_i=0$ 约束的存在,
不能用常规的梯度下降方法求解, 可以用SMO方法求解.
