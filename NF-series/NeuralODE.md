# Neural ODE

## 发表情况
- Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. Advances in neural information processing systems, 31.
  - NeurIPS 2018
  - Best Paper Award
- 主要作者：Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
  - 多伦多大学
- 截止2026.10 cite：11643+
- 将离散堆叠的神经网络层替换为由神经网络参数化导数的连续常微分方程，借助伴随方法反向传播，得到连续深度模型，并由此提出了连续归一化流
  - 从离散神经网络到连续动力系统，从 Normalizing Flows 到 Score-SDE 的重要理论桥梁
- [原文链接](https://proceedings.neurips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf)

## 摘要解读：
- 摘要的内容主要分为三层意思
  - 首先，摘要说明了NeuralODE的出发点：不再用神经网络直接构造从输入到输出的离散映射，而是让神经网络定义状态的变化率，再通过求解微分方程得到最终输出
  - 第二，借助这个框架，文章设计了“连续归一化流”（continuous normalizing flows）-- 这是一种生成模型，无需对数据维度进行划分或排序，即可通过最大似然估计进行训练
  - 考虑到求解ODE的复杂性，文章设计了一种实现在ODE网络求解器中进行反向传播的机制

## 背景说明：
- Neural ODE 同时连接了三个原本相对独立的研究方向：
  - 深度神经网络：将 ResNet 解释为常微分方程的离散化，进而提出连续深度网络
  - Normalizing Flows：从离散可逆映射扩展到连续时间概率密度变换，提出 Continuous Normalizing Flows（CNF）
  - 连续时间生成模型：为后来的 Probability Flow ODE、Score-SDE 以及 Flow Matching 提供了重要的数学和计算基础

- 2016 年，ResNet 提出了残差连接：

$$
\mathbf h_{k+1}
=\mathbf h_k+f_{\theta_k}(\mathbf h_k)
$$
- 这个表达式实际上与数值分析中的 Euler 方法非常接近：

$$
\mathbf h_{k+1}
=\mathbf h_k+\Delta t f(\mathbf h_k,t_k)
$$
- 从这个角度来看，ResNet 已经是一个 ODE 的粗离散近似，前面也有一些工作在思考ResNet和ODE之间的关系，但没有将ODE求解器真正作为神经网络的组成模块，并建立可训练的通用框架
  - 既然 ResNet 本质就是在离散近似一条连续轨迹，那我们为什么非要困在这个粗糙、固定步长的离散近似里？能不能直接建模那条**底层的连续动力学本身**，而不是它的离散采样？
 
- 如果从NFs的角度来思考NeuralODE的提出背景，我们又可以得到：
- NFs系列工作实际上都在解决一个问题：通过可逆变换实现分布之间的转换
  - 但可逆变换和Jacobian矩阵求解的要求，又使研究人员不得不在模型表达能力和计算效率之间权衡
- Neural ODE不同于前面设计特殊的网络结构使计算更容易，它的思路是：如果不再使用离散的有限变换，而改为连续的无穷小变换，是否能够从数学上改变密度变化公式本身？

- 上述两种思考方式在Neural ODE中交汇，**Neural ODE通过构建连续时间向量场，来实现连续深度网络和连续密度建模**

## 具体分析：
- Neural ODE 则把网络中的状态定义为随连续变量 $t$ 变化的函数：

$$
\frac{d\mathbf z(t)}{dt}
=f_\theta(\mathbf z(t),t)
$$
- 其中：
  - $\mathbf z(t)\in\mathbb R^D$：网络在时间 \(t\) 的隐藏状态
  - $f_\theta$：神经网络定义的向量场（Vector Field）
  - $t$：连续深度变量，不一定具有真实物理时间的含义
  - $\theta$：需要通过训练学习的网络参数
- 最终状态通过求解初值问题获得：

$$
\mathbf z(t_1)=
\operatorname{ODESolve}
\left(
\mathbf z(t_0),f_\theta,t_0,t_1
\right)
$$
- 例如，可以使用 Euler、Runge–Kutta 或 Adams 等数值积分方法

- **Neural ODE 并不是不再需要离散计算，而是不再把离散计算步骤本身作为模型定义的一部分**
  - 模型定义的是连续动力学，具体求解时仍然需要数值离散化
  - 这个区别直接带来两个结果：神经网络结构不再严格绑定求解步骤数；可以通过修改数值求解精度，自适应改变计算量

- 但上面的建模方式会带来一个问题：
  - 网络训练时需要反向传播
  - ODE求解过程理论上可以拆成很多离散步骤，然后使用自动微分
  - 但这样还是需要保存大量的中间状态，并且计算图依赖于具体的求解器的内部实现
  - Neural-ODE希望达到低内存训练目标，因此需要构思专属的反向传播机制
 
- 这篇文章在计算层面最重要的贡献，就是引入伴随敏感度方法，将梯度计算表示为另一个微分方程的求解过程：**Adjoint Sensitivity Method：把反向传播也变成 ODE**
- 定义伴随变量，它描述最终损失对时间 $t$ 处隐藏状态的敏感度：

$$
\mathbf a(t)=
\nabla_{\mathbf z(t)}\mathcal L
$$
- 其演化方程为：

$$
\frac{d\mathbf a(t)}{dt}=-
\left(
\frac{\partial f_\theta}{\partial\mathbf z}
\right)^{T}
\mathbf a(t)
$$
- 上面的这个伴随方程可以根据欧拉离散推导得到，这里不做展开，具体方式时求解隐藏状态的Jacobian矩阵，再通过链式法则得到
- 有了伴随方程，最终的参数梯度为：

$$
\frac{d\mathcal L}{d\theta}=
-\int_{t_1}^{t_0}
\left(
\frac{\partial f_\theta}{\partial\theta}
\right)^T
\mathbf a(t)\,dt
$$
- 引入伴随变量的意义就在于此：
  - 朴素的积分求参数梯度的过程中，需要保存中间所有过程的隐藏状态，然后求解这些隐藏状态关于参数的梯度
  - 但伴随法不需要，它只需要保存一个新的由伴随方程ODE确定的新状态 $a(t)$和对当前 $\mathbf z(t)$的网络参数求梯度即可
  - 这样求解梯度的全部过程就变成了：
    - **正向计算的时候一步一步得到中间的隐藏状态，然后通过一个固定求解器来求得最后的结果**
    - **反向传播的时候通过上面公式的反向积分来计算**
    - **反向积分中真正只需要伴随变量和每次输入不同的时间对应的隐藏状态下的网络输出结果关于当前固定参数的Jacobian矩阵**
    - **因此最简单的伴随敏感度方法就是存储中间的状态，然后计算对应的Jacobian矩阵，另外反向积分的时候伴随变量也是一个ODE的求解过程，因此我们在最后一次状态的时候能直接通过网络计算得到a(t)，随即通过ODE一步一步求解即可得到中间的伴随变量**
    - **具体来说，如果使用自动微分+ODE求解器来求解参数梯度，求解器每一步做的加法、乘法、RK 阶段更新，全部被 autograd 记录进计算图**
    - **但在伴随法中，我们先前向阶段跑一次，得到最后的状态变量；然后在反向阶段，我们扩充一个联合ODE，同时求解梯度积分中需要的VJP和伴随状态更新需要的变化ODE，调用两次VJP来求解这个结果，然后求解器推进伴随状态和梯度累加器更新，最终获得梯度结果**
- 反向传播可以不依赖求解器的内部计算图，只需要调用向量场及其 Vector–Jacobian Product（VJP）
- 这个过程中，我们仍然需要保存中间状态方便网络计算中间伴随状态和Jacobian矩阵，但不需要保存网络的中间状态和计算图

- 文章进一步提出Continuous Normalizing Flows
- 当概率分布随着 Neural ODE 连续演化时，我们如何计算其概率密度？
- 在一般 Normalizing Flows 中：

$$
\mathbf z_{k+1}=F_k(\mathbf z_k)
$$
- 对应的概率密度满足：

$$
\log p_{k+1}(\mathbf z_{k+1})=
\log p_k(\mathbf z_k)-
\log|\det J_{F_k}|
$$
- 现在将变换改为无穷小的 Euler 步骤：

$$
\mathbf z(t+\Delta t) =
\mathbf z(t)+\Delta t f_\theta(\mathbf z(t),t)
$$
- 其 Jacobian 为：

$$
J_F=
I+\Delta t J_f
$$
- 因此：

$$
\log p_{t+\Delta t}(\mathbf z(t+\Delta t))-
\log p_t(\mathbf z(t))=
-\log\left|\det(I+\Delta t J_f)\right|
$$
- 这里的关键是一个矩阵分析恒等式：

$$
\left.
\frac{d}{d\epsilon}
\log\det(I+\epsilon A)
\right|_{\epsilon=0}=
\operatorname{Tr}(A)
$$
- 于是：

$$
\log\det(I+\Delta t J_f)=
\Delta t\operatorname{Tr}(J_f)
+o(\Delta t)
$$
- 代回原式，并令 $\Delta t\rightarrow0$ ：

$$
\frac{d}{dt}
\log p_t(\mathbf z(t))=
-\operatorname{Tr}
\left(
\frac{\partial f_\theta}{\partial\mathbf z}
\right)
$$
- 这就是论文 Theorem 1：**Instantaneous Change of Variables（瞬时变量变换公式）**

- CNF 不再计算每一次有限变换带来的完整体积变化，而是沿时间累积无穷小的体积变化率
- 对于连续向量场来说，$\operatorname{Tr}(J_f)$ 也正是其散度
- 因此，CNF将 NFs 中的概率密度建模问题，转变为了一个向量场散度的积分问题：

$$
\frac{d}{dt}\log p_t(\mathbf z(t))=
-\nabla\cdot f_\theta(\mathbf z(t),t)
$$
- 正向过程可以和原始的隐藏状态联立ODE求解，因为同样是NFs的延续，可以精确求解似然
- **CNF不再强制要求特殊的可逆层结构，只要向量场满足适当的正则性条件，使初值问题存在唯一解，状态演化映射在相应区间内就自然具有可逆性**

## 实现细节：
- 文章用监督学习验证连续深度网络是否可行
- 使用 MNIST 图像分类，比较传统 ResNet、直接反传的 RK-Net 和使用 Adjoint 方法的 ODE-Net
- 结果说明：
  - ODE-Net 可以达到与 ResNet 接近的分类性能
  - 在该实验结构下，ODE-Net 使用更少参数
  - Adjoint 方法降低了相对于求解步骤数的激活保存开销

- 原文 Section 4.1 主要设计了两个实验
  - 第一个实验是 Density Matching：作者使用二维目标分布，比较不同层数的 Planar NF 与不同隐藏单元数量的 Planar CNF
  - 原文 Figure 4 展示，随着模型容量增加，CNF 通常能更好地拟合这些复杂分布，并在实验设置下取得更低的损失
  - 这说明 CNF 能够通过增加向量场的宽度获得更强的表达能力，而无需像早期 Planar Flow 那样依赖大量单隐藏单元的串联变换
  - 第二个实验是 Maximum Likelihood Training
  - 作者在 Two Circles、Two Moons 等二维数据分布上训练密度模型
  - 原文 Figure 5 可视化了从初始分布到目标分布的连续演化过程：CNF 学习到的轨迹较为平滑，并且可以逆向求解用于生成样本

- 原文 Section 5 还提出了另一类重要应用：通过 ODE 描述潜变量的时间演化

## 站在现在的角度，回看这篇文章
- 在 Neural ODE 之前，NFs 的核心问题是如何构建可逆映射；在 Score-SDE 之后，生成模型逐渐转向如何描述概率分布随时间的演化
  - Neural ODE**将学习一个可逆变换的目标转换为了学习一个变换向量场**
  - FFJORD是研究CNF的下一个工作，弥补了计算上的问题
    
- 如何思考Neural ODE和Score-SDE之间的关系：
  - SDE 中的样本轨迹具有随机性；其概率密度演化由 Fokker–Planck 方程描述
  - 这里不做展开，抓住一个点即可：**ODE的时间边际概率分布的形式和SDE的Fokker–Planck方程有对应关系，只需要相应的修改扩散项中的一部分，就能做到完全相同的时间边际概率分布，此时从一个概率到另一个概率的转换路径虽然不同，但对应的离散点上的数据分布完全相同**
  - **上述就是我们希望达到的效果，即用一个固定的ODE轨迹，能获得和原来SDE完全相同的时间边际概率分布**

## 参考文章和博客，感谢他们的分析解读

