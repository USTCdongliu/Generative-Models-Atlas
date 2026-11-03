# NCSN

## 发表情况
- Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32.
  - NeurlPS 2019
- 主要作者：Yang Song，Stefano Ermon
  - Standford
- 截止2026.10 cite：7618+
- 提出通过估计数据分布的对数梯度（分数），利用朗之万动力学采样实现生成建模
- [原文链接](https://proceedings.neurips.cc/paper_files/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf)

## 摘要解读：


## 背景说明：
- 这篇工作主要借鉴Energy-based Models相关的工作，以及在采样阶段引入了物理学的基于朗之万动力学的朗之万采样，背景部分的解读中主要分析EBM相关的前作，朗之万采样默认成立且说明本文在原始朗之万采样基础上的改进

- 2005年 A. Hyvärinen. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(Apr):695–709, 2005.
- 这篇文章是EBM理论史上非常关键的一篇论文，他解决的是如何训练一个"无法归一化的概率模型"
  - 在建模概率分布时，一个非常关键且绕不开的问题是，概率模型的归一化常数在高维情况下根本无法计算，原因是需要充分在高维空间中计算才能得到一个精确的归一化常数；归一化常数缺失导致在求解优化函数的梯度时没办法计算
  - 这篇文章的关键在于，**不去拟合PDF本身，而是去拟合PDF的梯度**，PDF梯度的计算过程中归一化常数直接为0，因此可以直接去掉归一化常数的部分
  - 下面的推导中有一部分是错误的，即不是假定各个维度之间相互独立，而是在方便计算的情况下合理拓展相关推导结论
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/1.jpg](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/1.jpg)
  - 上述推导说明了以下结论：
    - 通过引入PDF的梯度，可以不再需要知道真实数据的分布即可获得一个可行的优化目标，但当前推导的计算目标需要计算PDF的Hessian矩阵迹，对于高维的图像生成任务几乎无法计算

- 2008年 Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008, July). Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th international conference on Machine learning (pp. 1096-1103).
- 这篇文章是DAE(denoising autoencoder)的早期工作，没有太多的理论解释，但提出了降噪和生成之间的一些观察现象
  - 文章的出发点是**表征学习**，08年的时间节点大家都在探索如何学习自编码器进行表征学习，但从x到x的端到端学习容易坍缩，即latent code和x数据一一对应，学不到任何有意义的表示
  - Vincent的想法是在输入中加入噪声，训练目标是将带噪图像还原为原始图像，网路在完成去噪的同时必须学习数据生成规律

- 2011年 Vincent, P. (2011). A connection between score matching and denoising autoencoders. Neural computation, 23(7), 1661-1674.
- 这篇文章从理论上分析了DAE和生成概率模型之间的关系并指出了一条切实可行的优化路线，但在当时受限于计算资源和网络结构等问题，没有办法实现高维度的生成任务
- 具体的结论如下：
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/2.jpg](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/2.jpg)
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/3.jpg](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/3.jpg)
- 上述简化推导说明了以下结论：
  - 最佳去噪器的去噪方向，并不一定恢复某个唯一的真实样本，而是返回给定噪声观测下的后验均值，也即特定噪声强度下的脏数据的PDF上升最快方向
  - 上述结论也就是Tweedie's formula 在加性高斯噪声下的形式
  - 当噪声强度约为0时，可以认为是真实数据PDF的上升最快方向，同时这个过程中必须有噪声，因为如果完全没有噪声，梯度上升方向一定固定，此时从任意脏数据出发都会达到PDF的最高点，这就不是分布了而是一个固定的优化路线
  - **如果将上述结论中的PDF梯度(也就是后面的score function)作为一个网络去拟合，我们就可以走一步运行一次这个网络，等到基本噪声为0时，这就是我们想要的真实数据的PDF，这是我们现在的出发点和目标**，但现在我们并不知道真实的加噪数据的PDF梯度是什么样的，因此就没办法用这个数据去监督
  - 第二张图的推导就解决了这个问题，最终的结论就是使用Fisher identity为PDF梯度输出网络搭建一个真正可以学习的目标(这和前面的后验期望并不严格等价，但在一部分常数可以忽略的情况下，可以认为两种优化目标获得的结果是相同的)
  - 上述网络学习目标的期望是一个条件后验期望(代入最佳去噪器中)，也即最终的目标函数，这说明了**通过去噪的方式，同样能实现一个拟合数据分布PDF梯度的良好网络**，这也是后续一系列扩散模型工作的实现方式，去噪=生成

## 具体分析：
- 文章当中的两个关键因素是**score matching**和**Langevin dynamics**
- 从这篇文章和Sliced score matching: A scalable approach to density and score estimation，大家都会将扩散模型预测的这部分称为score function，即为PDF的梯度(与后续的预测噪声仅差一个缩放常数)

- score matching即承接第一段对于背景2005年文章的分析，可以通过真实数据分布的海森矩阵迹来计算，这是跳过真实数据分布的一个优化方式
- 但难点在于海森矩阵就是二阶导了，在高维情况下这个矩阵的迹基本上是无法计算的
- 通过这个思路继续计算需要进一步的优化，这里使用的是宋飏前期一篇文章 Sliced score matching: A scalable approach to density and score estimation
  - 大致思路是引入了一个同维度的简单分布随机采样的向量，对海森矩阵一项进行矩阵乘法运算，将矩阵运算直接写成一个标量，通过自动微分来计算这个值
  - **另外一个思路就是延续Vincent2011年的思路，去预测一个脏数据分布，当噪声的强度很小，可以近似为预测的一个真实数据分布**

- 在真正的生成过程中，假设我们能获得训练好的score function network，但仍需要一个具体的采样过程来让数据从原始我们假定的一个简化分布，一步一步通过计算score function来到达真实数据分布
- Langevin采样即学习朗之万动力学的原理，可以从一个简单分布逐渐优化到一个预期的复杂分布，在这个过程中，只需要一个随机采样的标准高斯分布噪声和真实数据PDF的对数梯度(即score)就能实现
- 原文的公式为：

$$
\tilde{x}_t = \tilde{x}_{t-1} + \frac{\epsilon}{2}\nabla_x \log p(\tilde{x}_{t-1}) + \sqrt{\epsilon}\, z_t,
$$
- 在采样算法和具体的优化目标确定后，NCSN的整体生成思路就很明确了，即：
- 先用一个网络能比较好的预测score，在推理的时候采用朗之万采样进行分布拟合，但这个范式有一些实现上的问题：拟合score的方式这里提到了两种，使用哪种更好；真实数据的分布是否符合理论；朗之万采样是否能够获得比较好的采样效果等等

## 实现细节：
- 这里文章对score-based generative modeling的建模难度进行了分析：
  - 首先是真实数据流形分布的假设，真实数据流形通常会是一个低维的情况，低维流形带来了两个问题：
    - 一是score是分布在整个高维空间上的数值，但如果低维流形会导致在很多的空间中score未被定义；
    - 二是在EBM的海森矩阵迹的假设中，要求真实数据PDF分布在整个高维空间中，低维流形同样不满足上述要求
  - 第二是在真实数据PDF很低的情况下也会出问题：
    - 因为score都是在采样中进行训练的，自然在PDF小的地方很可能采样不足甚至没有采样到数据，这样导致在这些地方的score预估会非常不准；
    - 文章中用两个二维高斯的混合来描述了这个问题，即slow mixing of langevin dynamics，两个高斯分布的相对强度不同，但因为在小PDF位置的score预估不准确，在固定采样次数的情况下，朗之万采样能还原两个高斯分布，但无法正确反映两个高斯分布的相对强度
   
- 上述问题其实都能归结到真实数据流形是低维上，如果我们能让这个数据的分布弥散到整个高维空间，并且在大多数位置的PDF都不小，这样采样就能充分且全面，这样一个很自然的想法是：
  - 使用随机高斯噪声扰乱数据分布，高斯噪声的引入使脏数据的分布不会局限于低维，且会填充原始数据分布中的低密度区域；
  - 开始引入大噪声，此时能充分填充高维数据空间
  - 之后逐渐引入更小的噪声，在理想情况下，最终的score和真实数据的score几乎无法区分

- 如果使用去噪的方式构造优化目标，本身加噪就是优化当中必不可少的一个步骤，因此整体实现很自然，开始在原始的图像上加入较大的噪声作为输入，之后一步一步减小噪声，最终几乎获得完全无噪声的图像
- 如果使用Sliced score matching也是可以的，因为Sliced score matching本质上是学习score function的方式，但从大到小的噪声也本身是一个朗之万采样的一个过程，加噪的大小对应着不同的数据分布，可以通过朗之万采样的方式计算每个加噪时间节点的score来进行变换采样
- 在NCSN中，和后续的DDPM不一样的是，原始的图像因为是加噪的形式，所以最开始的输入也是一张原始图像，但加上了方差相当大的噪声，而不是完全的标准高斯噪声

- 因为单个网络需要完成多个不同噪声强度的score matching，所以训练的loss是一个加权和，加权系数文章建议取成噪声方差，目的是平衡不同噪声强度的loss在总loss中的权重

- 朗之万采样中非常重要的影响因素是单步更新采样的步长，因此NCSN中对朗之万采样进行进一步的改进，文章中称为**退火朗之万采样**，如下图：
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/4.png](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/4.png)
- 大致的做法就是设置两个超参数L和T，L是不同的噪声强度数，T是单个强度下的采样次数
  - 算法思想就是通过对应噪声强度的方差来控制采样步长，但每一次的不同分布变化的采样次数相同
  - 这样从最强噪声引入的脏数据到次一级噪声引入的脏数据，中间会采样T次，以此类推这样的采样会做L次，最终得到想要的真实数据分布
  - 这里也有一个可优化的空间，就是对应步长的选择，文章中选择方差的正比，目的是为了让单次朗之万采样中的score和噪声的SNR大致保持在一个常数，不受到方差的影响

## 站在现在的角度，回看这篇文章


## 参考文章和博客，感谢他们的分析解读

