# DPS

- Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., & Ye, J. C. (2022). Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687.
  - Hyungjin Chung的代表作 ICLR 2023 Spotlight 当年三星论文奖金奖
  - 为数不多在做减法的文章，回顾了CMG的算法过程，重新用Tweedie’s formula和条件score思考了使用扩散模型解决逆问题的思路，并且去除了影响采样路径的正问题强约束步骤
  
- 文章的卖点是能解决逆问题模型中存在噪声+可泛化到多种逆问题包括非线性逆问题的求解当中，因此本文设计了六类任务，包括：线性的有四类，分别是超分、图像修复、高斯去模糊、运动去模糊，后两者的本质都是通过blur核构建在整个图像上能够进行线性计算的矩阵算子；非线性的有两类，分别是非均匀去模糊核和傅里叶相位恢复，非均匀去模糊就是通过某类前向过程计算了一个非线性的模糊过程，无法直接得到一个类似前面任务的算子矩阵，傅里叶相位恢复就是通过DFT得到频域信息后，抛弃相位信息仅用幅值图还原原始图像
- 在MCG的基础上，本文的整体过程会很简单，仍然可以看作是利用Tweedie’s formula的一步生成后验期望，然后用正问题计算和测量数据的差求梯度，指导每一次的采样路径，同时去除了之前解决逆问题时都会使用的正问题强约束投影，整体通过贝叶斯公式的角度来重新诠释这个过程

- 文章通过VP SDE的方式构建加噪方案，使用ADM的具体加噪策略，总体上类似于DDPM，原始采样算法选择DDPM中的the usual ancestral sampling

- 文章的一个出发点是：之前的很多使用预训练扩散模型解决逆问题的算法的核心都是通过类似projection的step来硬约束中间生成结果满足测量数据，但投影类似直接计算L1或L2loss，一个强假设是前向问题中噪声必须要接近0，因为这个强约束到一个程度上就是在拟合噪声
- 如果不使用类似宋飏ICLR2021和ICLR2022里面的思路，对测量数据y也进行加噪扩散，则中间步的x和测量数据y无法形成良好对应，因此整体上不满足直接计算似然的梯度的前提(即贝叶斯公式分解的通过given中间生成结果计算当前测量结果的概率梯度)，因此需要重新考虑构建正问题约束和测量数据y与中间生成结果x之间的关系

- 文章的理论突破点如下：
- ![DPS的核心观点1](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/20.png)
- ![DPS的核心观点2](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/21.png)
- ![DPS的核心观点3](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/22.png)
- 上述所有内容的观点是：通过贝叶斯公式分解只关注似然部分，这部分可以通过全概率公式进一步分解，因此中间生成结果由原始的图像决定，因此可以直接略去进行进一步优化
- 通过Tweedie’s formula可以一步计算当前步骤对应的预测图像
- 注意观察上面的两个概念，预测图像是一个后验期望，即given中间生成结果的预测图像期望；而似然部分通过全概率可以表示为单独每一个预测图像的似然的期望
  - 这两个部分类似一个计算某个变量的函数的期望，一个先计算这个变量的期望然后直接计算一个标量函数结果，因此误差可以用Jenson不等式表示，在一般情况下，如果噪声越大，这个误差越小，因此可以说明在大多数真实的逆问题情况下，这个思路都是可行的
  - 因为预测图像是一个估计的原始图像值，因此可以用正问题的模型直接计算得到这部分似然梯度
  - 文章希望泛化到不同噪声种类，因此在3.2的部分用不多的数学推导说明了在泊松噪声和高斯噪声的两种情况下如何计算这个似然梯度(本质上就是服从的分布不同导致计算的PDF数学形式不同，最后表现在用什么范数计算误差和前面的调节系数不同)
  - 在两种噪声的PDF推导之下，具体的伪代码表示如下：
  - ![DPS的伪代码](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/23.png)
 
- 文章用一张图提到了MCG和DPS(本文工作)的区别，主要表现在删除了单独的投影硬约束，主要原因如图：
- ![DPS和MCG的区别](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/24.png)
- 这张图很形象的说明了单独的投影步骤很大程度上会让输入网络的加噪数据处在网络训练中从未见过的流形上，此时网络对score的预测不准确，误差就会累计；从我的理解角度看，更大的误差原因发生在硬约束导致整体图像并未处在一个同噪声水平的数据之下，导致网络生成次优

- 关于两个非线性任务：
  - 傅里叶相位恢复是用DFT前向构建频域数据，但是抛弃相位部分的数据最终只保留幅值观测数据，从而还原原始信息，因此正向算子就是在傅里叶变换的基础上加了一个绝对值，文章中在傅里叶变换之前加了一个oversampling的矩阵，大致的思路是没有相位这个逆问题严重病态，扩大网格精度来获取更多的幅值信息
  - 非均匀模糊是用了一篇文章思路，通过几个不同时间帧的图像做组合再进行非线性变换，因此在不知道所有不同时间帧的图像的基础上没办法还原图像，在有这样一个图像配对数据的情况下，根据Phong Tran, Anh Tuan Tran, Quynh Phung, and Minh Hoai. Explore image deblurring via encoded blur kernel space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11956–11965, 2021.这篇文章的思路，将网络分为两部分预测blur核和前向算子部分，还引入了DIP优化参数来做结构先验，如果在StyleGAN的基础上，还可以直接通过输入隐变量，最终优化隐变量的方式改变x，网络输出的目标是分布优化原始图像x和blur kernel，最终让前向算子网络计算得到的模糊图像和当前配对的模糊图像完全相同(配对数据训练blur kernel的网络，blur网络输出结果和x一起用前向算子网络预测y，这部分是训练，预测的时候是通过重新训练DIP或者StyleGAN的w来得到干净的x，此时因为只有y)
