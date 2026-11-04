# SCD

- Barbano, R., Denker, A., Chung, H., Roh, T. H., Arridge, S., Maass, P., ... & Ye, J. C. (2025). Steerable conditional diffusion for out-of-distribution adaptation in medical image reconstruction. IEEE Transactions on Medical Imaging, 44(5), 2093-2104.
  - TMI 2024
 
- 本文瞄准的问题是作为图像先验的扩散模型的训练分布和测试分布不一致，在医学影像上表现为对比度不同、设备不同、组织不同等等；固定 score 仍然把样本推向训练分布的高概率区域，即使有测量一致性项保证
- SCD的出发点不是彻底放弃原先验，而是在原始score的基础上增加测量数据y来进行在线学习，微调当前的score

- SCD首先建立在DDS的框架之上，通过DDS的采样算法增加模型微调
- test-time training解决OOD数据集的思路由来已久，但扩散模型存在多步不同噪声程度的去噪，无法直接使用test-time training来解决
- 文章用LoRA只在卷积层和注意力层注入低秩残差，增加的参数量极小
- 只分析网络参数微调的部分，LoRA很类似于Deep Image Prior 式单样本拟合，都旨在优化网络参数，利用观测数据y进行监督，这是一作的主要想法，一作作者长期使用DIP等思路解决医学影像逆问题

- 从采样的角度看，SCD完全就是DDS的复刻，总结SCD在DDS上的创新：SCD 在扩散逆问题的每个反向采样时刻，通过当前测量的数据一致性误差优化低秩 LoRA 残差，使固定的训练域 score prior 转变为针对当前 OOD 测量的自适应 prior
- 一个很关键的点是：文章并没有把适配后的 ϵ 用在 DDIM 更新的所有部分：适配后的网络用于预测一步去噪结果；但原始网络输出用于DDIM的噪声方向项，原因是作者认为LoRA过后的网络不再是一个合法稳定的多噪声去噪器，如果用于DDIM的轨迹预测可能会破坏生成轨迹(这一点存疑)

- SCD的大致流程图如下：
- ![SCD流程图](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/41.png)
- 上述图表述不是很清楚，主要的点在于上面的椭圆合成数据集是预训练score network的，不是推理阶段的OOD数据集，推理阶段的LoRA和DDS采样都只依赖本身的测量数据y

- 首先作者的思路是类似test-time training,在推理的时候微调网络适应OOD数据的需求，但LoRA的最大问题是需要设计一个loss知道残差部分的参数优化
- 之后关键的思考如下：
- ![SCD的关键loss设置和近似](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/42.png)
- 公式11是LoRA的loss设置，关键是这里的一步去噪结果是conditional Tweedie estimate，因此需要对Tweedie estimate中的score更改为后验的score，仍然按照DPS的思路拆分后验score，最终得到公式12和公式13，但公式13求解难度较大，原因在于jacobian矩阵需要通过score network的反向传播进行计算，这里对jacobian矩阵做了简化，除了系数外直接将其设置为单位矩阵，因此conditional Tweedie estimate变成了一个优化问题，这里可以选择用多种方式求解，公式(15)就给出了一步CG的更新公式
- 在上述结论的基础上，得到了单步的loss，此时可以根据loss进行残差参数的优化，通过Adam等优化器进行多轮迭代优化更新参数

- 上述是SCD最关键的创新点，但也存在很多的问题：
  - 去噪初期的噪声巨大，此时的conditional Tweedie estimate反而会出现很大的偏差，可能导致优化目标次优
  - 本身DPS的误差就不小，再加上jacobian矩阵的简化，导致优化目标可能也不符合去噪的轨迹
  - DDIM的更新中，conditional Tweedie estimate和噪声方向来自不同的扩散模型，一个是LoRA过后的一个是原始的，可能存在冲突
  - 这个conditional Tweedie estimate存在比较大的问题，更好的做法是在单次迭代优化残差参数的循环中，新增一个对conditional Tweedie estimate的迭代优化， 但文章选择了简单的一步CG，但考虑到计算复杂度等因素，这个处理方式也是正确的
  - DIP的思想就体现在，这个loss中没有GT的指导，只有观测数据y的指导，同样依靠网络参数的优化

- 整体的伪代码如下图：
- ![SCD的伪代码](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/43.png)
