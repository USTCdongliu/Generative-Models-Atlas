# P2L

- Chung, H., Ye, J. C., Milanfar, P., & Delbracio, M. (2023). Prompt-tuning latent diffusion models for inverse problems. arXiv preprint arXiv:2310.01110.
  - ICML 2024
 
- 这篇的立意会更高，不再局限于最开始的DDPM和NCSN这样的扩散模型，转而思考更大规模、更复杂结构设计的基础模型，尤其是Stable Diffusion 这类带文本条件的潜空间扩散模型作为逆问题的通用图像先验时应该怎么处理
- 本文的写作方式和行文逻辑都和之前的文章有区别，是作者在Google Research实习时期的工作
- 本文的答案是P2L，在每个反向扩散时间步中，根据测量数据在线优化连续文本 embedding，并周期性地将预测 latent 经“像素域数据一致性近端更新—重新编码”拉回编码器的值域

- 使用通用的生成模型解决逆问题时，一个很关键的问题是：越通用的生成先验，覆盖的图像模式越多，分布越宽；为了生成清晰、典型的结果，往往需要条件引导把分布压缩到更窄的区域，但这样会牺牲多样性。作者认为本身SD这类的文本引导模型就是在求解一个测量结果无法精确量化质量的逆问题求解框架，因此在使用SD作为通用先验时，正确的文本引导尤其关键
- 依托于LDM的特点，本文解决逆问题的卖点在于通用生成模型(不再使用一类数据训练生成模型) 高分辨率 以及使用LDM解决逆问题，任务包括超分、图像修复和去模糊

- 和本文提到的其他关于在LDM引导逆问题求解的工作的区别：
  - LDPS，即用DPS的思路直接在LDM中改变，对应于原文的公式(6)，作者认为LDPS主要的缺点在于忽视了文本embd的影响，即寻找更适合求解的C会对整体性能有极大的提高
  - GML-DPS，文章3.2详细对比了区别，甚至做了一个小实验验证了本文的优越性，主要的区别在于正则化调整误差的区别，GML-DPS主要使用固定点的策略，但VAE的pipeline决定了这个训练目标的收敛效果不是很好
  - PSLD在 encode-decode 过程中加入 measurement-subspace gluing，但对A的正问题算子有要求

- 本文的主要观点是：
  - 对于LDM之类的通用扩散模型作为通用先验时，文本embd也很重要，之前的工作多数使用null text prompt参与逆问题求解，但null也是一个具有明确embd的变量，寻找利于逆问题的C是提升性能的关键，因此作者希望通过实时优化的方式将C和单步去噪结果同时作为变量进行优化
  - 对于上述两个优化目标，作者使用类似吉布斯采样的策略，交替更新C和x_t
  - 由于在潜变量空间的缘故，DPS的解决策略进一步增大误差，因此需要周期性地通过将latent拉回编码器的值域
  - 使用LDM 1.4的checkpoint，在ImageNet和FFHQ上进行训练

- prompt tuning algorithm表示如下：
- ![prompt tuning algorithm](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/34.png)
- prompt tuning algorithm这一段想表达的意思是在引入VAE的空间变换和C的文本prompt之后，同样可以使用DPS进行目标转换，将似然部分的PDF转换为DPS一类可以计算的目标，DPS可以通过有噪声的前向模型计算，有这个基础条件之后，用交替优化的方式优化C和z_t
  - DPS这里的近似建立在几个条件之下：首先在RED-diff中分析过，函数的期望和变量期望的函数本身有比较大的差距，但DPS文章分析差距不会过大；其次一次去噪结果在score network次优时可能不符合预期，最关键的是现在有VAE的decoder存在，decoder的非线性会进一步恶化误差
- C update的解释如下：
- ![prompt tuning algorithm1](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/35.png)
- ![prompt tuning algorithm2](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/36.png)
- 上述图中红色部分指的是作者在这里进一步通过观测值y调整了一步预测z，大致表示为(其实效果一般)：
- ![prompt tuning algorithm3](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/37.png)
- z_t update的解释如下：
- ![z_t update](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/38.png)
- 上述交替优化的一个很重要的点在于
  - 很多时候当前的去噪结果z_t和预测的z_0都会被一些观测梯度修正，此时更新后的latent可能处在不是Encoder生成的范围内，VAE和LDM训练时没有见过的位置
  - 不管是在C的更新还是在z_t的更新中，都会将预测的z_0经过decoder，然后去计算正问题中的误差，当decoder输入未曾见过的输入时，可能会出现比较大的误差
  - 文章的做法是周期性的约束预测的z_0，具体的做法如下：
- ![PROJECTION TO THE RANGE SPACE](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/39.png)
- 和前面重点提到的GML-DPS相比，GML希望通过encoder+decoder的方式让这个编码解码的过程约束z_0符合原本的生成方式(如果当前VAE能够识别这个z_0，那么它经过一次decoder再encoder应该还是它)，但很大的问题在于这个优化结果并不收敛，且VAE并不严格满足编码解码还是它的规则，因此P2L想到的约束方式是，我直接在图像域约束就好了，拿encoder输出这个结果一定就是encoder能够识别的了，具体的函数就是上面的公式(16)

- 整体上，文章的伪代码如下：
- ![P2L的伪代码](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/40.png)

- 理论并不严格；从embd的角度分析prompt对逆问题的影响，embedding 可能不是“描述图像内容”，而是成为控制 U-Net 的自由高维变量；prompt tuning要求的计算量很高
