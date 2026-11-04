# RED-diff

- Mardani, M., Song, J., Kautz, J., & Vahdat, A. (2024, May). A variational perspective on solving inverse problems with diffusion models. In International Conference on Learning Representations (Vol. 2024, pp. 28027-28053).
  - ICLR 2024
  - 区别于之前Chung和宋飏的工作，将预训练的DDPM作为正则化的部分，原文的说法是formulate sampling as optimization for inverse problems，区别于之前将正问题融入到条件采样生成过程中的做法，在形式上更类似RED之类将去噪器作为更强有力的先验的做法
  - 核心思想是通过变分推断获取优化结果，出发点和VAE和DDPM的理论类似，希望得到x_0 given y的概率，如果这个后验概率不好求得，就通过一个更方便获取的分布去拟合他

- 文章的主要对比对象是DDRM ΠGDM DPS
  - 文章批评ΠGDM和DPS的主要点在于：两个都非常依赖于一步去噪结果得到x_0 given x_t的分布，在大噪声情况下，given x_t很可能存在多解，用一个后验期望代替整个期望计算很大程度上会存在误差，文章中提到类似用一个简单的单峰分布在拟合多峰的复杂分布；ΠGDM更多的应用场景局限在线性或半线性的逆问题上(需要使用伪逆算子)
- 文章中提到了RED-diff和RED一类方法(regularization-by-denoising)的相似之处，从RED的角度来看，本文的算法类似用一个不同程度的去噪器做正则化，强度不同的去噪器起到了从语义到细节的约束作用；但文章也提到了和RED的不同：脱胎于变分推断、同时使用了多层的去噪(单次迭代中只有一个，但多轮循环中会重复随机)
- 文章的一个比较关键的地方在于权重的调整，如果观察RED-diff的伪代码和图示，本质上去噪器在计算类似预测噪声的误差，不同去噪步数的噪声强度不同，需要非常恰当的权重策略才能让正则化和优化效果变好
- 文章在相关工作中提到了当时在3D生成方向上的扩散模型，该工作和这个方向中的扩散模型很类似，3D重建在当时使用扩散模型的方式也是帮助训练NERF，他们让NERF生成的三维场景生成投影2D结果，再通过和扩散模型的生成结果做对比指导NERF重建，他在这里的意思就是，这些工作和RED-diff一样，反向的采样并不是解决逆问题的重点，单步的采样生成只是优化问题的一个约束，即扩散模型的反向采样不是算法的核心主角

- 文章使用VP SDE推导，使用ADM的框架代码

- 文章的核心推导如下：
- ![RED-diff的变分推断前提](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/50.png)
- 上述结论和VAE中的核心推导类似，但主角从最大化term(ii)变成了最小化KL散度，本质上还是不变的，由于在这个逆问题的框架中测量数据的PDF不变，因此设置一个新的q分布去拟合原始p分布的任务变成了优化term(i)的部分，仔细查看term1的两部分，实际上就是VAE中的概率最大化和先验引入，放到正问题中就是重建保真项和正则化项
- 接下来的第一个推论如下：
- ![Proposition1](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/51.png)
- 这个推荐建立在对term1的分解，前者就是按照PDF的形式计算了期望，后者用到了宋飏在Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. Advances in Neural Information Processing Systems, 34:1415–1428, 2021a.这篇文章的结论，大致的意思就是同一个反向SDE的演化过程中，两个PDF在反向SDE的KL散度的变化率跟两个score的平方误差成正比，具体可以参照附录A中的推导内容，因为在充分的反向SDE过程之后，两个分布都会变成完全相同的标准高斯分布，因此可以积分得到两个原始分布的KL散度的数值
- 上面的结论可以推导出公式8的结论：
- ![公式8的内容](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/52.png)
- 这里的假设建立在变分推断新构建的分布的方差为0的情况，也就是此时测量数据的后验分布探索到一个dirac分布，单一的x会确定唯一的y，这在文章的附录A的最后一部分对这个假设有进一步的分析，因为本身的q(x_0 | y)是一个已知的分布，因此按照特定的加噪策略可以写出q(x_t | y)的形式，结合score function就可以写成公式8的形式，公式7到公式8一个很关键的点在于reg部分变成了预测噪声和原始噪声之间的关系，在reg部分有测量噪声的方差的原因是将recon部分的分母写到了reg里面，相当于对整体的loss调整了若干倍，且优化的对象变成了均值u，如果要对变分推断新引入的后验分布进行精确建模，其实应该是同时优化均值u和方差\sigma，但本文的核心之一也是假设方差为0，导致最终的优化目标只剩u，即在reg和recon的共同约束下确定唯一的逆问题求解结果
- 下面是RED-diff能真正work的关键推导：
- ![proposition2](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/53.png)
- proposition2的含义是，本身对于reg部分在迭代中对均值的求导是比较复杂的，但在一些前提下，即proposition提到的新后验分布的方差为0和原始加权方差的初始值为0的情况下，可以将reg的梯度表示为非常简单的误差形式，这使得本身对于reg部分的优化变得非常轻松，区别于DPS等方法需要计算预训练扩散模型的反向传播，RED-diff的计算难度大大降低
  - 方差的假设之前提及，重点是如何理解加权系数，本文的出发点是第二段划红线的部分，即作者人为地更改了加权策略，在原始加权系数的基础上新增了一部分，使得最终的加权系数发生了更改，此时在proposition2的\lambda和具体的w(t)之间有对应关系，文章的最终落脚点是可以根据需要的reg和recon的正则化比例，倒推出符合对应关系的w(t)，因此实际不会去关心w(t)，只需要满足最后的\lambda即可
- 最后是关于权重机制的部分：
- ![权重机制](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/54.png)
- 这里的含义是，如果不考虑对权重进行机制调整，本质上在不同的时间步上，reg部分会通过不同强度的噪声来约束均值u，但不同强度的噪声导致reg部分在随机过程中会约束失效，权重机制通过引入一个SNR ratio将预测噪声等价转换为预测均值，即reg部分发生在图像本身的尺度上，基于一步去噪的尺度缩放系数来将正则化部分的约束缩放到图像尺度上
- 最终，RED-diff的伪代码如下：
- ![RED-diff的伪代码](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/55.png)

- 具体比较多的推导可以参照附录A，其中最后一部分对于方差为0的通用探讨，如果不假定方差为0，最终的优化loss也可以写出一个闭式解，此时reg部分的梯度就会涉及到方差，但作者认为方差为0是一个很好的假设，原因在于即便方差的值可以算出一个解，但高斯方式的加噪不会让一个具有稳定语义的图像就直接变成另外一张其他语义的图像，因此单纯的通过高斯噪声控制x的多样性不是一个很好的选择，因此作者认为这个假设是合理的

