# Score-SDE

## 发表情况
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations.
  - ICLR 2021
  - Outstanding Paper Award
- 主要作者：Yang Song，Jascha Sohl-Dickstein，Diederik P. Kingma，Abhishek Kumar，Stefano Ermon，Ben Poole
  - Kingma是VAE的作者
  - Google Brain
- 截止2026.10 cite：15745+
- 用**随机微分方程统一分数生成模型与扩散模型**，提出前向加噪 SDE 与反向去噪 SDE，设计 Predictor-Corrector 采样框架，实现高质量图像生成
- [原文链接](https://arxiv.org/pdf/2011.13456)

## 摘要解读：
- Score-SDE出现在NCSN和DDPM之后
  - NCSN中确立了以去噪为训练目标和退火朗之万采样的具体采样生成方式
  - DDPM则和NCSN不同，他明确的存在一个逐步加噪到去噪的过程，相当于在NCSN的整体流程中增加了一个正向加噪的过程

- Score-SDE建立在DDPM和NCSN的基础上，通过SDE的方式分析正向加噪和反向去噪过程，并通过score function的方式指导反向去噪采样

- 摘要中明确的说明了Score-SDE的几个贡献点：
  - Score-SDE涵盖了以往基于得分的生成式建模和扩散概率建模方法，并由此衍生出新的采样流程和建模能力--这一部分通过数学理论说明
  - 引入了一种“预测-校正”（predictor-corrector）框架，用于修正离散化反向时间 SDE 在演化过程中产生的误差--通过结合NCSN和DDPM的采样算法，进一步改进了如何精确求解真实数据分布的采样算法
  - 推导出了一个等价的神经常微分方程（Neural ODE），它不仅能从与 SDE 相同的分布中进行采样，还支持精确的似然计算，并提升了采样效率--这是后续很多扩散模型改进和更SOTA的采样算法的开端
  - 提供了一种利用基于得分的模型解决反问题（inverse problems）的新途径，并通过类别条件生成、图像修复（inpainting）和图像上色等实验进行了验证--这是扩散模型和逆问题结合的一个重要开端

## 背景说明：
- 背景就是SMLD和DDPM
  - SMLD（Score matching with Langevin dynamics）
    - 详见[https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/diffusion-series/NCSN.md](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/diffusion-series/NCSN.md)
  - DDPM（Denoising diffusion probabilistic modeling）
    - 详见[https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/diffusion-series/DDPM.md](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/diffusion-series/DDPM.md)

- 原文用两部分公式说明了SMLD和DDPM在优化目标和采样算法上的区别：
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/5.png](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/5.png)
- 在NCSN的基础上很好理解上述表述，和NCSN原文的差别在于加入了用于调节综合训练目标的权重系数、退火朗之万采样采用的新的采样算法和步长
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/6.png](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/6.png)
- DDPM的这部分推导就需要结合NCSN和DDPM两篇文章一起看，上面推导的符号和DDPM原文有一点出入，但不影响理解，具体来说，原始DDPM的预测目标是噪声，结合当前加噪数据的高斯分布的形式，可以写出具体的score function形式，内部的预测结果类似于一个服从 $(0,I)$ 分布的噪声，这和DDPM预测一个随机噪声的形式正好对应
  - 只有乘上前面的系数 $1 - \alpha_i$ 之后，最终预测的才是服从 $(0,I)$ 分布的噪声，因此这是预测噪声的目标所自带的权重系数
- score-SDE将DDPM的采样方式称为先祖采样（ancestral sampling），原因是明显有一步一步的趋势，并且当前的每一步都是完全不同的分布(由于加噪的形式)，这和NCSN的采样略显不同，NCSN的每一步是因为score function的调制才出现的不同，但本质上他不是原始设定的不同方差之间的直接转换
- 在上述分析基础上，Score-SDE想到将两者的采样方式进行结合

- 通过观察SMLD和DDPM的范式可以得到的结论是：两者在优化目标上虽然各自的展开形式不同，但能统一在一个理论背景中，两者的优化目标只有加权系数上的差别；同时，两种生成方法的采样过程各不相同，SMLD采用退火朗之万采样，更注重在单个分布上的拟合，不同的加噪步骤，直接用前一次的加噪的最后一次结果充当下一次的加噪的第一次结果，而DDPM则不在意单个加噪程度下的数据调整，只做不同加噪频率的数据转换

## 具体分析：
- 作者发现 SMLD 和 DDPM 都是离散噪声链，既然它们都在估计 score，那把离散噪声步取连续极限，得到连续扩散 SDE；再用 Anderson 逆时 SDE 公式，就发现二者都是这个 SDE 的不同离散化，于是 SDE 框架自然统一了两类生成模型
- 由此引出文章的下一部分，即通过SDE的角度来统一当前的生成模型理论，如图：
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/7.png](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/7.png)
- 上述的图片解释了SDE方程的具体形式，需要说明的是，这个公式是一个简化的SDE形式，等式右边的第一部分称为漂移项，是前向SDE在变化的时候固定的一部分，第二部分称为扩散项，是SDE变化中具有随机性的一部分，等式中的g(t)是一个矩阵形式，但在上述展示中简化为一个标量，同时 dw 表示的是一个标准的Wiener过程，在热力学中由布朗运动的表示，即粒子的随机移动部分，这部分的微分量纲和漂移项是不同的，原因是在布朗运动中，随机扩散的方差由时间决定，这导致本身的运动只是标准差大小，即根号2的dt

- 站在连续时间的SDE理论基础上，我们可以对之前的SMLD和DDPM做一些离散化的分析，将时间定为（0，1）之间，具体的推导如下（附录B的内容）：
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/8.png](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/8.png)
- 上述的推导最终可以写成下图的具体SDE形式，作者还加入了一种sub-VP SDE，它保留了 VP 的 drift，但采用更小的扩散强度，从而使条件扰动方差在中间时间受到 VP 对应方差的上界约束，并在实验中取得更好的似然结果
  - 三种具体的SDE实现都是连续SDE形式下的不同离散化方式，同时三种SDE都具有良好仿射性质的drift部分
![https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/9.png](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/9.png)
- 值得注意的是，这三种前向SDE的具体形式其实已经能说明各自的特点，VE-SDE的信号均值不收缩，噪声方差逐渐增大，直到填满数据空间；VP-SDE的信号逐步衰减，方差趋向有限稳态；sub-VP-SDE的条件扰动方差更小，采用比VP-SDE更小的diffusion项

- 在现有的三种SDE构建方式下，可以进一步分析反向SDE
- 既然生成过程已经被定义为反向 SDE，那么原则上就可以直接使用 Euler–Maruyama、随机 Runge–Kutta 等现有数值方法进行采样
- 这使生成算法不再局限于 DDPM 原有的 ancestral sampling 或 SMLD 的 Langevin Dynamics
- 这一部分的最终含义是：**采样器不再必须与某一种训练方法绑定，只要知道前向 SDE 和训练好的 score，就可以尝试不同的反向数值求解方法**
- 这里文章提出了几种具体的采样求解器，包括后面的Predictor-Corrector Samplers、由DDPM连续化获得的更优采样器reverse diffusion samplers、由ODE获得的probability flow sampling，这些需要结合ODE等点来统一讨论，此处不做展开

- 具体来说，这里的主要贡献是**Predictor-Corrector Samplers**
- 普通的 SDE 数值求解器只是沿着反向动力学推进样本，而 Score-SDE 还额外拥有一个重要信息：每个时刻的概率分布 score
- 由于 score 也可以直接用于 Langevin MCMC，因此在每一步使用数值求解器推进状态之后，还可以通过 MCMC 对当前样本进行修正，使它更接近该时刻的目标边际分布
- 由此形成了 Predictor-Corrector 采样框架：Predictor 负责推动样本从一个时间点移动到下一个时间点，Corrector 则负责在当前时间点改善采样结果的分布匹配程度
  - 在此基础上，**原始 SMLD 可以看成没有实质 predictor 的特殊 PC 方法**
  - **DDPM 可以看成没有 corrector 的特殊 PC 方法**

- **Probability Flow and Connection to Neural ODEs**是正文的另一大贡献
- 对于给定的前向扩散过程，可以构造一个确定性 ODE，使得它在每个时间点都具有与原 SDE 相同的边际概率分布。该 ODE 的 drift 可以直接由原始 SDE 的 drift、diffusion 和 score 确定，因此不需要额外训练生成模型。它与反向 SDE 的主要形式区别是 score 修正项的系数从 \(g^2(t)\) 变为 \(\frac12g^2(t)\)，同时去掉随机噪声项。由于这一 ODE 的向量场由神经网络估计，它也是一种 Neural ODE，并可以利用连续变量变换公式计算模型的归一化概率密度，实现似然评估

## 实现细节：
- 我们在实现细节部分补充附录和实验部分的一些具体细节

- 


## 站在现在的角度，回看这篇文章


## 参考文章和博客，感谢他们的分析解读


