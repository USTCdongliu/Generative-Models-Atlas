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
- 

## 具体分析：


## 实现细节：


## 站在现在的角度，回看这篇文章


## 参考文章和博客，感谢他们的分析解读


