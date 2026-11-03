# DDPM

## 发表情况
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. Advances in Neural Information Processing Systems, 33, 6840–6851.
  - NIPS 2020
- 主要作者：Jonathan Ho，Ajay Jain，Pieter Abbeel
  - UC Berkeley
- 截止2026.10 cite：50486+
- DDPM 通过前向加噪、反向去噪的马尔可夫扩散过程，以预测噪声的简化目标完成训练，建立了现代扩散生成模型的基础框架
- [原文链接](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)

## 摘要解读：


## 背景说明：


## 具体分析：


## 实现细节：


## 站在现在的角度，回看这篇文章
- 在VAE的基础上，DDPM完全可以看作是一个多层级的VAE，基本思路是一致的：引入隐变量在隐空间表示数据-对应加噪的中间值，ELBO推导出的优化目标-首先使用given原始数据的后验概率替代无法直接采样计算的反向去噪条件概率，之后使用网络代表的好计算的分布进行变分推断替代真实的后验概率，之后的loss的后两部分也可以完全对应到MAP建模中的重建数据回归和后验概率的还原，完美对应到MAP的先验和似然，因此是一个贝叶斯网络的体现

- 但DDPM中又有Flow-based models的影子，加噪中间量和原始数据的维度相同，整个的加噪和去噪过程也可以看成是一个可逆过程，但网络本身的结构并没有做太多的限制，可逆是通过马尔科夫链实现的，后期的Flow-based models也开始加噪和去噪的过程，但flow的加噪是为了让离散的像素分布更符合连续分布的假设所设置的，但加噪和去噪已有雏形

## 参考文章和博客，感谢他们的分析解读
- [扩散模型之DDPM - 小小将 - 知乎](https://zhuanlan.zhihu.com/p/563661713)
  - 这篇针对DDPM的公式推导和实现都有非常详细的介绍
