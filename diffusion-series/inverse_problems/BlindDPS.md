# BlindDPS

- Chung, H., Kim, J., Kim, S., & Ye, J. C. (2023). Parallel diffusion models of operator and image for blind inverse problems. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 6059-6069).
  - CVPR 2023
 
- 对于本文，任务比方法更重要，本文主要针对的场景是blind inverse proble（也有说法称为bilinear problem），更准确的说，是forward degradation 的结构已知，但前向算子的具体参数未知的情况
  - 对于这个概念，比较准确的解释是我们知道观测图像 y 是由真实图像 x 经过哪一类退化过程产生的，但不知道这个退化过程里的具体参数
  - 第一个场景是blind deblurring，最典型的 blind deconvolution，motion blur kernel 往往是稀疏、细长、方向性很强的，比如相机抖动会形成一条轨迹状 kernel；Gaussian blind deblurring通常是平滑、中心对称、低频化明显的，类似平滑低通滤波
  - 第二个场景是Imaging through turbulence，大气湍流的真实物理模型很复杂，因此实践中常用 tilt-blur model，turbulence imaging 有三个未知变量，不仅有模糊，还有空间形变，第一类是 局部几何扭曲。空气折射率变化会让图像局部位置发生偏移，表现为局部 warp、tilt、形变；第二类是模糊。湍流和成像系统会导致点扩散函数扩宽
  - blind inverse problem比普通逆问题更难求解的原因是operator 未知导致问题严重非凸、耦合、病态；强耦合指的是尺度不唯一导致blur kernal和原始图像产生多解，例如两个变量可以一个放大k倍一个缩小k倍同样满足正问题
 
- 使用VP SDE和ancestral sampling

- 文章的立意：
  - 首先就是blind inverse problem本身的性质，即病态和强耦合
  - 传统先验不足以刻画真实分布
  - 传统 blind deconvolution 通常采用 coarse-to-fine 策略：先在低分辨率/粗尺度估计，再逐步细化；文章提到之所以这样解决blind inverse problem，原因在于强耦合的原因只在当前层级解决逆问题会出现高频细节误导 kernel 估计，但这个经典策略和扩散模型的反向采样匹配，这是文章在算法结束后提出来的想法，并不是作者最开始就这样构思的
  - kernel sparsity 仍然有用，仍然可以通过核相关的正则化约束这个强耦合问题

- 文章算法的思路：
  - blind inverse problem 不应该只给 image 建先验，也应该给 operator 参数建生成式先验；可以为每个未知变量构造一条 reverse diffusion，再用 likelihood residual 把它们耦合起来
  - 下面拿deblur举例，类似大气成像其实就是在deblur的例子上再扩展一个生成模型
    - 基本的原理如下公式：
    - ![BlindDPS的核心公式](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/28.png)
    - 上述公式建立在第一个公式的基础之上，即blur kernel和数据的分布独立，在此基础上，引入两个生成模型分别预测核和图像的先验分布，同时根据求导可以进一步化简结果，公式17和公式18的似然都可以通过一步去噪公式进行替换，基本思路还是DPS，然后根据正问题的数学模型计算这个高斯分布的score
    - 整体的BlindDPS解决deblur问题的伪代码如下：
    - ![BlindDPS关于Blind Deblurring的伪代码](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/29.png)
    - 关于伪代码的步骤，需要补充的就是在核的似然梯度计算部分，通过l0范数等方式再次引入了正则化
    - 步骤7是通过文章公式14的形式约束kernel的参数值，目的是解决kernel和图像的强耦合
