# DDS

- Chung, H., Lee, S., & Ye, J. C. (2024, May). Decomposed diffusion sampler for accelerating large-scale inverse problems. In International conference on learning representations (Vol. 2024, pp. 38922-38949).
  - ICLR 2024

- 在文章中提到VP和VE都可以，但最终验证的结果说明VP的效果更好

- 主要针对两类任务：MRI加速重建，主要对比Chung本人最开始的两个工作，以加速重建和多线圈重建为指标；3d的稀疏视角CT重建和有限角度CT重建

- DDS的核心是：每个扩散时间步先用 Tweedie 公式得到干净图像估计，再以该估计为初值运行少量 CG 数据一致性迭代，最后把原来的扩散噪声分量重新加回去，进入下一个噪声层。(加噪指的是本身反向SDE的过程中引入噪声，DDS中为了加快采样因此采用DDIM的框架，因此可以通过调控加噪策略的方差来控制是否是稳定采样路线)

- DDS想解决的问题：
  - Score-MRI、MCG、DiffusionMBIR 等方法经常需要数千次网络函数评估。尤其多线圈 MRI 中，旧版 Score-MRI 还要分别处理不同 coil 和实虚部，实际 NFE 非常高
  - DPS等算法的主要计算瓶颈是需要反向传播通过整个 score/denoising 网络，既耗显存又耗时间，而且在低噪声或强约束情况下容易数值不稳定
  - 因此这个工作解决的问题一个是慢，一个是计算量大

- DDS的原理分析：
  - 作者希望将CG等传统算法和DPS等更新做结合，介绍了Newton-Krylov method等求解非线性逆问题的方式，主要就是利用Krylov子空间算法减少单步迭代的计算量
  - 使用DDIM的方式改变采样中的加噪策略，我怀疑是作者当时阅读了DDIM想到了可以用到加速采样中，但之前的工作也有这方面的尝试，并未在求解逆问题的采样框架中获得成功
  - 从原文的3 DECOMPOSED DIFFUSION SAMPLING的Key observation开始到本章结束
  - 主要的观点是Proposition1，该观点的含义是DPS关于用一步去噪预测结果求似然梯度的部分中，最为关键的是预测图像关于当前加噪结果的梯度，即Jacobian矩阵，在局部情况下可以看成是一个坍缩到低维数据流形的正交矩阵，原文公式(20)的内容就是预测图像减去乘上某个比例因子的似然梯度的结果等于原始数据流形上的投影，预测图像处的数据流形切空间，恰好可以表示为某个Krylov仿射子空间，CG算法中的M步解在原始预测图像的基础上也属于这样的一个张成空间
  - 论文据此声称不再需要显式计算 MCG，普通 CG 就可以替代切空间投影梯度
 
- 关于此原理的分析：
  - 本身公式19和公式20的成立依赖于数据流形是仿射子空间，数据在该子空间上均匀分布，socre是最优的等等
  - Krylov 子空间由正问题算子、测量数据、初始残差等等决定，因此很大概率和流形的切空间是不等的，因此此原理更好的说法是两者具有较大的重叠
  - 对上述结论可行性的可能性分析：
    - Tweedie estimate 是优质 warm start
    - CG 比一步梯度下降更好利用算子谱结构
    - early-stopped CG 本身就是正则化，CG的迭代步数不会很大不会强行拟合
    - 数据一致性在 clean domain 中更合理，对Tweedie estimate进行正问题约束本身就比对中间图像约束更强

- 文章中反复提到了一个工作DDNM，Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=mRieQgMtNTQ.
  - DDNM和DDS比较大的区别就是同样使用Tweedie estimate，但DDNM使用伪逆更新，按照本文的结论作者批评DDNM的点在于伪逆更新不能保证迭代结果处在切空间内，从实践角度来说伪逆有可能没办法获得
 
- 文章还有一个核心贡献是对noisy measurement也有一些改进，主要是针对y = Ax + n这样的有噪物理模型不再强制满足Ax=y，而是求解：

$$
\min_{x} \frac{\gamma}{2} \| y - A x \|_2^2 + \frac{1}{2} \| x - \hat{x}_t \|_2^2.
$$

  - 即通过Tweedie estimate的误差进行正则化约束，CG此时的求解方式就是改写正问题算子，将两个l2写到一起，此时的正问题算子通常是正定的，但正则化系数是搜索出来的

- 针对有噪的MRI重建，用一个简单的MRI重建来举例：
- ![DDS (PI MRI; VP; noisy)](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/32.png)
- 针对三维的CT重建，和之前的DiffusionMBIR在计算上完全一致，但DDS的区别在于用Tweedie estimate进行ADMM+CG的硬约束计算，同时使用DDIM加速(DiffusionMBIR的求解器中其实还做了z方向的硬约束正则化投影)，如图：
- ![DDS (3D CT recon; VP)](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/33.png)
- 不管是上面的哪个任务，都单独将最后一次反向求解独立出来，并没有做正问题的约束，原因是本身约束建立在Tweedie estimate之上，最后一步默认去噪完成已经没有再计算Tweedie estimate的必要

- 本文的主要问题是理论假设太强，是一个强结论的重建算法设计，且本身Krylov子空间算法在null space的部分的误差也无法衡量
