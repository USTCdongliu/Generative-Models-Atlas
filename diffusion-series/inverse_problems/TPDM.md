# TPDM

- Lee, S., Chung, H., Park, M., Park, J., Ryu, W. S., & Ye, J. C. (2023). Improving 3D imaging with pre-trained perpendicular 2D diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 10710-10720).
  - ICCV 2023
    
- 对比之前的DiffusionMBIR，解决的问题也类似，包括压缩感知的MRI重建、CT稀疏视角重建和MRI在z轴上的超分；其中z轴上的超分，文章中称为MR-ZSR是文章着重强调的任务，DiffusionMBIR在此任务上会失效
  - 文章总结其贡献在于它试图用两个互相垂直方向训练的 2D diffusion score model，近似构造一个 3D volume prior，从而避免直接训练 3D diffusion model 的高维成本，同时比单纯 2D prior + TV 的 DiffusionMBIR 更能约束跨切片结构
  - 在三个任务上达到了SOTA，并且实现了MR-ZSR这个任务
  - 可以作为非条件扩散模型生成无条件的三维体数据

- 使用VE SDE，从实现上对TPDM和DiffusionMBIR做一个进阶分析，首先两者的扩散模型训练类似，虽然TPDM需要训练两个score network，但训练单个扩散模型的时候都是通过将体数据拆分为切片数据，然后进行二维的切片数据训练，但最终在推理的时候是将一个体积内的所有切片作为一个batch进行训练，因为需要同时用正问题进行约束更新

- 文章的出发点是三维的体数据PDF可以表示为单一方向的PDF乘积
- 大致的内容表示如下：
- ![三维体数据的PDF近似1](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/30.png)
- ![三维体数据的PDF近似2](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/31.png)
- 上述两个表示的含义是3D 分布可以表示为两个方向的3D分布以某种乘积形式结合，原因是需要符合两个视角的切片视角，进一步假设每个方向的 volume distribution 又可以分解成独立 2D slice 的乘积，这个假设并不完全成立，因此单步更新其实和DiffusionMBIR是类似的，但文章因为不需要z方向上的TV进行求解，因此不再使用ADMM进行正问题约束，使用DPS进行后验似然约束
- 一个很大的问题是，每次更新的时候需要两个方向上的score计算，文章认为这个更新方式计算量较大，因此采用概率的方式进行替换，训练的两个视角的扩散模型有一个是主视角，一个是副视角，确定一个K的超参数，每K次中有一次用副视角进行更新，且副视角的更新不通过DPS更新，仅使用先验生成进行更新，表示如下：
- ![TPDM伪代码](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/32.png)

- 训练对数据要求很高，因为需要同时训练三个视角的数据，因此训练集不再是二维切片数据，而是必须完整能从垂直视角进行切分的三维体数据，并且TPDM 构造了一个由多个 2D slice prior 组成的 factorized pseudo-3D prior，这个假设的正确性也不太高
