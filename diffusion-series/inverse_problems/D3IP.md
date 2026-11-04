#  D3IP

- Chung, H., & Ye, J. C. (2024, September). Deep diffusion image prior for efficient ood adaptation in 3d inverse problems. In European Conference on Computer Vision (pp. 432-455). Cham: Springer Nature Switzerland.
  - ECCV 2024
 
- 作为SCD的后续工作，主要和SCD进行对比，在作者看来，SCD主要有几个问题：每个切片单独适配进行LoRA微调；适配损失中只做一次 CG；实际采样中又做多次 CG；适配目标与真正采样器不一致；三维体数据成本随切片数 N 线性增长

- 文章的一个贡献是把 SCD 抽象成与具体 posterior solver 无关的 DDIP 框架
  - DDS的除了用CG替代Jacobian求解之外，一个核心贡献就是在固定先验给出的Tweedie estimate附近，寻找了更满足测量的数据一致性点，SCD则利用这个条件估计训练LoRA和替换参与反向采样的Tweedie estimate为conditional Tweedie estimate，因此 DDS 在SCD中不是竞争意义上的完全独立模块，而是可以作为 DDIP 内部的conditional Tweedie estimate估计器
  - 从上面的角度看，作者抽象出DDIP，DDIP不限定条件后验均值必须如何估计；DDS、DDNM、DPS 都可以作为内部 approximator
  - 作者还将adapter和采样阶段的CG次数做了统一
 
- 作者使用VE的probability-flow ODE进行推导，但实现是通过VP

- 作者在DDIP的基础上针对三维医学影像逆问题提出了D3IP
  - D3IP 将逐切片的扩散模型测试时适配改写为对整组三维测量的随机多切片(选择了一些，没必要全部去，作者发现到了一定的数量提升有限)共享适配，用一套体级 LoRA 在反向扩散过程中学习 OOD 结构，并可进一步结合三维数据一致性或元学习
  - 从DDIP的角度分析，ODE的反向采样关键在于如何求解每一个时间步的conditional Tweedie estimate，但预训练的扩散模型通常得到的都是Tweedie estimate，类似DPS之类的方法在处理OOD数据集时时通过正问题约束来纠正Tweedie estimate，但DDIP之类的方法(包括SCD)选择使用LoRA微调参数，让Tweedie estimate变成conditional Tweedie estimate
  - 具体来说，DDIP在每一个时间步中通过loss做有限步CG优化，再通过DDIM将conditional Tweedie estimate和原始score network预测的噪声方向放入到快速采样框架中
  - D3IP的核心贡献是：如果DDIP要处理三维体数据，只能按照切片方向LoRA处理多个不同切片对应的score network，但D3IP改成体级期望，工程实现上只能选择K个随机切片来近似这个期望

- 从统一框架的角度，延伸出两个版本：
  - 开始之前总结一下D3IP的base版本：
    - 原始的SCD分为了两个大部分，一个是通过LoRA实现的adaptation steps，具体的做法是通过一定的更新方式确定了conditional Tweedie estimate(CG求解的一个具体形式)，然后利用这个conditional Tweedie estimate制定了LoRA的loss指导参数进行更新
    - 另一个是利用相同的更新方式(同样CG求解)，在新参数上的score network上更新了conditional Tweedie estimate，然后利用DDIM采样
    - 上述两个部分最关键的就是conditional Tweedie estimate如何求解，D3IP的base版本相较于SCD的改进在于volume-size的adaptation，但base版本并不关心conditional Tweedie estimate的具体求解形式
    - 除此之外base版本还强调adapter和采样部分的数据一致处理应该要一致(按照SCD给出的伪代码其实是一致的，都是简化的一步CG)
  - D3IP(mbir)
    - 这里回忆一下DiffusionMBIR，mbir希望通过二维的预训练扩散模型解决三维的重建问题，因此基本思路是在每一次Tweedie estimate的基础上配合ADMM算法做约束
    - mbir的改进就是作者认为之前在SCD的conditional Tweedie estimate计算不好，使用mbir的方式，通过正问题+切片方向的TV正则化两重l2形式配合ADMM求解更好，也即公式(14)，此时需要对切片做连续采样处理
  - D3IP(meta-learning):
    - 在开始这部分之前，先介绍一下元学习：
      - 元学习需要知道的概念包括MAML FOMAML和Reptile
      - 元学习的核心目标是让模型习得快速学习新任务的能力，MAML 是基于优化路线最经典的元学习框架，FOMAML 和 Reptile 都属于 MAML 思路下的简化衍生方案，存在清晰的包含与演进关系。MAML 完整形式依靠双层嵌套梯度优化：内层针对每个任务利用少量样本完成微调，外层基于微调后的模型在同任务其余样本上的损失更新初始权重，整个过程需要求解二阶梯度；但二阶梯度会带来极大算力与显存开销，很难直接落地工程，因此诞生一阶近似版本 FOMAML，它直接舍弃二阶导数项，仅保留一阶梯度用于外层参数更新，大幅降低计算负担，是实际项目里最常使用的 MAML 变体，二者可以概括为：FOMAML 是对标准 MAML 做工程折中得到的近似算法。在此基础上 Reptile 进一步简化，彻底摒弃 MAML 这套内外循环嵌套求梯度的范式，不再需要区分支持集、查询集来构建元梯度，训练逻辑更加朴素：多次采样不同任务微调初始权重，最后把多次微调得到的参数变化量平均，缓慢更新原始初始参数。从实现链条来看：元学习是一大类问题范式，MAML 是其中基于优化的代表性算法，FOMAML 属于 MAML 的一阶工程实现版本，Reptile 是跳出严格 MAML 梯度推导、轻量化改造得到的同类元学习方法。
    - 这里的meta-learning版本就是借鉴Reptile，既然统一参数对所有切片基本都是次优，那么保存每一个切片对于参数的修改量，在修改量之前加上一个小于1的比例系数，更新完一轮后得到一个公共的更新后参数，如果需要提高质量，就可以单独再用DDIP更新单个切片的质量
    - 作者还在初始化时设置了一个比较具有实际意义的初始化值，并不是从头开始采样

- 整体结合的伪代码如下：
- ![D3IP的伪代码](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/44.png)
- DDIP D3IP和元学习版的图示如下：
- ![D3IP的图示](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/45.png)
