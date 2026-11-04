# DiffusionMBIR

- Chung, H., Ryu, D., McCann, M. T., Klasky, M. L., & Ye, J. C. (2023). Solving 3d inverse problems using pre-trained 2d diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 22542-22551).
  - CVPR 2023
 
- 基于几个思考：
  - 直接通过扩散模型作为3D重建逆问题的算法具有难度，生成过程需要在数据同维度的空间中，计算和内存成本极大
  - 直接使用2D diffusion的问题在于：每个 slice 的 posterior sampling 是独立的，z 方向没有耦合，所以 coronal/sagittal view 会出现结构不连续、毛刺、错位、shaggy artifacts
  - 传统MBIR算法的问题是：TV prior 太简单，只能表达 piecewise smooth / edge-preserving，不能充分表达“医学图像应该长什么样”。论文明确说 TV 虽然 edge-preserving 且 imposing smoothness，但落后于现代 data-driven prior，因为它太简单，不能 fully model how the image “looks like”

- 因此文章的主旨是：
  - 3D prior 不一定要完整学习，3D prior 可以被拆成“强 2D 数据先验 + 弱 z 方向模型先验”
  - 逐 slice diffusion 本身没有错，错的是缺少跨 slice coupling
  - TV 不应该加在所有方向上，只应该加在 diffusion prior 没覆盖的方向上
  - ADMM/CG 没必要在每个 diffusion step 完整收敛，只需要借助连续 diffusion trajectory 做少量更新
- 基于此，作者的主张是：通过ADMM算法约束3D重建逆问题，ADMM算法作为反向采样中的硬约束步骤，考虑到本身的重建算法有三重循环：本身反向SDE的采样器、ADMM的迭代步骤、在数据保真项的CG算法迭代次数，作者将ADMM的两个变量作为跨SDE iteration的变量，每次反向SDE只做一次ADMM和一次CG；2D扩散模型本身可以作为x-y平面比TV更强的先验，因此在ADMM中其实只需要关心在z方向上的正则化约束即可

- 文章的三个任务分别是稀疏视角CT重建、有限角度CT重建、压缩感知MRI重建(有限角度和稀疏视角的区别主要在于例如180度的投影视角，只采取90度的部分，即仍然是比较多的视角，但局限在一部分角度)
- 具体的解释如下，这里的测量仍然是图像域而非正弦图，其实测量仍然是正弦图，但可以做类似FBP或者反投影得到类似图像域的测量数据：
- ![DiffusionMBIR的三个任务](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/25.png)

- 作者有对比同样解决三维重建问题的基于扩散模型的算法，之前的工作在三维重建中偏向于使用点云估计和隐空间(缩小数据规模)两种方式规避数据规模过大，但前者不适用于医学图像，因为需要精确构建内部结构，后者仍然需要大量三维数据进行训练

- 使用VE SDE，使用ncsnpp模型进行训练

- 硬约束的部分主要根据以下公式来展开：

$$
\min_{x} \frac{1}{2} \| y - A x \|_2^2 + \| D_z x \|_1
$$

- 上述公式中的D_z主要是TV本身是从三个维度约束结果的，但现在只需要在z方向上进行即可，通过ADMM的拆解，整体的硬约束步骤可以写成如下形式：
- ![ADMM约束形式](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/26.png)

- 类似在主张部分的内容：当前的采样+硬约束面临三重循环，严重降低重建速度，尝试将ADMM中的约束算子构建为全局变量，基于的原理是后期单步之间的结果本身差距不大，因此可以写成整体的伪代码形式如下：
- ![DiffusionMBIR的快速伪代码形式](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/27.png)

- 局限和思考：
  - DiffusionMBIR展现出了一定的OOD能力，但很大程度上因为强约束才成立，如果OOD pathology 很强，同时 CT views 极少或 k-space 缺失严重，模型可能会 hallucinate normal anatomy
  - 如果真实 volume 在 z 方向存在强烈非连续变化，例如薄层结构、小病灶、突然出现/消失的血管或肿瘤边界，Z方向的TV可能会把它们当作 artifacts 压掉
  - 总体其实还是落到了2D的先验本身仍具有局限性，是在计算资源不足的情况下考虑的一个解决方案
