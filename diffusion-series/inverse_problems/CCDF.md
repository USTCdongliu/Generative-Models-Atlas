# CCDF

- Chung, H., Sim, B., & Ye, J. C. (2022). Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 12413-12422).
  - CVPR 2022
  - 基本思路是将初步通过网络预测得到的结果作为正向加噪的某一步按照加噪策略注入一定的噪声，再从对应的步数反向去噪得到最终的生成结果

- 文章在discussion部分也提到了SDEdit，这两个具有高度的思想重合，都是将一个具有明确语义(SDEdit中是stroke风格的guide图像，这篇文章中是一个初步通过网络监督学习到的预估初始值)作为前向加噪的初始结果，通过特定的加噪策略形成前向过程中的某一步中间结果，剩下的直接根据原始的反向过程来求解即可；作者在回答和SDEdit的不同时主要的观点是：SDEdit在编辑的任务中将初始时间设置在0.3-0.6之间，目的是大的时间能够扰乱当前stroke图像和真实图像的分布，但本文的出发点是加快逆问题框架的推理，因此通过初始图像使初始时间缩小到0.02左右

- 文章设置了三个任务，两个在自然图像中，分别是超分和图像修复，一个是MRI重建，在前面MIA2022工作框架的基础之上；三个任务中在自然图像的两个中选择通过Improved denoising diffusion probabilistic models实现基础框架，MRI重建中通过宋飏的NCSNPP实现

- 文章中通过高斯噪声的性质分析了初始图像和GT之间的误差大小，然后通过the stochastic contraction property of the stochastic difference equation推导了最终和GT的生成之间的误差大小：求解逆问题的反向扩散过程中，通常是普通扩散+正问题约束两步交替进行，文章在这里对正问题约束做了一个non-expansive mapping的限制，投影法或是梯度下降法都可以在这个限制下满足，前向扩散会增加误差，增加的误差可以通过高斯噪声的性质来分析得到；之后反向去噪过程中根据stochastic contraction property误差会指数收缩，公式17不等式右侧的第一部分就是注入噪声在正问题约束步下的误差下限，因此就可以得到本文最关键的结论；对于一个任意误差的最终生成需求，都能找到一个最小的去噪步数满足，另外如果初始的guide图像和GT误差越小，这个最小去噪步数也越小，这就达成了推理加速的效果

- 整体来说，这篇文章建立在MIA2022工作的基础上，MIA2022通过扩散模型解决MRI的重建问题，但弊端是推理步数过多导致重建速度拖慢，因此有了这篇文章，通过一个较为粗糙的监督网络来得到一个预估初始值也不会对本身的逆问题求解系统带来过大的负担，为了比较符合CVPR的口味，作者相当于开辟了超分和去噪两个子任务，并且自然图像任务中一个卖点是两个子任务的非条件生成网络的训练是相同的，即扩散模型训练不受具体的任务影响
