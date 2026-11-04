# SDEdit

- Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J. Y., & Ermon, S. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. In International Conference on Learning Representations.
  - 扩散模型 图像编辑的开山之作 ICLR 2022

- 从比较简单的角度来分析这篇文章：在这之前效果很好的生成图像编辑任务都通过基于GAN的模型来实现，包括CycleGAN和Pix2Pix，但在Score-SDE的大一统框架之下，SDEdit就提出了在Score-SDE生成模型下的图像编辑算法，SDEdit是免训练的，直接用预训练得到的非条件生成模型，在采样的过程中将参考图像喂入即可完成编辑任务

- 文章中将SDEdit定位为一个统一的图像合成和编辑框架，主要完成三类任务(文章中将输入的风格指导、编辑任务指导图像称为guide，high-level的guide任务类似stroke painting to image的任务，通过粗糙抽象的guide来指导生成；mid-level的guide指的是类似stroke-based editing的任务，通过在原始图像中的一部分经过stroke风格的提示来指导生成，low-level的guide就是image compositing，即直接在原始的图像中通过图像patch的拼接来指导图像生成)：
- ![SDEdit的图像编辑任务](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/46.png)
- 上述任务中，统一的模式都是通过RDB格式的guide图像来指导图像生成，无论其具体任务是什么

- 总体来看，文章的算法很简单，其实就是在反向SDE的基础上不断去噪，在原始的guide图像中加噪，再将加噪得到的图像作为反向SDE求解当中的其中一步，对其在反向SDE的过程中进行去噪即可
- 文章中的主要扩散模型实现方式是VE-SDE，因此有下面的推导：
- ![SDEdit正文部分的图像合成反向采样流程](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/47.png)
- 剩下的思路就很简单了，将guide图像作为均值，添加一个特定时间步数的方差，对其进行采样，即可得到一个在特定步数的正向加噪结果，再对其进行去噪就是SDEdit做的所有事情
- ![SDEdit的思路](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/48.png)
- 上述算法的实现途径很暴力，但也很简单，原因就是只需要一个采样+反向求解SDE即可，完全不需要额外的任何训练，但因为需要确定一个特定时间步数，因此t_0的这个选定时间，就成了文章的一个核心讨论点，即面对上述的一系列图像编辑任务，文章需要在两个评价标准上进行一个trade-off，即忠于原始的guide和生成的真实性，如果将t_0这个前向的加噪时间选的比较大，就能保证极高的生成真实性，但如果时间比较短，就能保证比较好的faithfulness

- 附录当中详细分析了文章的很多方面，我认为值得读的有以下内容：
  - 首先图像编辑任务没办法通过Score-SDE中的条件生成来做，首先条件生成在Score-SDE中的解决方式是训练一个加噪图像的分类器，但图像编辑很难实现的一点就是guide信息的各异，每个人可能想要调整的内容不同、编辑的方式不同，没办法设计一个比较完备的分类器，这就导致在计算条件score的时候很困难，这套范式就不是很适用(但文章中也做了一个使用条件生成来解决图像编辑的对比，应该是在特定的编辑情况下进行训练得到的结果)
  - 因为反向SDE求解基本上没有任何改变，所以不光是正文的VE-SDE可以实现编辑，也同样可以在VP-SDE中实现，这就使得DDPM的相关训练结果可以无缝的插入到SDEit的编辑体系之下
  - 在有一些特定的图像编辑任务当中，图像的生成更为细节，例如mid-level的编辑任务中，只在图像的特定部分进行编辑，这种生成效果更期望在不能编辑的图像区域最好一点编辑都不要，如果最简单的实现方式一定是直接用一个mask来实现，但一个更重要的点在于，score的训练方式是通过一整张加噪图像的方式输出一个噪声预测来实现的，如果只对图像中的一部分进行加噪，这时网络是看不懂的，所以文章中对于这类图像编辑任务的做法是同样在对应的时间步骤中向不可编辑的部分同样添加相同的噪声，按照x_t given x_0的方式改变这部分图像内容，具体的实现方式就需要通过VE SDE和VP-SDE的具体加噪形式来构建，以下是4个比较关键的伪代码，分别对应VE SDE和VP SDE的加mask版本和不加mask版本的SDEdit
  - ![SDEdit的伪代码](https://github.com/wanna-bornb-disciplinant/Generative-Models-Atlas/blob/main/images/diffusion/49.png)
