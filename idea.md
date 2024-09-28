# review paper
organize students to write review paper
1) 制定方案
   
     a) 确定多个小主题（围绕熟悉的大主题）

     b) 搜集文献

     c) 理解并绘制历史图及算法关系图和问题

     d) 可能的研究方向
   
# 研究思路

1）找到领域近两年的精典文献，仔细总结研究历史和发展，以前两年文献为基础，查找引用它的相关文献直到当前，理解其研究扩展思路得到启发。

# generative AI
1) 基于混合高斯的随机编码器的自编码器：
   
    a) 设计随机编码器，将训练数据映射到随机向量，理论上该编码器可以将相似的样本映射到邻近位置，该随机编码无需学习
   
    b) 解码器的训练与通常自编码器一样
   
    c) 样本生成：将随机编码器生成的所有随机编码看作是一个混合高斯分布，随机选取一个高斯组分，按照高斯进行随机采样，输入解码器生成新样本
   
   该方法可以克服GAN的模式崩跌问题，训练较好较稳定，也可以生成与高斯组分类似的样本，达到条件GAN的目的

2) 基于分类器的样本生成及可解释性

   a) 任选一个随机向量输入分类器，通过梯度微调输入样本使分类输出指定的类，从而生成各类样本，还可以通过调节分类概率控制各类样本的相似度；

   b) 观察统计样本的变化过程，可以解释分类器使用图像哪些区域和哪些特征做出的决策，有助于可解释性
   
3) 表示学习的本质是从几何体的高维观测信号（图像、语音、电磁波等）找到几何体的关键数学定义参数，而忽略某些不重要的信息，如颜色、亮度等。从数学定义参数到几何体的
   生成需要复杂的数学概念和定义，抽象地说需要复杂的映射，正好神经网络可以提供复杂映射。只要定义数学参数的维数和几何体呈现信号的维数，神经网络总可以在两者之间建立映射。
   然而几何体呈现信号的维数可以给定（如图像的大小，序列的长度等），但是几何体数学定义参数的个数就不知道了。我们可以利用稀疏限制的概念，就是定义一个相对较大的几何体参数维数，并要求
   参数向量满足稀疏条件，从而让神经网络通过学习的方式自动找到恰当数量的几何体参数，这既依赖于样本数量，也依赖于神经网络的模型复杂度。其实这是自编码器的思想
   
4) 基于GAN的变分自编码器隐空间标准正态学习：

    a) 在编码器部分附加一个区分器来区分标准正态样本和非标准正态样本，从而使编码器的输出符合标准正态；这与变分自编码器使用分布近似不同，我们考虑的是总体样本的分布
    
    b) 同时解码器不变，它重构输入样本；
5) 基于CGAN的数据扩增解决数据稀少、类别不平衡、非标注数据学习等
    a) 基于少量数据训练CGAN，并扩充数据，再训练，再扩充；循环直到满足需要；b)对非标注数据进行分类，选择分类值信度高的样本到相应的类别中，增加样本数据；c) 利用GAN也可以扩充非标注数据
6) 多标签混合监督非监督CGAN
    a) 对于真实条件样本，使用类别监督和真实与非真实非监督，即两个标签；b)对于非真实条件样本，使用类别和非真实两个标签训练

# 基于稀疏的个数自动选择

1）对于混合高斯模型，使用softmax作为组份比例，然后对softmax加稀疏限制，从而使组份比例中含零，达到自动选择个数的目的. IEEE SPL.

# 贝叶斯方法

1）基于变分贝叶斯生成模型的图像融合： 
  a) 建立深度概率生成模型生成两个模态图
  b) 定义依赖于模态图的变分分布
  c) 使用变分推理进行学习
2）基于变分贝叶斯生成模型的半监督学习

# 压缩盲去卷积

1）[Unrolled Compressed Blind-Deconvolution TSP2023](https://ieeexplore.ieee.org/document/10132064), [Learning-Based Reconstruction of FRI Signals TSP2023](https://ieeexplore.ieee.org/document/10169093#citations)[An Efficient Estimation Method for the Model Order of FRI Signal Based on Sub-Nyquist Sampling TIM2023)[https://ieeexplore.ieee.org/document/10267985), [Parameter Estimation of Hybrid LFM and MPSK Pulse Sequences Based on Sub-Nyquist Sampling TIM2024](https://ieeexplore.ieee.org/document/10648820) [VDIP-TGV: Blind Image Deconvolution via Variational Deep Image Prior Empowered by Total Generalized Variation 2023](https://arxiv.org/abs/2310.19477), [Blind Image Deconvolution Using Variational Deep Image Prior TPAMI 2023](https://ieeexplore.ieee.org/document/10146429/citations?tabFilter=papers#citations), [RPIR: A Semiblind Unsupervised Learning Image Restoration Method for Optical Synthetic Aperture Imaging Systems With Co-Phase Errors 2024](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10645065), [IMU-Assisted Accurate Blur Kernel Re-Estimation in Non-Uniform Camera Shake Deblurring TIP 2024](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10558778), 通过变分贝叶斯扩展

2) __用贝叶斯压缩盲去卷积扩展如下列表中的去卷积方法(产生很多论文）__ [A curated list of resources for Image and Video Deblurring (code)](https://github.com/subeeshvasu/Awesome-Deblurring)



   
