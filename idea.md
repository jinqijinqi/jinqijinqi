# [Mechanical Systems and Signal Processing 1区 Top,较快，计算易][Classifier-guided neural blind deconvolution: A physics-informed denoising module for bearing fault diagnosis under noisy conditions](https://www.sciencedirect.com/science/article/pii/S0888327024006484)
# review paper
organize students to write review paper
1) 制定方案
   
     a) 确定多个小主题（围绕熟悉的大主题）

     b) 搜集文献

     c) 理解并绘制历史图及算法关系图和问题

     d) 可能的研究方向
   
# 研究思路

1）找到领域近两年的精典文献，仔细总结研究历史和发展，以前两年文献为基础，查找引用它的相关文献直到当前，理解其研究扩展思路得到启发。

# 深度学习与概率逻辑规则联合学习的通用人工智能

1） [DeepProbLog](https://github.com/ML-KULeuven/deepproblog) 

# 基于KAN+Sparse Bayesian的模型改造

1) (2024) [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756). [KAN-Tutorial(code)](https://github.com/pg2455/KAN-Tutorial)


# 基于styleGAN的图像逆问题

1）[Robust Unsupervised StyleGAN Image Restoration 2024 CVPR](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10205099)

# 指纹

1）[Synthesis and Reconstruction of Fingerprints using Generative Adversarial Networks 细节点控制](https://github.com/rafaelbou/fingerprint-generator/)

2）[Enhancing Fingerprint Image Synthesis with GANs, Diffusion Models, and Style Transfer Techniques](https://arxiv.org/pdf/2403.13916)

# 图像融合

1）cyclegan and stylegan 图像同时融合与增强: a) high quality image guided cyclegan based unsupervised image fusion and defusion: fused image and high quality image are inputted to a discriminator, high quality image could be other natural image or histogram equalization image of VIS and IR images; use cycle, adversarial and fidelity loss, the fusion and defusion in the same framework. the defusion generator can provide source images from any natural image, thus provide groundtruth for image fusion problem. 2) we can use binary attention based multiple discriminators for IR, VIS; learning the binary attention mask from fused image then discriminate the masked fused image from IR and VIS, 这样可以从融合图像中分离出红外与可见光主要信息并把它们与源图像进行区分，避免非主要信息干扰，我们可以检查MASK学习到红外的主要区域。也可以基于我们的分割IR图像代替源IR图像输入区分器，相应的VIS图像也可以这么做。 不同代数的融合模型的融合结果取平均可能有更好的结果，因为观察到不同代数模型的融合结果有较小的差别。

2) [Unpaired high-quality image-guided infrared and visible image fusion via generative adversarial network 2024](https://www.sciencedirect.com/science/article/pii/S0167839624000591?dgcid=rss

3) [CycleFusion: Automatic Annotation and Graph-to-Graph Transaction Based Cycle-Consistent Adversarial Network for Infrared and Visible Image Fusion 2024 IEEE TETCI](https://ieeexplore.ieee.org/document/10526484)

4) 红外可见光融合退化模型 [A Novel Teacher–Student Framework With Degradation Model for Infrared–Visible Image Fusion](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10522962)

5) 用现有分割方法检测原图清晰或具有重要信息的部分，用清晰部分指导神经网络融合图像，这就是部分监督训练方法并利用了神经网络的插值能力。[PR2025:MMAE: A universal image fusion method via mask attention mechanism(code)](https://www.sciencedirect.com/science/article/pii/S0031320324007921) ：该文章的二值MASK具有排它性，并且在学习阶段需要标注MASK.

6) 对于多聚焦融合，我们可以用分割网络推广我们基于模糊检测的融合方法

7) * **图像增强与融合(直方图均衡)**[LEFuse: Joint low-light enhancement and image fusion for nighttime infrared and visible images(2025 neurocomputing, code](https://www.sciencedirect.com/science/article/pii/S0925231225002644)
   * **虚拟低级图像处理任务辅助的多任务图像融合** [One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion (2025 CVPR, code](https://github.com/AWCXV/GIFNet)

8) 基于fast style transfer network 的图像融合： 1）输入图像红外与可见光，输出图像逼近我们分割强化后加权的初始融合图像；2）网络图像可以是我们初始融合后图像的直方图均衡图像，参考[https://ww2.mathworks.cn/help/deeplearning/ug/train-fast-style-transfer-network.html](https://ww2.mathworks.cn/help/deeplearning/ug/train-fast-style-transfer-network.html)

9) [EV-Fusion: A Novel Infrared and Low-Light Color Visible Image Fusion Network Integrating Unsupervised Visible Image Enhancement(IEEE Sensor journal 2024 codes](https://ieeexplore.ieee.org/document/10380532)

10) diffusion model for image fusion: with our psuedo groundth truth fusion image, we can build conditional diffusion model for IVIF problem [LFDT-Fusion: A latent feature-guided diffusion Transformer model for general image fusion (code 2025 IF)](https://www.sciencedirect.com/science/article/pii/S1566253524004172) [DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion (2023 ICCV code)](https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_DDFM_Denoising_Diffusion_Model_for_Multi-Modality_Image_Fusion_ICCV_2023_paper.html) [Dif-Fusion: Toward High Color Fidelity in Infrared and Visible Image Fusion With Diffusion Models (2023 TIP code)](https://ieeexplore.ieee.org/abstract/document/10286359)

11) AC-GAN + fake 类= more stability+ fast converging+ intra class diversity+ class consistent: 添加异常类的ACGAN是对ACGAN的扩充，基于此有无穷多应用：a) AC-GAN 模型改进；b) 数据扩充（类平衡与非平衡）；c）混合生成与分类模型训练；d) 开集识别与异常类检测；e) 图像融合：输入红外与可见光，分类红外、可见光、fake、truthorfalse四类；[Conditional GANs with Auxiliary Discriminative Classifier](https://proceedings.mlr.press/v162/hou22a.html)

12) 基于软或自适应直方图的图像融合
13) SuperResolution ESRGAN+ LEFuse model[LEFuse: Joint low-light enhancement and image fusion for nighttime infrared and visible images(2025 neurocomputing, code](https://www.sciencedirect.com/science/article/pii/S0925231225002644)

# 改进softmax

1）直接稀疏化softmax的输出，参考文献[Sparse-softmax: A Simpler and Faster Alternative Softmax Transformation 2022](https://arxiv.org/pdf/2112.12433), [Adaptive Sparse Softmax: An Effective and Efficient Softmax Variant for Text Classification 2023](https://openreview.net/forum?id=5cio7DSIXLQ), [MultiMax: Sparse and Multi-Modal Attention Learning 2024](https://arxiv.org/pdf/2406.01189), [Exploring Alternatives to Softmax Function 2020](https://paperswithcode.com/paper/exploring-alternatives-to-softmax-function),[通向概率分布之路：盘点Softmax及其替代品](https://spaces.ac.cn/archives/10145)


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
7) GAN or CGAN mode collapsing problem solution: 1) treat each sample as a class and train CGAN; 2) the discriminator from CGAN could be used as classifier by inputing the test sample with possbile class combination and find the most possible combination to obtain class label; need to compare with conventional trained classifier; 3) using cycle consitency loss to deal with mode collapse: two training cycles from A to B to A and from B to A to B
8) Reconstruction GAN from Both  Sides: from A to B to reconstruct A and from B to A to reconstruct B; A is a set of samples from standard normal distribution, B is a set of true data. similar to denoising diffusion model but much different. use this to improve VAE.
9) AAE改进： 添加一个区分器区分生成的样本，同时采样先验分布对解码器GAN进行训练，就可以联合VAE的避免模式崩塌能力和GAN的高质量图像生成能力.
10) GAN 改进： 从x=G(z),区分x与真实数据, 然后z1=F(x)，区分z1和标准正态
11) remark： 模式崩塌不可避免，它的好处是可以生成高质量样本，坏处是样本缺少多样性，但是质量与多样性应该是一种折中，AAE和GAN改进就是这种折中。
12) 我们可以对生成器生成的图像加上自然图像限制，从而约束生成模型
13) 利用噪声注入改善GAN，VAE, diffusion model: 参考styleGAN.
14) AC-GAN + fake 类= more stability+ fast converging+ intra class diversity+ class consistent: 添加异常类的ACGAN是对ACGAN的扩充，基于此有无穷多应用：a) AC-GAN 模型改进；b) 数据扩充（类平衡与非平衡）；c）混合生成与分类模型训练；d) 开集识别与异常类检测；e) 图像融合：输入红外与可见光，分类红外、可见光、fake、truthorfalse四类；[Conditional GANs with Auxiliary Discriminative Classifier](https://proceedings.mlr.press/v162/hou22a.html)

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

3) 针对非限制采样和FRI信号的压缩去卷积及盲去卷积及贝叶斯[code](http://alumni.media.mit.edu/~ayush/USF_RR_Code/TSPCode/Demo_Code.html)，[Deep Learning for Modulo Sampling of FRI Signals ICASSP2025](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10890151) [Noise-Resilient Unlimited Sampling and Recovery of Sparse Signals](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10888741) [Unrolled Compressed Blind-Deconvolution](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10132064) [Compressed Sensing Based Residual Recovery Algorithms and Hardware for Modulo Sampling](https://arxiv.org/abs/2412.12724)

4) [贝叶斯扩展, 使用FFT-ReLU和相位自相关的稀疏图像和源模糊图像深度学习卷积核和清淅图像或selfdeblur：Blind Image Deblurring with FFT-ReLU Sparsity Prior(code 2025 WACV)](https://github.com/Metalicana/Blind-Image-Deblurring-with-FFT-ReLU-Sparsity-Prior）[Phase-only-Image-Based-Kernel-Estimation-for-Blind-Motion-Deblurring]（https://github.com/panpanfei/Phase-only-Image-Based-Kernel-Estimation-for-Blind-Motion-Deblurring）[DeepPSF: On-Orbit Point Spread Function Estimation of Space Camera With Deep Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10971376)
6) [L0 范数多尺度梯度域稀疏逼近原图，从而找到强边缘，10%强边缘估计核]
7) selfdeblur 改进：1）网络结构可以是无下采样结构，并用我们的方法搜索最佳结构；2）网络性能提升：原来需要对每个图像进行训练，大的时间开销限制应用，现在输入模糊图、噪声图像、FRT（fourier relu transformation)变换后的图像进行训练，训练好的网络可直接用于去模糊，速度很快；3）对于生成模糊核的网络，输入模糊图、噪声和FRT（fourier relu transformation)变换后的图像；4）网络结构也可以是生成图像和模糊核的双分支网络，从而共享特征，在模糊核的分支可以多次输入FRT（fourier relu transformation)变换后的图像；[selfdeblur](https://ieeexplore.ieee.org/document/9156633/citations?tabFilter=papers#citations)
8) Bayesian version or other extension [2025MSSP-Classifier-guided neural blind deconvolution: A physics-informed denoising module for bearing fault diagnosis under noisy conditions](https://www.sciencedirect.com/science/article/pii/S0888327024006484)

# 神经网络训练

1）__多专家神经网络___：对于全体样本尽量增广训练简单的神经网络，根据ostu方法自动分割输出分数的熵图，将熵高的样本用于训练第二个神经网络，依次直到样本数较少为止，在测试时，根据训练过程依次用多个网络进行测试，取熵最小的结果。或者依次测试，如果熵不达要求（超过阀值），则测试下一个，如果所有都不达要求，则取熵最小的结果。

2）__基于相位和神经网络峰值检测的图像校准__： 利用基于相位的图像对齐方法生成仿真图像，然后训练神经网络进行峰值检测，达到图像校准的目的。

3）__基于图像对齐的神经网络测试__: 对于用增广样本训练好的网络，对图像进行分类检测，如果分类检测的分类熵较大，则根据可能的前几个分类模板平均图像对待测图像校准，之后再分类检测，得到结果

4）__小样本训练半监督__: 增广样本训练，然后对未标记样本进行分类，对未标记样本使用分数自适应加权其损失函数对原网络进行训练，这样确定性的样本贡献较大，如果训练准确度为100%，则继续增加样本，如果训练有误不收敛，
则对训练样本分类，移除错分的样本。

5）__基于变分混合高斯聚类指导的无监督神经网络学习__: 使用变分高斯提供训练样本，训练神经网络

# 基于深度学习的K-means

1) 建立分类映射模型 $p_i=f(x_i,\theta)$ , 其中 $p_i$ 是概率向量softmax；
2) 构建目标函数

    $$J(\theta)=\sum_{i,k}||x_i p_i^k-\sum_{j=1}^N x_j p_j^k||_2^2$$

   并优化得到 $p_i=f(x_i,\theta)$ 分类器

# 穿地雷达（GPRmax仿真groundtruth)
1) [seismic-deeplearning（代码，数据，分类，分割）](https://github.com/microsoft/seismic-deeplearning?tab=readme-ov-file#dataset-download-and-preparation)

2) [Sparse multichannel blind deconvolution of seismic data via spectral projected-gradient (public data)](https://arxiv.org/pdf/1611.03754v4)
 
3) [GPR Subsurface Diagnosis (GPR-SD) dataset](https://github.com/ubriustc/GPR_data/tree/main)

4) [TIGPR: A Multi-View Ground Penetrating Radar Detection Data for Damage Assessment of Transportation Infrastructure](https://data.mendeley.com/datasets/ckgvrft232/1)

5) [Intelligent recognition of subsurface utilities and voids: A Ground Penetrating Radar dataset for Deep Learning applications](https://data.mendeley.com/datasets/ww7fd9t325/1)

6) [Ground Penetrating Radar dataset with ground-truth data of utility surveying activities](https://data.4tu.nl/datasets/96303227-5886-41c9-8607-70fdd2cfe7c1)

7) [GPR-GAN](https://github.com/xiongGPR/GPR-GAN)




   
