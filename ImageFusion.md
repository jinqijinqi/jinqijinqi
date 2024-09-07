# [paper](#paper-section-id)
# [dataset](#dataset-section-id)
# [benchmark](#benchmark-section-id)
# [idea](#idea-section-id)
-------------------------------------

<a name="paper-section-id" />

# paper
 ## survey
 ### 2023
  1.  X. Zhang and Y. Demiris, ["Visible and Infrared Image Fusion Using Deep Learning,"](https://ieeexplore.ieee.org/abstract/document/10088423) in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 8, pp. 10535-10554, Aug. 2023. [code(matlab)(see benchmark)](#benchmark-section-id)

<a name="dataset-section-id" />

# dataset

<a name="benchmark-section-id" />

# benchmark
 ## 2023
   1. [VIF benchmark (code(python))](https://github.com/Linfeng-Tang/VIF-Benchmark), 2023
 ## 2020
   1. X. Zhang, P. Ye and G. Xiao, ["VIFB: A Visible and Infrared Image Fusion Benchmark,"](https://ieeexplore.ieee.org/document/9150987) 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Seattle, WA, USA, 2020, pp. 468-478. [code (matlab)](https://github.com/xingchenzhang/VIFB)

      
<a name="idea-section-id" />

# idea
1. 本质上，图像融合是由几副图像生成一副保留互补信息的图像，其困难在于没有真实的目标图像存在（groundtruth 不存在或获取困难），因而比普通的具有ground truth的图像处理更具有挑战性。为此，通常是使生成的图像与两个源图像尽可能相似，显然这并不能保证融合图像具有自然性，也不能保证具有互补信息，甚至可能有虚假不自然的内容。为此，我们从构造伪目标图像出发，用它训练我们的函数， 但是伪目标图像与理想目标图像有差异，我们不能强制要求融合图像与伪目标图像高度一致，只能是大致一致，这正好可以通过训练GAN来达到目标，因为GAN不具有一一强制对应，只能是大致对应，这是由GAN的机制决定的；另一方面，为了保证大致一致，我们可以用特征损失（如要求梯度相似，结构相似，感知特征相似等）训练通常的CNN，而不要求像素强度一致。另外，融合过程很少考虑图像增强，我们希望将融合与增强结合，这需要无监督的增强方法；同时，将融合与下游可监督任务结合，使融合图像更适合下游任务；以及与网络结构搜索和下游监督任务结合，我们称为多任务融合。
2. 使用生成器生成融合图像，然后融合图像与高质量高对度进行区分，融合图像再通过另两个生成器生成输入红外与可见光图像，并分别进行区分，这样类似于cycleGAN我们就生成了高质量的整合图像，简化了损失函数的设计，同时达到既整合又增强的目的。
