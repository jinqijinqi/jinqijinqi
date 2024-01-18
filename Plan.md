## 格院2023科研见习项目 （论文为主）
李小龙，蒲志聪,杨立兆，冯建刚，邹筱甫
1. 细胞核分割: nature methods: [Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation](https://www.nature.com/articles/s41592-022-01639-4)
2. 自动指纹识别: 1)指纹仿真及数据库开发；2）基于仿真的数据开发基于深度学习的指纹处理及识别方法; 3)[Synthesis and Reconstruction of Fingerprints using
Generative Adversarial Networks(代码，基于GAN的指纹合成及身份细节保持而外观变化的仿真](https://arxiv.org/pdf/2201.06164.pdf) 4）[Fingerprint dataset collection](https://github.com/robertvazan/fingerprint-datasets)
3. infrared and visible image fusion with image enhancement: retinex +GAN+ distance map+ transformer. [Physics driven deep Retinex fusion for adaptive infrared and visible image fusion](https://www.spiedigitallibrary.org/journals/optical-engineering/volume-62/issue-08/083101/Physics-driven-deep-Retinex-fusion-for-adaptive-infrared-and-visible/10.1117/1.OE.62.8.083101.full?SSO=1) [Image fusion transformer](https://github.com/Vibashan/Image-Fusion-Transformer)
   1) [基于深度学习的图像融合方法综述(数据集，代码）2023](http://www.cjig.cn/jig/ch/reader/create_pdf.aspx?file_no=20230102&st=alljournals); 2)[马佳义github](https://github.com/jiayi-ma)[马佳义lab](http://mvp.whu.edu.cn/jiayima/);[tang linfeng](https://github.com/Linfeng-Tang)
   3) [DIVFusion: Darkness-free infrared and visible image fusion(低光照对比度增强）](https://www.sciencedirect.com/science/article/abs/pii/S156625352200210X)
   4) [ReFusion: Learning Image Fusion from Reconstruction with Learnable Loss via Meta-Learning (无学习，损失函数学习）](https://arxiv.org/pdf/2312.07943.pdf)
4. reinforcement learning for image enhancement: refer to our paper [AUTOMATIC IMAGE CONTRAST ENHANCEMENT BASED ON REINFROCEMENT LEARNING](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10016571)
5. 2023(segmentation) refer to [seUNet-Trans: A Simple yet Effective UNet-Transformer Model for Medical Image Segmentation](https://arxiv.org/pdf/2310.09998.pdf)

（1. 能够用pytorch进行新方法的快速实现和验证；2. 对深度学习理论有所掌握，能够快速设计自己的网络；3. 具有较强的英文论文收集、阅读、理解能力；4. 具有较好的英文写作能力；5. 诚实守信，刻苦努力，听从合理按排，每天坚持到实验室；6. 必须坚持到毕业，短期的或者打算中途跑路的千万不要来；7. 实验室根据项目情况，必要时可能会按排做其它项目，根据贡献可能有一些资金补助。）

## Reinforcement learning for air combat
1. data from game: digital combat simulator world(DCS) or other game
2. tacview record data and visualization and analysis
3. experiment with reinforcement learning for air combat based on above platform

## deep learning for radar interference mitigation
1. [bi level l1 interference reduction for millimeter radar (data, code)](https://github.com/Z-H-XU/Bi-level-L1-InterferenceReduction)
   paper's objective function J(x)=1/2 ||y-s-A*x||_2+\lambda_1 *||s||_1+\lambda_2 *||x||_1 replaced by J(x)=1/2 ||y-W*d-A*x||_2+\lambda_1 *||d||_1+\lambda_2 *||x||_1, where s=W*d, W and A is wavelet and fourier transformation matrix. we hope better performance since d is sparser than s (quick paper using the github code and dataset)
   
## quick way to dive into research and education by curated lists of lists
1. Path to a free self-taught education in Computer Science! [link](https://github.com/ossu/computer-science)
2. curated lists of lists [link](https://github.com/cuuupid/awesome-lists)
3. radar [link](https://github.com/ZHOUYI1023/awesome-radar-perception)


## Streamline AI development
1. [Keeping Up with PyTorch Lightning and Hydra — 2nd Edition](https://towardsdatascience.com/keeping-up-with-pytorch-lightning-and-hydra-2nd-edition-34f88e9d5c90)
2. [hydra-torch](https://github.com/pytorch/hydra-torch/tree/main)
## 25 institute GAN for augment radar parameter
1. [Generative Adversarial Network for Radar Signal
Generation](https://arxiv.org/pdf/2008.03346.pdf)
2. text to image (Conditional GAN Idea borrowed)
3. [A comprehensive review on GANs for time-series signals](https://link.springer.com/article/10.1007/s00521-022-06888-0)
4. [Generate Synthetic Signals Using Conditional GAN](https://ww2.mathworks.cn/help/signal/ug/generate-synthetic-pump-signals-using-conditional-generative-adversarial-network.html)
5. [Data Augmentation techniques in time series domain: a survey
and taxonomy](https://link.springer.com/article/10.1007/s00521-023-08459-3)
6.[FMCW-MIMO-Radar-Simulation(matlab code)](https://github.com/ekurtgl/FMCW-MIMO-Radar-Simulation)

## AI model deployment and Lab Website (all student projects required to deploy here)
1. [webrtc and deployment example- live stream template (classification and detection and media stream with webcam](https://github.com/whitphx/streamlit-webrtc-example) （现场摄像头支持）
2. [Eagle Vision for stuent project template static image case -project organization]https://github.com/Joshmantova/Eagle-Vision (所有学生参考这个组织自己的项目，主要包含训练源码和streamlit 展示代码，训练数据要在readme中指明可下载的地方，可以是自己的网盘或其它地方）
3. [Deploy Streamlit using Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)

# General Training Idea: 
with data augment, learning rate is reduced slowly based on training loss rather than valid/test loss since initial test/valid loss could be much lower than training loss; due to contineous decreasing of learning rate, the model with only training data without validation data is good without overfitting.



# 2023
--------------------------------
  ## S M SOLAYMAN HOSSEN ROKIB 
     task: develop road crack detection system
   1. [DeepCrack: Learning Hierarchical Convolutional Features for Crack Detection, IEEE Transactions on Image Processing 2019 (code and dataset)](https://github.com/qinnzou/DeepCrack) 
     

# 19:00-21:00, 2022/11/3
----------------------
## 黄沿鑫，曹彤
BSG: vision transformer for segmentation+ level set for postprocessing the result 
 1. [A Cascaded Deep Convolutional Neural Networkfor Joint Segmentation and Genotype Prediction of Brainstem Gliomas (TBME 2018)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8375811)



  ## 王凌锋(presenter) 、黎昊程、黄沿鑫、
  达到或超越前三名
  
  ### reference paper: 
  
  1. [Action unit detection by exploiting spatial-temporal and label-wise attention with transformer(CVPR2022)](https://github.com/jinqijinqi/jinqijinqi/blob/main/Wang_Action_Unit_Detection_by_Exploiting_Spatial-Temporal_and_Label-Wise_Attention_With_CVPRW_2022_paper.pdf)
   
    more experiments: on different number of transformers, five fold cross validation to learn five models, ensemble prediction, 
    
    more skills: augmentation, early stopping and learning rate reduction, label smoothing postprocessing
  
  2. [A Multi-task Mean Teacher for Semi-supervised Facial Affective Behavior
Analysis(iccv2021)](https://github.com/jinqijinqi/jinqijinqi/blob/main/Wang_A_Multi-Task_Mean_Teacher_for_Semi-Supervised_Facial_Affective_Behavior_Analysis_ICCVW_2021_paper.pdf)
    
    more experiments:  five fold cross validation to learn five models, ensemble prediction, class balancing
    
    more skills: augmentation, early stopping and learning rate reduction, label smoothing postprocessing
    
  3. [AutoMER: Spatiotemporal Neural Architecture Search for Microexpression Recognition(TNNLS 2022)](https://ieeexplore.ieee.org/document/9411707)
   
    skill: extend our transformer in paper 1 to this microexpression recognition with/without NAS
    
    dataset: this paper provides 5 dataset for microexpression recognition

     
  
  ## Deboch Eyob Abera (presenter),陈生阳、曹彤、 董哲明、马博俊
  
  Do presentation (algorithm+experiments) on visible and infrared image fusion
  
  reference paper:
  
  1. (code and dataset benchmark) [VIFB: A Visible and Infrared Image Fusion Benchmark(CVPR2020)](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w6/Zhang_VIFB_A_Visible_and_Infrared_Image_Fusion_Benchmark_CVPRW_2020_paper.pdf)
  
  ## Mola Natnael Fanose(presenter)、徐烁翔(presenter)、许智信、王富臣、童顺
  
  show quantitative results on multiple datasets
  
  reference paper:
  
  1. [NAS-SCAM: Neural Architecture
Search-Based Spatial and Channel Joint
Attention Module for Nuclei Semantic
Segmentation and Classification(MICCAI2020)](https://github.com/jinqijinqi/jinqijinqi/blob/main/NAS-SCAM.pdf)
   
    more experiments: using center+boundary+contour+distance map, overlapping segmentation, 
    level set, five fold cross validation to learn five models, ensemble prediction, 
    
    more skills: augmentation, early stopping and learning rate reduction, label smoothing postprocessing
  
  2. [Integrating deep convolutional neural networks with marker-controlle d watershe d for overlapping nuclei segmentation 
  in histopathology images(Neurocomputing 2020)](https://github.com/jinqijinqi/jinqijinqi/blob/main/Integrating%20deep%20convolutional%20neural%20networks%20with%20marker-controlled%20watershed%20for%20overlapping%20nuclei%20segmentation%20in%20histopathology%20images.pdf))
    
    more experiments: using center+boundary+contour+distance map, level set, 
    five fold cross validation to learn five models, ensemble prediction, 
    
    more skills: augmentation, early stopping and learning rate reduction, label smoothing postprocessing
    
   3. 黎昊程: Calibrated Adversarial Refinement for Stochastic Semantic Segmentation (ICCV2021): https://github.com/EliasKassapis/CARSSS
   
    nuclei segmentation experiments+ watershed, level set for cluster splitting
    
   4. variational approach for segmentation, refer to [Mumford and Shah Model and its Applications to Image Segmentation and Image Restoration (https://www.math.ucla.edu/~lvese/PAPERS/Springer-Segm-Chapter.pdf)
   [Applied Calculus of Variations for Engineers](http://www.iust.ac.ir/files/fnst/ssadeghzadeh_52bb7/calculus_of_variations.pdf)[Calculus of variations in image processing](https://www.math.u-bordeaux.fr/~jaujol/PAPERS/variational08.pdf)
   [The Calculus of Variations](https://www-users.cse.umn.edu/~jwcalder/CalculusOfVariations.pdf)
     
     skill: 1)learning signed distance directly, convex shape based level set
           
            2) new distance regularization term: (s^2-1)^2, replace curve length constraint with gradient sparsity constraint, refer [Distance regularization energy terms in level set image segment model: A survey](https://www.sciencedirect.com/science/article/pii/S0925231222003071?ref=pdf_download&fr=RR-2&rr=772a94f3c9b8044a)
            
            3) 0 level set controled level set by constraint with regularization term \int_{init_boundary} (\phi-0)^2 d\phi, where init_boundary is 
               given by some DL method.
            
            4) gpu implementation of level set for paper ["An Efficient Algorithm for Level Set Method](https://apps.dtic.mil/sti/pdfs/ADA557314.pdf)
Preserving Distance Function
     
            5) extend our simultaneous compressive and deblur model 
  5. deep segmentation with uncertainty: 1) using [selfsecure network (SCN)](https://github.com/kaiwang960112/Self-Cure-Network) to suppress uncertainty from edge pixels for faster convergence and obtain edge uncertain pixels corresponding large uncertainty. 2)this uncertainty map can be used as the segmentation map in watershed.
  
  ## 刘春银(presenter)、谢秋, 田子稼,王晶(presenter)
  
  实现如下论文7中的方法，并展示实验结果
  
  reference paper:
  
  1. [A Robust Approach for Singular Point Extraction Based on Complex Polynomial
Model
](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W01/papers/Qi_A_Robust_Approach_2014_CVPR_paper.pdf)
   
    more experiments: using center+boundary+contour+distance map, focus loss and regression loss and BCE losss,
    class balancing by randomly mask out background pixels,
     five fold cross validation to learn five models, ensemble prediction, 
    
    more skills: augmentation, early stopping and learning rate reduction
  
  2. [Fingerprint Detaset](https://github.com/robertvazan/fingerprint-datasets)
     
     [biometric dataset](https://tsapps.nist.gov/BDbC/Search)
  
  3. [SourceAFIS](https://sourceafis.machinezoo.com/)
  
  4. [deep recognition](https://github.com/JinZhuXing/Fingerprint_TF) 
     
     [NIST Biometric image software: type classification, minutia detection, quality estimation, matcher]
     (https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis)
  
  5. [Singular points detection with semantic segmentation networks](https://arxiv.org/ftp/arxiv/papers/1911/1911.01106.pdf)
  
  6. [Anil Jain archive paper list](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=Jain%2C+A&terms-0-field=author&terms-1-operator=AND&terms-1-term=fingerprint&terms-1-field=title&classification-physics_archives=all&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first)
     [Learning a Fixed-Length Fingerprint Representation](https://arxiv.org/pdf/1909.09901.pdf)
  
  7. [synthetic fingerprint generator(open source)](https://dsl.cds.iisc.ac.in/projects/Anguli/)
  
     more experiments: multitask fingerprint network(MTFN) for normalization, segmentation, orientation, frequency, enhanced map,
     thin map, singular point, minutia, class, fixed length feature vector with simulated groundtruth. 
     five fold cross validation to learn five models, ensemble prediction, 
    
    more skills: augmentation, early stopping and learning rate reduction
  

  ## ***(presenter): Neural Architecture Search:
  
  1. [Neural Architecture Search as Sparse Supernet](https://arxiv.org/pdf/2007.16112v1.pdf)
  
     skill: l0 sparsity version
     
     skill: assign node architecture variable for each node a_n and operation architecture variable o_n for each operation for each node. a_n *o_n is the operation 
     architecture weight for node n and operation o. thus we only need to require element sparse about a_n and o_n rather than group sparse in above paper.
     
     skill: bayesian version of this paper by using method in the following paper 2 and 3.
     
     skill: extend this method for dense connected network and cross resolution (by simple down pooling) dense connected network (especially for unet or resunet)
     
  2. [Bayesian-Learning-of-Neural-Network-Architectures:]( https://github.com/antonFJohansson/Bayesian-Learning-of-Neural-Network-Architectures)
     
     skill: nonbayesian sparse version of this paper for width and depth learning as in paper 1 or DARTS method
     skill: extend this method for cell search
     skill: extend this method for dense connected network and cross resolution (by simple down pooling) dense connected network (especially for unet or resunet)
     
  4. [Bayesian Compression for Deep Learning](https://github.com/KarenUllrich/Tutorial_BayesianCompressionForDL)
  
     skill: extend this method for width and depth learning as in paper 2
     skill: extend this method for cell search as in paper 1
     skill: extend this method for dense connected network and cross resolution (by simple down pooling) dense connected network (especially for unet or resunet)
     
 ## compressive sensing
 
 1. [dataset (Microwave SAR, MRI):](https://github.com/RosaZheng?tab=repositories )
 
 2. Patent: low cost portable single pixel camera with passive illuminance and lensless, refer to paper [A Low-Cost and Portable Single-Pixel Camera](https://ieeexplore.ieee.org/abstract/document/9723132)
     
     1). system: DMD+photodetector
     2). DMD: random reflected pattern
     3) lensless
     4) passive illuminance
     
 3. compressive sensing+extreme learning machine for efficient data acquisition and inference, refer paper [Deep learning for compressive sensing: a ubiquitous systems
perspective](https://link.springer.com/article/10.1007/s10462-022-10259-5)
 
 4. extreme of extreme learning machine, refer to paper [Extreme ensemble of extreme learning machines](https://www.researchgate.net/publication/347611805_Extreme_ensemble_of_extreme_learning_machines)
 
 5. new data fusion method for information fusion based on the data from [Single-pixel based Data Fusion algorithm for spectral-temporal-spatial reconstruction](https://github.com/jinqijinqi/SinglePixelDataFusion4D)
 
## Bilinear (inverse) imaging/signal  model
   ''' 
   $$y=\Phi_1 \Phi_2 *x+n $$
   ''' 
1. compressive sensing + deconvolution，superresolution+deconvolution, imageinpainting+deconvolution, compressive sening+imageinpainting, matrix factorization, PCA, Low rank matrix factorization,SVD etc.
   ''' 
   $$y=\Phi h*x+n $$
   ''' 
   ### i) tranditional non bayesian (energy method)
    1) [Deconvolution using natural image priors (code for spatial and frequency domain)](Deconvolution using natural image priors)
    2) unrolled method with neural network: 1) [CDLNet: Noise-Adaptive Convolutional Dictionary Learning Network for Blind Denoising and Demosaicing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9769957); 2)[Gabor is Enough: Interpretable Deep Denoising
with a Gabor Synthesis Dictionary Prior](https://arxiv.org/pdf/2204.11146.pdf)
    3) [Blind Image Deconvolution Using Variational Deep Image Prior](https://arxiv.org/abs/2202.00179) [VDIP-TGV: Blind Image Deconvolution via
Variational Deep Image Prior Empowered by Total
Generalized Variation](https://arxiv.org/pdf/2310.19477.pdf)
   
   ### ii) Bayesian method
   1) h given, using bayesian method (sparse Bayes(type II maximum likelyhood), variational bayes, EM method, time and frequency domain, unrolled method)
   2) h not given, using bayesian blind deconvolution (EM method, variational method, constant h or random h, time and frequency domain, unrolled method):
     
     a) convolution matrix H is represented by linear model 
      '''
      $$H=\sum_{m=1}^M h_m S_m $$
      '''
      refer to [1. Unrolled Variational Bayesian Algorithm for Image
Blind Deconvolution (code)](https://github.com/yunshihuang/unfoldedVBA)

      b) to using existing blind bayesian deconvolution method, we propose a new model which separate unkown h and x from compressive sensing:
      '''
      $$y=\Phi z+n, z=h*x $$
      '''
      refer to [2. Variational Bayesian Blind Color Deconvolution of Histopathological Images(code)](https://github.com/vipgugr/BCDSAR)
      
      [3.A VARIATIONAL APPROACH FOR BAYESIAN BLIND IMAGE DECONVOLUTION( frequency implementation and 3 methods](http://www.ssp.ece.upatras.gr/galatsanos/PAPERS/IPL_papers/T-SP04.pdf)
      
      [4.FREQUENCY DOMAIN BLIND DECONVOLUTION IN MULTIFRAME IMAGING USING ANISOTROPIC SPATIALLY-ADAPTIVE DENOISING( frequency implementation)](https://webpages.tuni.fi/lasip/papers/EUSIPCO2006-FrequencyDomainBlindDeconvolutionInMultiframeImaging.pdf)
      
      [5.VBLab (matlab package for Varational Bayesian](https://vbayeslab.github.io/VBLabDocs/)
      
      [6. pyro: probability and deep learning](http://pyro.ai/)
      
      [7. numerical tour: code template](https://www.numerical-tours.com/matlab/#inverse)
      
      [8. Image identification and restoration based on the expectation-maximization algorithm: frequency domain, EM, type II](https://ivpl.northwestern.edu/wp-content/uploads/2019/02/1990_optical.pdf)
      
      [9. Revisiting Bayesian Blind Deconvolution: EM, type II](https://jmlr.org/papers/volume15/wipf14a/wipf14a.pdf)
      
      [10. Maximum Likelihood Blur Identification and Image Restoration Using the EM Algorithm: frequency domain](https://www.researchgate.net/publication/3314072_Maximum_Likelihood_Blur_Identification_and_Image_Restoration_Using_the_EM_Algorithm#fullTextFileContent)
      
      [11. Bayesian Sparse Blind Deconvolution Using MCMC
Methods Based on Normal-Inverse-Gamma Prior: sparse deconvolution, substitute sampling method to general VB](https://arxiv.org/pdf/2108.12398v1.pdf)
   
      [12. Blind Deconvolution Using a Variational Approach
to Parameter, Image, and Blur Estimation](https://ivpl.northwestern.edu/wp-content/uploads/2019/02/BDC-2006.pdf)

      [13. Covariance-Free Sparse Bayesian Learning: covariance free faster implementation](https://arxiv.org/pdf/2105.10439.pdf)
      
      [14. Variational Bayesian Blind Image Deconvolution: A review](https://www.sciencedirect.com/science/article/pii/S105120041500144X?fr=RR-2&ref=pdf_download&rr=799310902a8396cf)
      
      [15. Sparse Bayesian blind image deconvolution with parameter estimation](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/1687-5281-2012-20)
      
      [16. Bayesian Blind Deconvolution with
General Sparse Image Priors: code ](http://www.dbabacan.info/software.html)

      [17. Fast Bayesian blind deconvolution with Huber Super Gaussian priors: frequency domain](https://www.sciencedirect.com/science/article/pii/S1051200416301245?via%3Dihub)
      
      18 Variational Bayesian software: 
       1) [babacan](http://www.dbabacan.info/software.html)
       2) [Molina](https://ccia.ugr.es/vip/software.html)
       3) [VBLab (matlab package for Varational Bayesian)](https://vbayeslab.github.io/VBLabDocs/)
     
      19 extend hyperparameter estimation by MAP or TypeII for spatially independent hyperparameter(signal sparse or noise sparse): for example
        ''' 
        $$y=\Phi_1 \Phi_2 *x+n, n \approx N(0,Diag(\sigma)), x \approx  N(0, G^T *Diag(\delta)G)$$  
        ''' 
        estimate noise and signal parameter $\sigma,\delta$ vectors
        
   [Maximum likelihood estimation of regularisation parameters Part II Theoretical Analysis](https://arxiv.org/pdf/2008.05793.pdf);     
   [Maximum likelihood estimation of regularisation parameters Part I Methodology and Experiments](https://arxiv.org/pdf/1911.11709.pdf);
[Maximum likelihood estimation of regularisation parameters in
high-dimensional inverse problems: an empirical Bayesian
approach](https://arxiv.org/pdf/1911.11709v1.pdf)
       20. Bayesian Classifier with class label treated as real number (class index is treated as normal distribution): we extend the following active bayesian classifier to multiple class clasifier with high dimention normal distribtution and for other application (such as medical image, detection, segmentation, classification), we use deep learning (or multiscale gabor filters) for feature extraction and logistic regression for active learning, please refer to ["Bayesian Active Remote Sensing Image
Classification"](https://ccia.ugr.es/vip/resources/BAL.html)

      21. a fast simple object instance detection method with boundary and center marker and background learning and marker based watershed and object verify: 1) use semantic segmentation to obtain foreground marker and background and boundary map; 2) thinning background to obtain background marker; 3) use boundary score map as segmentation map with local minimum constraint in foreground marker and background marker position; use marker based watershed to obtain instance objects; 4) for each instance object, using network to verify it is true or false object.  
      
      

## reinforcement learning for image enhancement
1. Underwater Image Enhancement With Reinforcement Learning (2022 with codes)
 
## image transform + deep learning
1. gabor filters + 1by1 filters: 通道用Gabor filter, 混合用1by1 to reduce #parameters, with random augmentation, we found it is better than [gabor convolutional netowrk (TIP2018)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8375811)
2. wavelet transform for encoder + deep learning for decoder: 用小波编码图像，用深度学习解码，类似于Unet.(多尺度图像分析）
3. we also find that conventinal cnn with 1by1 channel mixing filters could be better than original CNN. 
 
     
 ## fast journal
 1. [Journal of Neural Engineering( median 47 days, Q2)](https://iopscience.iop.org/article/10.1088/1741-2552/aca2de)
   
   example paper: [MRASleepNet: A multi-resolution attention network for sleep stage classification using single-channel EEG](https://iopscience.iop.org/article/10.1088/1741-2552/aca2de)
   
 2. [optics express, median 27 days, Q2](https://opg.optica.org/oe/journal/oe/about.cfm)
 
## Ai video surveillance platform (租云服务器作为主机，搭建基于WebRTC的监控平台）
  1. [vidgear: video processing platform (python)](https://github.com/abhiTronix/vidgear)
  2. [VPF (GPU codec, 500 fps) python](https://github.com/NVIDIA/VideoProcessingFramework) 
  3. [Top 5: Best Open Source WebRTC Media Server Projects](https://ourcodeworld.com/articles/read/1212/top-5-best-open-source-webrtc-media-server-projects)
  4. [Private Home Surveillance with the WebRTC DataChannel (Ivelin Ivanov)](https://webrtchacks.com/private-home-surveillance-with-the-webrtc-datachannel/)
  5. [nginx-http-flv-module: media streaming server](https://github.com/winshining/nginx-http-flv-module)
  6. [使用WebRTC广播IP摄像头视频流](https://blog.csdn.net/biaobro/article/details/64129620?spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-64129620-blog-89626766.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-64129620-blog-89626766.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=15)
  7. [Get started with WebRTC](https://web.dev/webrtc-basics/)
  8. 5G 和Wifi6无线组网WebRTC监控平台：结合2中的方法，用wifi6 路由器连接本地多个摄像头，然后用5G网关连接互联网，用WEBRTC 通信，做到无布线。
  9. [red5 media server: Live video streaming solved. Broadcast video to millions in under 500 milliseconds](https://github.com/Red5/red5-server)
  10. [python aiortc for webrtc and ortc](https://github.com/aiortc/aiortc) 

## AI model deployment and Lab Website (all student projects required to deploy here)
1. [streamlit: simple and free and can be deployed in local server (nginx inverse proxy for intranet)](https://streamlit.io/);
2. [webrtc and deployment example- live stream template (classification and detection and media stream with webcam](https://github.com/whitphx/streamlit-webrtc-example) （现场摄像头支持）
3. [Eagle Vision for stuent project template static image case -project organization]https://github.com/Joshmantova/Eagle-Vision (所有学生参考这个组织自己的项目，主要包含训练源码和streamlit 展示代码，训练数据要在readme中指明可下载的地方，可以是自己的网盘或其它地方）
4. Lab website in streamlit cloud or rent cloud server
5. [github Lab website template](https://zhuanlan.zhihu.com/p/370549865)
   
  
## hardware seller
1. [waveshare electronics](https://www.waveshare.net/cattree.html)

## software reference
1. Variational Bayesian method: 1) [babacan] (http://www.dbabacan.info/software.html); 2) [Molina](https://ccia.ugr.es/vip/software.html)
   3)[VBLab (matlab package for Varational Bayesian](https://vbayeslab.github.io/VBLabDocs/)
      
2. [pyro: probability and deep learning](http://pyro.ai/)
      
3. [numerical tour](https://www.numerical-tours.com/matlab/#inverse)

## matrix factorization and its application to noise remove, background and foreground separation
1. low rank matrix factorization
2. 

## ground penetrating radar
1. [basic principle](https://gpg.geosci.xyz/content/GPR/index.html)
2. matrix factorization for high-way crack picture noise remove and enhancement

## patent
1. 小型轻量化无人驾驶路面金属自动收集车设计
  参考日本专利：[1. Metal piece removal apparatus for cleaning vehicle that removes metal pieces scattered on the road](http://www-webofscience-com-s.vpn.uestc.edu.cn:8118/wos/alldb/summary/8d6c3f0e-07fc-4777-b85f-7217903ed9b0-7161dab1/relevance/1) [2. Vehicle for collecting metal pieces scattered on road, floor surface has catch flap, provided in metal piece collector, which prevents metal pieces from dropping once these pieces are adsorbed by magnet of metal piece collector](http://www-webofscience-com-s.vpn.uestc.edu.cn:8118/wos/alldb/summary/8d6c3f0e-07fc-4777-b85f-7217903ed9b0-7161dab1/relevance/1)

## NSFC Proposal
1. key techiques in facial affective behaviour analysis: transformer based task wise and label wise correlation learning + semi-supervised leaarning+efficient network architecture learning


## TI radar tutorial (best than book)
1. [Intro to mmWave Sensing : FMCW Radars ](https://www.ti.com/sitesearch/en-us/docs/universalsearch.tsp?langPref=en-US&searchTerm=introduction%20to%20mmwave%20sensing%20fmcw%20radars&preFilter=products_Sensors&nr=6#q=introduction%20to%20mmwave%20sensing%20fmcw%20radars&sort=relevancy&numberOfResults=25&f:products=[Sensors,mmWave%20radar%20sensors]&f:videos=[Video,Video%20series])
2. [SAR成像原理和仿真实现](https://blog.csdn.net/a1367666195/article/details/112073456)
## Benchmark State of the art method (best methods to borrow)
1. [paper with code](https://paperswithcode.com/)
2. [benchmarks.AI](https://benchmarks.ai/)

## dataset port
[Mendeley data:](https://data.mendeley.com/) 

[Medical Radar Signal Dataset for Non-Contact Respiration and Heart Rate Measurement](https://www.sciencedirect.com/science/article/pii/S2352340921009999?ref=pdf_download&fr=RR-2&rr=7c40d72abe390443#section-cited-by) : 1. Heart and respiration rate computation based on radar doppler frequency (shortage： heart rate and respiration rate could be chenged intentially which leads to failure of frequency based method) ; 2. ECG subwave detection (ECG analysis) based on radar signal; respiration analysis based on radar signal can solve above intentially changed HR/RR rate; 3. generate respiration or ECG curves by learning.  

## collaborator
[S. Derin Babacan](http://www.dbabacan.info/software.html)
[Rafael Molina](https://ccia.ugr.es/vip/software.html)

## GAN for different imaging modality 
1. GAN transfer different style image (such as different imaging modality) into one template style for image analysis to deal with image diversity problem

## Sparsity for boosting radar distance resolution for wide pulse
1. repalce matching filtering process with sparse representation learning for better distance resolution

## nature method
 1. 细胞核分割: nature methods: [Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation](https://www.nature.com/articles/s41592-022-01639-4) 

   omnipose limitation: distance definition failed in overlapping or occlusion cases (one pixel belongs to more than two objects).
   
    solution: 1)after omnipose process, overlapping area (overlapping area detection) is processed by 
   [segment anything model](https://github.com/facebookresearch/segment-anything), 
    [Instance Segmentation of Dense and Overlapping Objects via Layering](https://arxiv.org/pdf/2210.03551.pdf), 
    [Occlusion-Aware Instance Segmentation via BiLayer Network Architectures](http://aixpaper.com/view/occlusionaware_instance_segmentation_via_bilayer_network_architectures).

    dense nuclei/cell segmentation based on four color map: 1) coloring original image with manual mask and train generative network to predict four color map; 
    
    dense nuclei/cell segmentation based on edge enhanced image: 1)using mask to generate edge enhanced original image;2)using Generative network to predict edge enhanced 
    original image; 3) then segment the image after edge enhancement; improve any segment methods;

 3. cell segmentation and track: Maška, M., Ulman, V., Delgado-Rodriguez, P. et al. [The Cell Tracking Challenge: 10 years of objective benchmarking. Nat Methods (2023).](https://doi.org/10.1038/s41592-023-01879-y)
 4. Griebel, M., Segebarth, D., Stein, N. et al. [Deep learning-enabled segmentation of ambiguous bioimages with deepflash2. Nat Commun 14, 1679 (2023)](https://doi.org/10.1038/s41467-023-36960-9)
 5. Sheridan, A., Nguyen, T.M., Deb, D. et al. [Local shape descriptors for neuron segmentation. Nat Methods 20, 295–303 (2023).](https://doi.org/10.1038/s41592-022-01711-z)
 6. Primakov, S.P., Ibrahim, A., van Timmeren, J.E. et al. [Automated detection and segmentation of non-small cell lung cancer computed tomography images. Nat Commun 13, 3423 (2022).](https://doi.org/10.1038/s41467-022-30841-3)
 7. Shrestha, P., Kuang, N. & Yu, J. [Efficient end-to-end learning for cell segmentation with machine generated weak annotations. Commun Biol 6, 232 (2023).](https://doi.org/10.1038/s42003-023-04608-5)
 8. De Falco, A., Caruso, F., Su, XD. et al. [A variational algorithm to detect the clonal copy number substructure of tumors from scRNA-seq data. Nat Commun 14, 1074 (2023).](https://doi.org/10.1038/s41467-023-36790-9)
 9. de Teresa-Trueba, I., Goetz, S.K., Mattausch, A. et al. [Convolutional networks for supervised mining of molecular patterns within cellular context. Nat Methods 20, 284–294 (2023).](https://doi.org/10.1038/s41592-022-01746-2)
 10. idea: increamental learing : Pachitariu, M., Stringer, [C. Cellpose 2.0: how to train your own model. Nat Methods 19, 1634–1641 (2022).](https://doi.org/10.1038/s41592-022-01663-4)
 11. Dayao, M.T., Brusko, M., Wasserfall, C. et al. [Membrane marker selection for segmenting single cell spatial proteomics data. Nat Commun 13, 1999 (2022).](https://doi.org/10.1038/s41467-022-29667-w)
 12. 

## cooperate with hospital with paper framework
1. ["IRENE: A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics"](https://www.nature.com/articles/s41551-023-01045-x), Nature BME (2023). [code link](https://github.com/RL4M/IRENE)
2. [Foundation models for generalist medical artificial intelligence. Nature 616, 259–265 (2023)](https://www.nature.com/articles/s41586-023-05881-4)
3. [A framework for artificial intelligence in cancer research and precision oncology. npj Precis. Onc. 7, 43 (2023)](https://www.nature.com/articles/s41698-023-00383-y)
4. [Health system-scale language models are all-purpose prediction engines. Nature (2023)](https://www.nature.com/articles/s41586-023-06160-y)
5. [Learning consistent subcellular landmarks to quantify changes in multiplexed protein maps. Nat Methods (2023)](https://doi.org/10.1038/s41592-023-01894-z)
