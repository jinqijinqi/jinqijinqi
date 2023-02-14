# 19:00-21:00, 2022/11/3
----------------------
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
 
## Bilinear imaging/signal  model
1. compressive sensing + deconvolution: 
   ''' 
   $$y=\Phi h*x+n $$
   ''' 
   # tranditional non bayesian (energy method)
   
   # Bayesian method
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
       
       refer to [2. Variational Bayesian Blind Color Deconvolution of
Histopathological Images
 (code)](https://ivpl.northwestern.edu/wp-content/uploads/2021/06/Variational-Bayesian-Blind-Color-Deconvolution-of-Histopathological-Images-compressed_compressed_compressed.pdf))
      

## reinforcement learning for image enhancement
1. Underwater Image Enhancement With Reinforcement Learning (2022 with codes)
 
 
 
     
 ## fast journal
 1. [Journal of Neural Engineering( median 47 days, Q2)](https://iopscience.iop.org/article/10.1088/1741-2552/aca2de)
   
   example paper: [MRASleepNet: A multi-resolution attention network for sleep stage classification using single-channel EEG](https://iopscience.iop.org/article/10.1088/1741-2552/aca2de)
   
 2. [optics express, median 27 days, Q2](https://opg.optica.org/oe/journal/oe/about.cfm)
 
## Ai video surveillance platform (租云服务器作为主机，搭建基于WebRTC的监控平台）
  1. [vidgear: video processing platform (python)](https://github.com/abhiTronix/vidgear)
  2. [Private Home Surveillance with the WebRTC DataChannel (Ivelin Ivanov)](https://webrtchacks.com/private-home-surveillance-with-the-webrtc-datachannel/)
  3. [nginx-http-flv-module: media streaming server](https://github.com/winshining/nginx-http-flv-module)
  4. [使用WebRTC广播IP摄像头视频流](https://blog.csdn.net/biaobro/article/details/64129620?spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-64129620-blog-89626766.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-64129620-blog-89626766.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=15)
  5. [Get started with WebRTC](https://web.dev/webrtc-basics/)
  6. 5G 和Wifi6无线组网WebRTC监控平台：结合2中的方法，用wifi6 路由器连接本地多个摄像头，然后用5G网关连接互联网，用WEBRTC 通信，做到无布线。
  7. [red5 media server: Live video streaming solved. Broadcast video to millions in under 500 milliseconds](https://github.com/Red5/red5-server)

## AI model deployment (all student projects required to deploy here)
1. [streamlit: simple and free and can be deployed in local server (nginx inverse proxy for intranet)](https://streamlit.io/);
2. [webrtc and deployment example (classification and detection and media stream with webcam](https://github.com/whitphx/streamlit-webrtc-example)
   
  
## hardware seller
1. [waveshare electronics](https://www.waveshare.net/cattree.html)
