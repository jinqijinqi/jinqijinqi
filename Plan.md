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
     
 ## fast journal
 1. [Journal of Neural Engineering( median 47 days, Q2](https://iopscience.iop.org/article/10.1088/1741-2552/aca2de)
