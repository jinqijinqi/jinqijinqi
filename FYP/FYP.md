# 2026-2027 (SICE)

# 三个本科毕业设计题目及任务内容

以下按照高校本科毕业设计任务书的常用要求编写，包括中文题目、英文题目、主要任务、预期成果、参考文献和开源代码。三个选题均以成熟的开源算法为基础，强调可实现、可测试、可验收，适合安排给三名本科生。

三个题目构成“指纹结构仿真—指纹图像真实化—指纹深度识别”的完整研究链条，但每个学生均可独立开展工作。

## 题目一：基于结构模型的指纹图像仿真生成方法研究与实现

英文题目： Research and Implementation of Fingerprint Image Simulation and Generation Based on Structural Models

难度：较低｜无需 GPU

### 1. 主要任务

本课题研究指纹图像的基本结构特征及其计算机仿真生成方法，利用传统结构模型生成具有不同纹型和采集特征的指纹图像，开发指纹仿真生成程序，并分析不同参数对生成结果的影响。

具体任务如下：

1\. 指纹结构及仿真方法学习。 了解指纹的脊线、谷线、方向场、细节点和基本纹型，学习基于结构模型的指纹图像生成原理。

2\. 指纹生成算法实现。 学习和使用 Anguli 开源软件，实现不同虚拟身份、不同纹型的指纹图像生成，并保存图像及对应参数。

3\. 采集条件仿真。 利用 Python 和 OpenCV 实现旋转、平移、模糊、噪声及局部遮挡等变换，模拟不同采集条件下的指纹图像。

4\. 仿真参数及生成质量分析。 研究旋转角度、平移量、模糊强度和噪声水平等参数对指纹纹线清晰度及图像质量的影响。

5\. 仿真程序开发与测试。 实现基本的参数设置、批量生成、图像显示和结果保存功能。

### 2. 预期成果

- 完成指纹图像仿真生成软件一套，能够生成不同虚拟身份的指纹图像。
- 实现不少于四种采集变化模拟，包括旋转、平移、模糊和噪声。
- 建立不少于 1,000 幅图像的合成指纹样例数据集，并保存生成参数。
- 完成不同参数下的生成结果展示及图像质量分析。
- 提交完整程序源码、运行说明、实验结果及本科毕业论文。

### 3. 参考文献

[1] Wyzykowski A, Segundo M P, Lemes R P. Multiresolution synthetic fingerprint generation[J]. IET Biometrics, 2022. DOI: 10.1049/bme2.12083.&#x20;

[论文链接](https://doi.org/10.1049/bme2.12083?utm_source=chatgpt.com)

[2] Cappelli R, Maio D, Maltoni D. Synthetic fingerprint-database generation[C]//Proceedings of the 16th International Conference on Pattern Recognition. 2002.&#x20;

[论文检索](https://scholar.google.com/scholar?q=Synthetic+fingerprint-database+generation+Cappelli+Maio+Maltoni\&utm_source=chatgpt.com)

[3] Bouzaglo R, Keller Y. Synthesis and reconstruction of fingerprints using generative adversarial networks[EB/OL]. 2022.&#x20;

[论文链接](https://arxiv.org/abs/2201.06164?utm_source=chatgpt.com)

### 4. 参考开源代码

| 开源项目         | 用途              | 链接                                                                            |
| ------------ | --------------- | ----------------------------------------------------------------------------- |
| Anguli       | 传统指纹结构生成，主要代码   | [GitHub](https://github.com/wfr/Anguli?utm_source=chatgpt.com)                |
| OpenCV       | 图像旋转、平移、模糊、噪声处理 | [GitHub](https://github.com/opencv/opencv?utm_source=chatgpt.com)             |
| scikit-image | 图像质量分析及处理       | [GitHub](https://github.com/scikit-image/scikit-image?utm_source=chatgpt.com) |

Anguli 的公开镜像已作部分 Qt5 和 OpenCV 兼容性修改，但仍属于较旧的 C++/Qt 工程。建议指导教师先确认能否编译；如果环境搭建困难，可以提供预生成的结构指纹，让学生重点完成图像变化模拟与评价程序。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 题目二：基于 CycleGAN 的合成指纹图像真实化方法研究与实现

英文题目： Research and Implementation of Synthetic Fingerprint Image Realism Enhancement Based on CycleGAN

难度：中等｜建议配备 NVIDIA GPU

### 1. 主要任务

本课题研究基于生成对抗网络的图像风格迁移方法，利用少量真实指纹图像作为目标域数据，将传统结构模型生成的合成指纹转换为具有真实采集风格的指纹图像，并评价风格迁移效果以及指纹结构的保持情况。

具体任务如下：

1\. 相关方法学习。 了解 GAN 和 CycleGAN 的基本原理，重点学习无配对图像域迁移方法，以及指纹纹线、细节点等结构信息的特点。

2\. 指纹数据集构建。 收集并整理传统结构仿真指纹及公开真实指纹图像，建立合成指纹域与真实指纹域两个数据集，完成灰度归一化、尺寸调整等预处理。

3\. CycleGAN 网络复现。 基于公开 PyTorch 代码搭建 CycleGAN 训练环境，完成合成指纹向真实指纹风格的转换。

4\. 真实化方法分析。 比较风格迁移前后图像的纹理、对比度、噪声和纹线结构差异，分析 CycleGAN 对指纹细节点和整体结构的影响。

5\. 实验系统实现。 实现指纹图像批量转换、生成结果对比、数据保存和基本质量评价功能。

### 2. 预期成果

- 完成基于 CycleGAN 的合成指纹真实化程序。
- 构建合成指纹和真实指纹两个训练域，完成数据预处理。
- 生成不少于 500 幅真实风格的合成指纹图像。
- 完成转换前后图像的视觉对比、灰度统计及结构保持分析。
- 与未进行风格迁移的原始合成指纹进行对照实验。
- 提交训练或推理代码、模型权重、结果图表、使用说明及本科毕业论文。

不强制要求修改 CycleGAN 网络结构，也不以超越现有论文的生成质量作为毕业验收条件。可将轻量级结构一致性损失作为选做内容。

### 3. 参考文献

[1] Zhu J Y, Park T, Isola P, et al. Unpaired image-to-image translation using cycle-consistent adversarial networks[C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 2223–2232.&#x20;

[论文链接](https://arxiv.org/abs/1703.10593?utm_source=chatgpt.com)

[2] Wyzykowski A, Segundo M P, Lemes R P. Multiresolution synthetic fingerprint generation[J]. IET Biometrics, 2022. DOI: 10.1049/bme2.12083.&#x20;

[论文链接](https://doi.org/10.1049/bme2.12083?utm_source=chatgpt.com)

[3] Park T, Efros A A, Zhang R, et al. Contrastive learning for unpaired image-to-image translation[C]//European Conference on Computer Vision. 2020.&#x20;

[论文链接](https://arxiv.org/abs/2007.15651?utm_source=chatgpt.com)

### 4. 参考开源代码

| 开源项目               | 用途                 | 链接                                                                                               |
| ------------------ | ------------------ | ------------------------------------------------------------------------------------------------ |
| CycleGAN / pix2pix | 主要训练与推理框架          | [官方 GitHub](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix?utm_source=chatgpt.com)      |
| MR-SF              | 已发表的合成指纹数据，可用于前期实验 | [项目页面](https://andrewyzy.github.io/MR-SF/?utm_source=chatgpt.com)                                |
| CUT                | 选做的域迁移对比方法         | [官方 GitHub](https://github.com/taesungp/contrastive-unpaired-translation?utm_source=chatgpt.com) |

CycleGAN 的官方 PyTorch 工程提供自定义数据集、训练和测试入口，能够作为该毕业设计的主要实现基础。需要注意，未经额外约束的 CycleGAN 不能保证所有细节点均保持不变。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

junjunzia/cyclegan · GitHub



## 题目三：基于合成数据增强的深度指纹识别方法研究与实现

英文题目： Research and Implementation of Deep Fingerprint Recognition Based on Synthetic Data Augmentation

难度：中等｜建议使用预训练网络

### 1. 主要任务

本课题研究基于深度学习的指纹特征提取与匹配识别方法，利用传统仿真指纹以及经过真实化处理的合成指纹扩充训练数据，比较不同训练数据条件下的识别性能，分析合成数据增强的有效性。

具体任务如下：

1\. 深度指纹识别方法学习。 掌握卷积神经网络、指纹特征提取、特征相似度计算及指纹验证的基本原理，了解合成数据增强在指纹识别中的应用。

2\. 指纹数据集整理。 收集公开真实指纹及合成指纹数据，完成图像尺寸统一、归一化和身份标签整理，并建立相互独立的训练集与测试集。

3\. 深度指纹识别程序实现。 基于已有 DeepPrint 实现或轻量级 CNN，提取指纹特征，采用余弦相似度或欧氏距离完成指纹匹配。

4\. 合成数据增强实验。 比较仅使用少量真实数据、使用传统合成数据辅助训练，以及使用真实化合成数据辅助训练等方案的识别性能。

5\. 识别结果评价。 计算 ROC、EER、TAR\@FAR 等评价指标，分析合成数据数量和类型对识别结果的影响，完成程序测试和实验总结。

### 2. 预期成果

- 完成基于深度特征的指纹匹配识别程序，实现两幅指纹的相似度计算及匹配判定。
- 建立包含真实指纹、传统合成指纹和真实化合成指纹的实验数据集。
- 完成至少三组训练或微调实验，比较不同数据增强策略的识别性能。
- 绘制 ROC 曲线，计算 EER、TAR\@FAR 等评价指标。
- 完成合成数据增强效果分析，说明其有效条件与局限性。
- 提交完整程序源码、模型文件、实验结果及本科毕业论文。

考虑本科生的工作量，可以采用预训练深度特征提取网络，并仅微调轻量级模块或较小的识别网络。不要求从头训练完整 DeepPrint。

此外，实验中必须保证真实训练与测试身份不重叠，不能直接将新生成的虚拟指纹赋予某个真实手指的身份标签。

### 3. 参考文献

[1] Rohwedder T, Osorio-Roig D, Rathgeb C, Busch C. Benchmarking fixed-length fingerprint representations across different embedding sizes and sensor types[C]//2023 International Conference of the Biometrics Special Interest Group (BIOSIG). IEEE, 2023.&#x20;

[论文链接](https://arxiv.org/abs/2307.08615?utm_source=chatgpt.com)

[2] Engelsma J J, Cao K, Jain A K. Learning a fixed-length fingerprint representation[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, 43(6): 1981–1997.&#x20;

[论文链接](https://doi.org/10.1109/TPAMI.2019.2961349?utm_source=chatgpt.com)

[3] Wyzykowski A B V, Jain A K. Synthetic latent fingerprint generator[EB/OL]. 2022.&#x20;

[论文链接](https://arxiv.org/abs/2208.13811?utm_source=chatgpt.com)

第三篇直接研究利用合成潜在指纹增强深度匹配训练，并报告了在 NIST SD27 上的识别改善，为本选题提供了直接的实验依据。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



### 4. 参考开源代码

| 开源项目                                | 用途                        | 链接                                                                                                    |
| ----------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------- |
| Fixed-Length Fingerprint Extractors | DeepPrint 特征提取、训练与评价，主要代码 | [GitHub](https://github.com/tim-rohwedder/fixed-length-fingerprint-extractors?utm_source=chatgpt.com) |
| PyTorch                             | CNN 训练、微调与实验              | [GitHub](https://github.com/pytorch/pytorch?utm_source=chatgpt.com)                                   |
| SourceAFIS                          | 传统指纹匹配对照，可选               | [GitHub](https://github.com/robertvazan/sourceafis-java?utm_source=chatgpt.com)                       |
| MR-SF 数据集                           | 提供可直接使用的合成指纹              | [数据集页面](https://andrewyzy.github.io/MR-SF/?utm_source=chatgpt.com)                                    |

Fixed-Length Fingerprint Extractors 项目支持模型训练、特征提取和匹配基准测试，还提供数据集配置与预训练模型的使用示例。因此，学生不必从零开发完整识别网络。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 四、三个题目的统一任务安排与验收建议

| 项目       | 题目一：结构仿真      | 题目二：真实化     | 题目三：识别           |
| -------- | ------------- | ----------- | ---------------- |
| 核心技术     | Anguli、OpenCV | CycleGAN    | DeepPrint、CNN    |
| 主要输入     | 仿真参数          | 合成与真实指纹     | 带身份标签的指纹         |
| 主要输出     | 合成指纹图像        | 真实化合成图像     | 匹配结果与性能指标        |
| 必须训练深度网络 | 否             | 是           | 不必从头训练           |
| 主要评价     | 图像质量、结构变化     | 风格迁移效果、结构保持 | ROC、EER、TAR\@FAR |
| 主要软件成果   | 指纹生成程序        | 指纹图像转换程序    | 指纹识别程序           |
| 建议时间     | 12—14 周       | 14—16 周     | 14—16 周          |

### 三个学生之间的协作与独立性

建议指导教师统一提供少量示例指纹及数据组织规范，同时约定各项目输出接口：

- 题目一输出合成指纹图像、虚拟身份编号和仿真参数。
- 题目二输入合成及真实图像，输出真实化合成指纹，并保留对应原始图像和身份编号。
- 题目三读取合成与真实数据，完成深度特征提取、匹配和实验评价。

同时，为保证每名学生独立完成，建议为题目二预先提供 MR-SF 合成数据，为题目三预先提供一批合成和真实化样例图像，而不是必须依赖其他学生按时交付。

### 最终推荐的正式题目汇总

| 编号 | 中文题目                        | 英文题目                                                                                                  |
| -- | --------------------------- | ----------------------------------------------------------------------------------------------------- |
| 1  | 基于结构模型的指纹图像仿真生成方法研究与实现      | Research and Implementation of Fingerprint Image Simulation and Generation Based on Structural Models |
| 2  | 基于CycleGAN的合成指纹图像真实化方法研究与实现 | Research and Implementation of Synthetic Fingerprint Image Realism Enhancement Based on CycleGAN      |
| 3  | 基于合成数据增强的深度指纹识别方法研究与实现      | Research and Implementation of Deep Fingerprint Recognition Based on Synthetic Data Augmentation      |

建议将这三份内容作为本科毕业设计任务书的技术基础。 三个选题的定位分别是传统算法实现、生成式图像处理和深度学习识别，工作量适中、层次清晰，并且可以共享研究成果，构成一个完整的本科科研训练系列。
