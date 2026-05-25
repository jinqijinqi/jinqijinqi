---------2026-2027-----------------------------
可以。可以把 10 个题目严格围绕 ClassBD 的两个核心滤波部分来设计：

```text
A. 时域滤波部分：改 ClassBD 的 quadratic convolutional neural filter
B. 频域滤波部分：改 ClassBD 的 frequency linear neural filter
```

ClassBD 官方代码说明中明确指出，它的 neural BD 模块由两个级联部分组成：**时域二次卷积神经滤波器**和**频域线性神经滤波器**；时域部分用于增强周期冲击提取能力，频域部分用于频域滤波并提升 BD 性能。([GitHub][1])

下面给出 **10 个本科生可做、但具备小论文潜力** 的题目。

---

# 一、时域滤波部分改进：5 个题目

---

## 1. 基于门控二阶特征交互的 ClassBD 时域滤波改进方法研究

### Prerequisite Skills

Python、PyTorch、1D 卷积、张量逐点相乘、Sigmoid 门控、ClassBD 基础代码阅读。

### Project Description

**1. Background**
原始 ClassBD 的时域滤波部分采用 quadratic convolutional neural filter，本质上利用二次卷积增强周期冲击。但如果直接对输入信号构造二次项，可能会同时放大故障冲击和随机噪声。

**2. Main Task**
将原始输入级二次卷积：

[
z=W_1*x+W_2*x^2
]

改为卷积特征级二阶交互：

[
u=W_1*x
]

[
v=W_2*x
]

[
g=\sigma(W_g*x)
]

[
z=u+g\odot(u\odot v)
]

即先提取两个卷积特征，再进行门控相乘。

**3. Expected Outcome**
与原始 ClassBD 对比，在强噪声条件下：

```text
ACC / F1 提升 2%—5%
随机噪声尖峰误增强现象减少
输出信号的周期冲击更明显
```

### Reference Paper and Code

* ClassBD 原论文与官方代码：([arXiv][2])
* MNNBD 可作为神经盲去卷积结构参考：([GitHub][3])

---

## 2. 基于自适应高阶时域卷积的强噪声轴承故障诊断方法研究

### Prerequisite Skills

PyTorch、自定义网络层、归一化、高阶非线性、训练稳定性分析。

### Project Description

**1. Background**
ClassBD 使用二次卷积增强冲击，但轴承故障信号中可能存在更强的非高斯、高阶冲击结构。因此可以研究三阶或高阶项是否有助于增强弱故障特征。

**2. Main Task**
构造自适应高阶时域滤波模块：

[
z=W_1*x+\alpha_2 W_2\phi_2(x)+\alpha_3 W_3\phi_3(x)
]

其中：

[
\phi_2(x)=\frac{x^2}{mean(|x|^2)+\epsilon}
]

[
\phi_3(x)=\frac{x^3}{mean(|x|^3)+\epsilon}
]

(\alpha_2,\alpha_3) 为可学习权重。

**3. Expected Outcome**

```text
比较 Original ClassBD、二阶 ClassBD、三阶 ClassBD、自适应高阶 ClassBD
观察 α2、α3 的变化
验证三阶分支是否在强噪声下有效
```

目标是比原始 ClassBD 在 −4 dB / −6 dB 下提升 2%—4%。

### Reference Paper and Code

* ClassBD 官方代码：([GitHub][1])
* MNNBD 的多目标 neural BD 结构可作参考：([GitHub][3])

---

## 3. 基于多尺度时域盲去卷积的 ClassBD 改进方法研究

### Prerequisite Skills

PyTorch、多尺度 1D 卷积、卷积核大小设置、特征拼接、消融实验。

### Project Description

**1. Background**
轴承故障冲击包含不同时间尺度：短尺度尖峰、中尺度共振、长尺度周期结构。原始 ClassBD 的时域滤波器如果尺度单一，可能难以同时捕捉这些信息。

**2. Main Task**
将 ClassBD 的时域滤波改成多尺度结构：

[
u_3=Conv_{3}(x)
]

[
u_5=Conv_{5}(x)
]

[
u_9=Conv_{9}(x)
]

然后融合：

[
z=Fuse(u_3,u_5,u_9)
]

也可以设计：

[
z=u_3+g\odot(u_3\odot u_9)
]

其中小核捕捉尖锐冲击，大核捕捉周期趋势。

**3. Expected Outcome**

```text
小核 / 中核 / 大核 / 多尺度融合对比
观察不同核尺度对强噪声诊断的影响
输出信号中周期冲击更稳定
```

目标是在强噪声条件下比原始 ClassBD 提升 2%—5%。

### Reference Paper and Code

* ClassBD 官方代码：([GitHub][1])
* MNNBD 代码中包含 neural blind deconvolution 的 PyTorch 实现思路：([GitHub][3])

---

## 4. 基于周期感知时域卷积核的 ClassBD 改进方法研究

### Prerequisite Skills

轴承故障特征频率、采样率与故障周期计算、1D 卷积、dilated convolution、PyTorch。

### Project Description

**1. Background**
轴承故障信号的关键不是单个冲击，而是周期性冲击。ClassBD 的时域滤波增强冲击，但没有显式让卷积核感知故障周期。

**2. Main Task**
根据故障频率计算故障周期：

[
T_f = \frac{f_s}{f_{fault}}
]

然后设计周期感知卷积结构：

```text
普通卷积：捕捉局部冲击
空洞卷积：扩大感受野，捕捉周期间隔
周期约束：鼓励输出信号在 Tf 附近重复出现冲击
```

可加入周期自相关损失：

[
L_{period}=-R_e(T_f)
]

**3. Expected Outcome**

```text
输出信号包络自相关中故障周期峰值增强
包络谱中故障频率及倍频更明显
强噪声下分类准确率提升
```

### Reference Paper and Code

* ClassBD 官方代码：([GitHub][1])
* BD-RCC 代码可作为周期冲击和循环平稳思想参考；该仓库说明 BD-RCC 可用于恢复噪声振动中的重复冲击信号。([GitHub][4])

---

## 5. 基于残差收缩时域滤波的 ClassBD 随机冲击抑制方法研究

### Prerequisite Skills

PyTorch、残差结构、soft-thresholding、强噪声信号处理、损失函数设计。

### Project Description

**1. Background**
ClassBD 的时域二次滤波会增强冲击，但强噪声下随机冲击也可能被增强。可以在时域 BD 输出后加入可学习阈值收缩模块，抑制随机噪声尖峰。

**2. Main Task**
在时域滤波输出 (s(t)) 后加入软阈值：

[
\tilde{s}=sign(s)\cdot max(|s|-\tau,0)
]

其中 (\tau) 为可学习阈值，或由网络根据样本自适应生成。

**3. Expected Outcome**

```text
随机噪声尖峰减少
故障周期冲击保留
低 SNR 下误判率降低
```

实验比较：

```text
Original ClassBD
ClassBD + fixed threshold
ClassBD + learnable threshold
```

### Reference Paper and Code

* ClassBD 官方代码：([GitHub][1])
* MNNBD 中提到传统 time-domain criterion 可能受随机脉冲影响，这一问题可作为本题背景依据。([GitHub][3])

---

# 二、频域滤波部分改进：5 个题目

---

## 6. 基于自适应频域门控的 ClassBD 频域滤波改进方法研究

### Prerequisite Skills

FFT/IFFT、复数张量处理、PyTorch FFT、频域 mask、Sigmoid 门控。

### Project Description

**1. Background**
ClassBD 的频域滤波部分通过 FFT、线性神经层和 IFFT 对信号进行频域滤波。官方说明中也指出频域线性滤波器对提高 BD 性能很关键。([GitHub][1])

**2. Main Task**
将频域线性滤波改为自适应频域门控：

[
S(f)=FFT(s)
]

[
G(f)=sigmoid(a(f))
]

[
\tilde{S}(f)=G(f)\odot S(f)
]

[
\tilde{s}=IFFT(\tilde{S})
]

其中 (G(f)) 是可学习频域 mask。

**3. Expected Outcome**

```text
模型自动选择故障相关频带
非故障频带被抑制
可视化 G(f)，提高可解释性
```

目标是在强噪声下比原始 ClassBD 提升 2%—5%。

### Reference Paper and Code

* ClassBD 官方代码：([GitHub][1])
* MathWorks 示例说明 envelope spectrum 和 spectral kurtosis 可用于轴承故障诊断中的频带选择与诊断分析。([MathWorks][5])

---

## 7. 基于自适应共振频带选择的 ClassBD 轴承故障诊断方法研究

### Prerequisite Skills

FFT、带通滤波、Hilbert 包络、包络谱、spectral kurtosis、PyTorch filter bank。

### Project Description

**1. Background**
轴承故障冲击通常激发高频共振带，然后在包络谱中表现出 BPFO、BPFI、BSF、FTF 等故障特征频率。频带选择是包络谱诊断中的关键步骤。Spectral kurtosis 常用于定位最有冲击性的共振频带。([MathWorks][5])

**2. Main Task**
在 ClassBD 频域滤波部分加入可学习带通滤波器组：

[
\tilde{S}_m(f)=H_m(f)S(f)
]

[
\tilde{s}=\sum_m \alpha_m IFFT(\tilde{S}_m)
]

其中 (H_m(f)) 是不同候选频带，(\alpha_m) 是可学习频带权重。

**3. Expected Outcome**

```text
自动选择故障共振频带
包络谱中故障频率峰值增强
比原始 ClassBD 更适合强噪声和弱故障
```

### Reference Paper and Code

* ClassBD 官方代码：([GitHub][1])
* Optimised Spectral Kurtosis 论文说明，选择最佳解调频带对从强干扰中提取轴承故障冲击内容很重要。([ScienceDirect][6])
* MATLAB 轴承诊断示例含 envelope spectrum、spectral kurtosis 和 kurtogram 分析流程。([MathWorks][5])

---

## 8. 基于包络谱故障频率软约束的 ClassBD 频域滤波方法研究

### Prerequisite Skills

Hilbert 变换、包络谱、轴承故障特征频率计算、PyTorch 可微 FFT、损失函数设计。

### Project Description

**1. Background**
ClassBD 的频域滤波不一定显式保证输出信号在故障特征频率及倍频处具有高能量。可以将 BPFO、BPFI、BSF、FTF 等频率信息加入频域损失。

**2. Main Task**
对 ClassBD 输出信号计算包络谱 (E(f))，对样本所属类别 (c) 构造故障频率集合：

[
\Omega_c={f_c,2f_c,3f_c}
]

加入软约束：

[
L_{env}=-\sum_{f\in \Omega_c}|E(f)|^2
]

同时使用宽频带 soft mask，避免频率点过窄导致过拟合。

**3. Expected Outcome**

```text
故障类对应频率及倍频峰值增强
分类准确率提高
包络谱图更容易解释
```

### Reference Paper and Code

* ClassBD 官方代码：([GitHub][1])
* MathWorks 示例列出了 BPFO、BPFI、FTF、BSF 等轴承关键频率，并演示包络谱分析在轴承诊断中的作用。([MathWorks][5])
* MNNBD 中的 frequency.py 包含 Hilbert transform 和 envelope spectrum 相关实现，可作参考。([GitHub][3])

---

## 9. 基于正常样本频域抑制的 ClassBD 误诊降低方法研究

### Prerequisite Skills

包络谱、正常/故障样本标签处理、频域能量统计、损失函数设计。

### Project Description

**1. Background**
很多方法只强调“故障样本要增强故障频率”，但忽略了“正常样本不应该出现故障频率峰值”。这会导致正常样本在强噪声下被误判为故障。

**2. Main Task**
对正常样本增加故障频率抑制损失：

[
L_{normal}=\sum_{f\in \Omega_{fault}}|E(f)|^2
]

其中 (\Omega_{fault}) 包含 BPFO、BPFI、BSF、FTF 及其倍频邻域。

总损失：

[
L=L_{ClassBD}+\lambda_1 L_{env}+\lambda_2 L_{normal}
]

**3. Expected Outcome**

```text
正常样本误判率下降
正常样本包络谱中故障频率峰值降低
强噪声下 Normal vs Fault 区分更清楚
```

### Reference Paper and Code

* ClassBD 官方代码：([GitHub][1])
* MathWorks 示例指出正常信号不显示明显 amplitude modulation，而故障信号会在包络谱中体现故障签名，这可支撑正常样本频域抑制的设计。([MathWorks][5])

---

## 10. 基于时频二阶交互的 ClassBD 滤波融合方法研究

### Prerequisite Skills

时域卷积、FFT、频域门控、特征相乘、PyTorch 多分支网络。

### Project Description

**1. Background**
原始 ClassBD 是时域 quadratic convolutional filter 与频域 linear neural filter 级联。可以进一步改成时域冲击分支与频域故障分支的交互融合，而不是简单串联。

**2. Main Task**
设计两个分支：

[
F_t=TimeBD(x)
]

[
F_f=FreqBD(x)
]

然后进行二阶交互：

[
F=F_t+g\odot(F_t\odot F_f)
]

其中 (F_t) 表示时域冲击特征，(F_f) 表示频域故障频带特征。

**3. Expected Outcome**

```text
只有时域冲击和频域故障特征同时出现时才增强
随机噪声冲击被抑制
强噪声下诊断性能提升
```

实验比较：

```text
Original ClassBD
仅改时域滤波
仅改频域滤波
时频二阶交互 ClassBD
```

### Reference Paper and Code

* ClassBD 官方代码中明确包含时域 quadratic convolutional filter 和 frequency linear neural filter，可作为直接改造对象。([GitHub][1])
* MNNBD 强调时域 kurtosis 与频域 (G-l_1/l_2) criterion 的互补性，可作为时频联合优化的参考依据。([GitHub][3])

---

# 三、推荐优先级

如果目标是本科生做，但又希望发表希望大一些，我建议这样排序：

| 优先级 | 题目                              | 难度 | 发表潜力 |
| --: | ------------------------------- | -: | ---: |
|   1 | 基于门控二阶特征交互的 ClassBD 时域滤波改进方法研究  |  中 |    高 |
|   2 | 基于自适应共振频带选择的 ClassBD 轴承故障诊断方法研究 |  中 |    高 |
|   3 | 基于包络谱故障频率软约束的 ClassBD 频域滤波方法研究  |  中 |    高 |
|   4 | 基于时频二阶交互的 ClassBD 滤波融合方法研究      | 中高 |    高 |
|   5 | 基于周期感知时域卷积核的 ClassBD 改进方法研究     |  中 |   中高 |
|   6 | 基于自适应频域门控的 ClassBD 频域滤波改进方法研究   |  中 |   中高 |
|   7 | 基于正常样本频域抑制的 ClassBD 误诊降低方法研究    |  中 |   中高 |
|   8 | 基于多尺度时域盲去卷积的 ClassBD 改进方法研究     |  中 |    中 |
|   9 | 基于自适应高阶时域卷积的强噪声轴承故障诊断方法研究       | 中高 |   中高 |
|  10 | 基于残差收缩时域滤波的 ClassBD 随机冲击抑制方法研究  |  中 |    中 |

---

# 四、统一实验设计

每个题目都可以只做：

```text
Original ClassBD vs Improved ClassBD
```

但必须增加对应的特殊指标：

| 方向       | 除 ACC/F1 外建议增加的指标              |
| -------- | ------------------------------ |
| 门控二阶交互   | 门控权重分布、输出冲击峰值、包络谱              |
| 高阶时域卷积   | (\alpha_2,\alpha_3) 权重变化、训练稳定性 |
| 多尺度时域滤波  | 不同卷积核尺度消融                      |
| 周期感知卷积   | 包络自相关故障周期峰值                    |
| 残差收缩滤波   | 随机尖峰抑制比例、输出信噪比                 |
| 自适应频域门控  | 频域 mask 可视化                    |
| 共振频带选择   | 选中频带中心频率、带宽                    |
| 包络谱约束    | 故障频率能量占比                       |
| 正常样本频域抑制 | 正常类故障频率能量下降                    |
| 时频二阶交互   | 时域/频域/交互三组消融                   |

---

# 五、最推荐的组合题目

如果你想从这 10 个里面选最强的一个，我建议选：

> **基于门控二阶特征交互与自适应共振频带选择的 ClassBD 改进方法研究**

它同时改了 ClassBD 的两个核心结构：

```text
时域滤波：输入级二次卷积 → 门控二阶特征交互
频域滤波：频域线性滤波 → 自适应共振频带选择
```

这个题目比单独加一个损失函数更像“方法创新”，也更贴近 ClassBD 原始结构。

[1]: https://github.com/asdvfghg/ClassBD "GitHub - asdvfghg/ClassBD · GitHub"
[2]: https://arxiv.org/abs/2404.15341?utm_source=chatgpt.com "Classifier-guided neural blind deconvolution: a physics-informed denoising module for bearing fault diagnosis under heavy noise"
[3]: https://github.com/asdvfghg/MNNBD "GitHub - asdvfghg/MNNBD · GitHub"
[4]: https://github.com/aresmiki/Blind-deconvolution-based-on-the-ratio-of-cyclic-content?utm_source=chatgpt.com "Blind-deconvolution-based-on-the-ratio-of-cyclic-content"
[5]: https://www.mathworks.com/help/predmaint/ug/Rolling-Element-Bearing-Fault-Diagnosis.html "Rolling Element Bearing Fault Diagnosis - MATLAB & Simulink
"
[6]: https://www.sciencedirect.com/science/article/abs/pii/S0888327015005993 "Optimised Spectral Kurtosis for bearing diagnostics under electromagnetic interference - ScienceDirect"



---------------------------------------------------------------------------------------------------------------------------------------
# final year project UOG 2025-2026
1. (舒森鑫），海南8- blind radar signal restoration with GAN. [BRSR-OpGAN: Blind Radar Signal Restoration using Operational Generative Adversarial Network](https://github.com/MUzairZahid/Blind-Radar-Signal-Restoration)
   (冉常红）[“Unsupervised Blind Source Separation with Variational Auto-Encoders“](https://github.com/jundsp/VAE-BSS)[Blind Source Separation of Radar Signals in Time
Domain Using Deep Learning)(https://arxiv.org/pdf/2509.15603)
2. 海南9- blind radar signal restoration with regressor network [CoRe-Net: Co-Operational Regressor Network with Progressive Transfer Learning for Blind Radar Signal Restoration](https://github.com/MUzairZahid/Blind-Radar-Signal-Restoration)
3. 海南10-passive target detection based on blind channel identification [Multi-target Detection by Distributed Passive Radar Systems without Reference Signals](https://github.com/LiuRuiQi/Blind-Channel-Identification)
4. [__成都1-7__2025-2026-Comparison-of-Blind-Source-Separation-techniques: 7 papers/methods](https://github.com/TUIlmenauAMS/Comparison-of-Blind-Source-Separation-techniques) [Fast time domain stereo audio source separation using fractional delay filters](https://aes2.org/publications/elibrary-page/?id=20583)
5. 成都9-Esitmate the blur kernel to extend the paper [Joint Multichannel Deconvolution and Blind Source Separation](https://github.com/CEA-jiangming/DecGMCA?tab=readme-ov-file)
6. [A Nonconvex Approach for Exact and Efficient Multichannel Sparse Blind Deconvolution (1d and 2D)](https://github.com/qingqu06/MCS-BD/tree/master)
7. __成都10-海南1-6 Chatgpt for how__ using bayesian version to extend the paper(estimate number of sources/kernels; source are gradient sparse, L0 sparse, double sparsity,using selfdeblur): Multisource Multichannel[Deconvolutional Unrolled Neural Learning (DUNL) for Computational Neuroscience](https://github.com/btolooshams/dunl-compneuro): Insert a batch normalization layer after each convolution in the encoder. Assess training speed and generalization; Replace standard convolutions with dilated convolutions in the encoder. Train and test on one dataset;Change the network so each unrolled layer’s sparsity threshold is a learnable parameter. Analyze event detection and reconstruction quality; Modify the unrolled DUNL network so each layer’s step size is a learnable parameter. Train and compare with baseline.
8. [Semi-Blind Source Separation with Learned Constraints](https://github.com/RCarloniGertosio/sGMCA)
9. [Blind Source Separation with Non-Coplanar Interferometric Data](https://github.com/RCarloniGertosio/wGMCA)
10. [成都8-2d-Joint deconvolution and unsupervised source separation for data on the sphere(2d)](https://github.com/RCarloniGertosio/2DecGMCA)
11. [1d-Joint deconvolution and unsupervised source separation for data on the sphere (1d)](https://github.com/RCarloniGertosio/SDecGMCA)
12. [海南7-MSSP2025 Classifier-guided neural blind deconvolution-A physics-informed denoising module for bearing fault diagnosis under noisy conditions (https://www.sciencedirect.com/science/article/pii/S0888327024006484)

# final year project SICE 2025-2026
## multisource and multichannel blind deconvolution
1. [orchmfbd: a flexible multi-object multi-frame blind deconvolution code](https://github.com/aasensio/torchmfbd)
2. [Learning to do multiframe blind deconvolution unsupervisedly](https://github.com/aasensio/unsupervisedMFBD?utm_source=chatgpt.com)
3. [Focused blind deconvolution](https://github.com/pawbz/FocusedBlindDecon.jl?utm_source=chatgpt.com)
4. [Learning Blind Motion Deblurring](https://github.com/cgtuebingen/learning-blind-motion-deblurring?utm_source=chatgpt.com)
5. [image restoration](https://zoi.utia.cas.cz/index.php/research/restoration)
6. [Accelerating Multiframe Blind Deconvolution via Deep Learning](https://github.com/aasensio/neural-MFBD)
7. [New algorithms for sparse multichannel blind deconvolution](https://github.com/kenjinose/smbd_algorithms)
8. 
