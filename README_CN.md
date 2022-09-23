# 目录

<!-- TOC -->

- [目录](#目录)
- [Conv-Tasnet介绍](#Conv-Tasnet介绍)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [数据预处理过程](#数据预处理过程)
        - [数据预处理](#数据预处理)
    - [训练过程](#训练过程)
        - [训练](#训练)  
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出mindir模型](#导出mindir模型)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# Conv-Tasnet介绍

全卷积时域音频分离网络（Conv-TasNet）由三个处理阶段组成, 编码器、分离和解码器。首先，编码器模块用于将混合波形的短段转换为它们在中间特征空间中的对应表示。然后，该表示用于在每个时间步估计每个源的乘法函数（掩码）。然后，利用解码器模块对屏蔽编码器特征进行变换，重构源波形
Conv-Tasnet被广泛的应用在语音分离等任务上，取得了显著的效果

[论文](https://arxiv.org/abs/1809.07454): Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation

# 模型架构

模型包括  
encoder：类似fft，提取语音特征。
decoder：类似ifft，得到语音波形
separation：类似得到mask，通过mix*单个语音的mask，类似得到单个语音的一个语谱图。通过decoder还原出语音波形。

# 数据集

使用的数据集为: [librimix](<https://catalog.ldc.upenn.edu/docs/LDC93S1/TIMIT.html>)，LibriMix 是一个开源数据集，用于在嘈杂环境中进行源代码分离。
要生成 LibriMix，请参照开源项目：https://github.com/JorisCos/LibriMix

# 环境要求

- 硬件（ASCEND）
    - ASCEND处理器
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 通过下面网址可以获得更多信息:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 依赖
    - 见requirements.txt文件，使用方法如下:

```python
pip install -r requirements.txt
 ```

# 脚本说明

## 脚本及样例代码

```path
conv-tasnet
├─ README.md                          # descriptions
├── scripts
  ├─ run_distribute_train.sh          # launch ascend training(8 pcs)
  └─ run_stranalone_train.sh          # launch ascend training(1 pcs)
  └─ run_eval.sh                      # launch ascend eval
├─ train.py                           # train   script
├─ evaluate.py                        # eval
├─ preprocess.py                      # preprocess json
├─ data.py                            # postprocess data
└─ export.py                          # export mindir script
├─ network_define.py                  # define network
├─ model.py                           # conv_tasnet
├─ loss.py                            # loss function

```

## 脚本参数

数据预处理、训练、评估的相关参数在`train.py`等文件

```text
数据预处理相关参数
in-dir                    预处理前加载原始数据集目录
out-dir                   预处理后的json文件的目录
sample-rate               采样率
train_name                预处理后的训练MindRecord文件的名称
test_name                 预处理后的测试MindRecord文件的名称  
```

```text
训练和模型相关参数
train_dir                  训练集
valid_dir                  测试集
segment                    取得音频的长度
cv_maxlen                  最大音频长度
N                          autoencoder的卷积核数量
L                          拆分后音频长度
B                          瓶颈层卷积核通道数
H                          分离模块卷积块通道数
P                          卷积核大小
X                          分离层每个循环中重复的卷积块数量
R                          分离层中重复次数
C                          说话者数量
lr                         学习率
l2                         权重衰减
momentum                   动量
```

```text
评估相关参数
model_path                 ckpt文件
data-dir                   测试集路径
batch_size                 测试集batch大小
```

```text
配置相关参数
device_traget                硬件，只支持ASCEND
device_id                    设备号
```

# 数据预处理过程

## 数据预处理

数据预处理运行示例:

```text
python preprocess.py
```

数据预处理过程很快，大约需要三分钟时间

# 训练过程

## 训练

- ### 单卡训练

运行示例:

```text
python train.py
参数:
train_dir                  训练集
valid_dir                  测试集
segment                    取得音频的长度
cv_maxlen                  最大音频长度
N                          autoencoder的卷积核数量
L                          拆分后音频长度
B                          瓶颈层卷积核通道数
H                          分离模块卷积块通道数
P                          卷积核大小
X                          分离层每个循环中重复的卷积块数量
R                          分离层中重复次数
C                          说话者数量
lr                         学习率
l2                         权重衰减
momentum                   动量
```

或者可以运行脚本:

```bash
bash run_standalone_train.sh [DEVICE_ID]
```

上述命令将在后台运行，可以通过train.log查看结果  
每个epoch将运行10小时左右

- ### 分布式训练

分布式训练脚本如下

```bash
bash run_distribute_train.sh 2 1 /home/hccl_2p_01_127.0.0.1.json
```

# 评估过程

## 评估

运行示例:

```text
python eval.py
参数:
model_path                 ckpt文件
data-dir                   测试集路径
batch_size                 测试集batch大小
```

或者可以运行脚本:

```bash
bashrun_eval.sh [DEVICE_ID]
```

上述命令在后台运行，可以通过eval.log查看结果,测试结果如下

# 导出mindir模型

## 导出

```bash
python export.py
```

# 推理过程

## 推理

### 用法

```bash
bash scripts/run_infer_310.sh [MINDIR_PATH] [TEST_PATH] [NEED_PREPROCESS]
```

### 结果

```text
Average SISNR improvement: 12.59
```

# 模型描述

## 性能

### 训练性能

| 参数                 | Conv-Tasnet                                                      |
| -------------------------- | ---------------------------------------------------------------|
| 资源                   | Ascend910             |
| 上传日期              | 2022-8-25                                    |
| MindSpore版本           | 1.6.1                                                          |
| 数据集                    | Librimix                                                 |
| 训练参数       | 1p, epoch = 100, batch_size = 2, lr=0.01   |
| 优化器                  | SGD                                                           |
| 损失函数              | SI-SNR                                |
| 输出                    | SI-SNR(11.8)                                                    |
| 损失值                       | -14.5                                                       |
| 运行速度                      | 1p 585.295 ms/step                                   |
| 训练总时间       | 1p:约1000h;                                  |                                           |

# 随机情况说明

随机性主要来自下面两点:

- 参数初始化
- 轮换数据集

# ModelZoo主页

 [ModelZoo主页](https://gitee.com/mindspore/models).
