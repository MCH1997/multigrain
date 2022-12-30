# Multigrain 

<!-- TOC -->

- [Multigrain ](#Multigrain)
    - [概述](#概述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [特性](#特性)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本描述](#脚本描述)
        - [脚本代码结构](#脚本代码结构)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
            - [Ascend 910](#ascend-910)
            - [GPU](#gpu)
        - [推理过程](#推理过程)
            - [Ascend 910](#ascend-910-1)
            - [GPU](#gpu-1)
    - [模型描述](#模型描述)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo首页](#modelzoo首页)

<!-- /TOC -->

## 概述

MultiGrain是一种网络架构，产生的紧凑向量表征，既适合于图像分类，又适合于特定对象的检索。它建立在一个标准分类主干上。网络的顶部产生包含粗粒度和细粒度信息的嵌入，这样就可以根据对象类别、特定对象或是否失真的副本对图像进行识别。网络模型结构建立在一个标准分类主干上，在图像检索中主要是对pooling层的改进，使用Gempooling，使用了结合两种损失函数的loss。交叉熵损失用于分类。检索使用triplet loss损失。
检索部分使用数据集：Imagenet2012

## 模型架构

Multigrain的总体网络架构如下：[链接](https://arxiv.org/abs/1902.05509.pdf)

## 数据集

使用的数据集：ImageNet2012

- 数据集大小：共1000个类的224*224彩色图像
    - 训练集：1,281,167张图像
    - 测试集：5万张图像

- 数据格式：JPEG
    - 注：数据在dataset.py中处理。

- 下载数据集ImageNet2012。

> 解压ImageNet2012数据集到任意路径，目录结构应包含训练数据集和验证数据集，如下所示：

```shell
    ├── train                   # 训练数据集
    └── val                     # 验证数据集
```

## 环境要求

- 硬件：昇腾处理器（Ascend或GPU）
    - 使用Ascend或GPU处理器搭建硬件环境。

## 快速入门

- Ascend处理器环境运行

```python
# 分布式训练运行示例
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [DEVICE_NUM]

# 推理运行示例
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

> 对于分布式训练，需要提前创建JSON格式的HCCL配置文件。关于配置文件，可以参考[HCCL_TOOL](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)
。

- GPU处理器环境运行

```python
# 分布式训练运行示例
bash run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM]

# 推理运行示例
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
 ```

## 脚本描述

### 脚本代码结构

```shell
└──multigrain
  ├── README.md
  ├── config                              # 参数配置   # 高性能版本：性能提高超过10%而精度下降少于1%
    ├── resnet50_imagenet2012_Ascend_Thor_config.yaml
    ├── resnet50_imagenet2012_config.yaml
    ├── resnet50_imagenet2012_GPU_Thor_config.yaml
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend评估
    ├── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
    ├── run_distribute_train_gpu.sh        # 启动GPU分布式训练（8卡）
    ├── run_eval_gpu.sh                    # 启动GPU评估
    ├── run_standalone_train_gpu.sh        # 启动GPU单机训练（单卡）
  ├── src
    ├── dataset.py                         # 数据预处理
    ├── eval_callback.py                   # 训练时推理回调函数
    ├── CrossEntropySmooth.py              # ImageNet2012数据集的损失定义
    ├── mixLoss.py                         # 联合损失定义
    ├── lr_generator.py                    # 生成每个步骤的学习率
    └── resnet.py                          # ResNet骨干网络，包括ResNet50、ResNet101和SE-ResNet50
    ├── model_utils
       ├── config.py                       # 参数配置
       ├── device_adapter.py               # 设备配置
       ├── local_adapter.py                # 本地设备配置
       └── moxing_adapter.py               # modelarts设备配置
  ├── eval.py                              # 评估网络
  ├── export.py                            # 导出IR模型
  └── train.py                             # 训练网络
```

### 脚本参数

在config.py中可以同时配置训练和推理参数。

- Ascend 910参数说明

```shell
"class_num":1001,                	# 数据集类数
"batch_size":480,                	# 输入张量的批次大小
"loss_scale":1024,			# 损失等级
"momentum":0.9, 			# 动量优化器
"weight_decay":1e-4,             	# 权重衰减
"epoch_size":50,                 	# 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,  	 	# 加载预训练检查点之前已经训练好的模型的周期大小
"save_checkpoint":True,          	# 是否保存检查点
"save_checkpoint_epochs":5,      	# 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        	# 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":2, 			# 热身周期数
"lr_decay_mode":"Linear",        	# 用于生成学习率的衰减模式
"use_label_smooth":True,           	# 标签平滑
"label_smooth_factor":0.1,      	# 标签平滑因子
"lr_init":0.05672                       # 初始学习率
"lr_decay":4.9687,  			# 学习率衰减率值
"lr_end_epoch":70, 			# 学习率结束epoch值
"damping_init":0.02345,      		# 阻尼衰减率
"damping_decay":0.5467,   		# 更新二阶矩阵的步长间隔
"frequency": 834,                       # 更新二阶信息矩阵的步长间隔（应为每个epoch step数的除数）
```

- GPU参数

```shell
"class_num":1001,                	# 数据集类数
"batch_size":480,                	# 输入张量的批次大小
"loss_scale":1024,			# 损失等级
"momentum":0.9, 			# 动量优化器
"weight_decay":1e-4,             	# 权重衰减
"epoch_size":50,                 	# 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,  	 	# 加载预训练检查点之前已经训练好的模型的周期大小
"save_checkpoint":True,          	# 是否保存检查点
"save_checkpoint_epochs":5,      	# 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        	# 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":2, 			# 热身周期数
"lr_decay_mode":"Linear",        	# 用于生成学习率的衰减模式
"use_label_smooth":True,           	# 标签平滑
"label_smooth_factor":0.1,      	# 标签平滑因子
"lr_init":0.05672                       # 初始学习率
"lr_decay":4.9687,  			# 学习率衰减率值
"lr_end_epoch":70, 			# 学习率结束epoch值
"damping_init":0.02345,      		# 阻尼衰减率
"damping_decay":0.5467,   		# 更新二阶矩阵的步长间隔
"frequency": 834,                       # 更新二阶信息矩阵的步长间隔（应为每epoch step数的除数）
```

> 由于算子的限制，目前Ascend中batch size只支持3的倍数。

### 训练过程

#### Ascend 910

```shell
  bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [DEVICE_NUM]
```

训练结果保存在当前路径下，文件夹名称以“train_parallel”开头。您可在日志中找到checkpoint文件以及结果，如下所示。

```shell
...
epoch: 2 step: 995, loss is 4.1976256
epoch: 2 step: 995, loss is 4.6285763
epoch: 2 step: 995, loss is 3.7275014
epoch: 2 step: 995, loss is 3.2510743
epoch: 2 step: 995, loss is 3.963184
epoch: 2 step: 995, loss is 3.4830532
...
```

#### GPU

```shell
bash run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM]
```

训练结果保存在当前路径下，文件夹名称以“train_parallel”开头。您可在日志中找到checkpoint文件以及结果，如下所示。

```shell
...
epoch: 2 step: 1150, loss is 4.791112
epoch: 2 step: 1151, loss is 4.2517877
epoch: 2 step: 1152, loss is 5.4372764
epoch: 2 step: 1153, loss is 5.95784
epoch: 2 step: 1154, loss is 5.1854725
epoch: 2 step: 1155, loss is 5.3210454
...
```

### 推理过程

在运行以下命令之前，请检查用于推理的checkpoint路径。请将checkpoint路径设置为绝对路径，如`username/resnet_thor/train_parallel0/resnet-42_5004.ckpt`。

#### Ascend 910

```shell
  bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

此脚本需设置两个参数：

- `DATASET_PATH`：验证数据集的路径。
- `CHECKPOINT_PATH`：checkpoint文件的绝对路径。

> 训练过程中可以生成checkpoint。

推理结果保存在示例路径，文件夹名为`eval`。您可在日志中找到如下结果。

```shell
  result: {'top_1_accuracy': 0.7626506024096386, 'top_5_accuracy': 0.9281124497991968} ckpt= /home/work/user-job-dir/checkpoint.ckpt
```

#### GPU

```shell
  bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

推理结果保存在示例路径，文件夹名为`eval`。您可在日志中找到如下结果。

```shell
  result: {'top_1_accuracy': 0.7626506024096386, 'top_5_accuracy': 0.9281124497991968} ckpt= /home/work/user-job-dir/checkpoint.ckpt
```

## 模型描述

### 训练性能

| 参数 | Ascend 910                                                                       | GPU                                                                          |
| -------------------------- |----------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| 模型版本 | Multigrain                                                                       | Multigrain                                                                   |
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8                                  | GPU(Tesla V100 SXM2)-CPU 2.1GHz 24核-内存128G                                   |
| 上传日期 | 2022-11-13                                                                       | 2022-11-13                                                                   |
| MindSpore版本 | 1.5.0                                                                            | 1.5.0                                                                        |
| 数据集 | ImageNet2012                                                                     | ImageNet2012                                                                 |
| 训练参数 | epoch=45, steps per epoch=5338, batch_size = 30                                  | epoch=45, steps per epoch=5338, batch_size = 30                              |
| 优化器 | THOR                                                                             | THOR                                                                         |
| 损耗函数 | mix_loss混合损失                                                                     | Softmax交叉熵                                                                   |
| 输出 | 概率                                                                               | 概率                                                                           |
| loss | 1.6453942                                                                        | 1.645802                                                                     |
| Speed | 20.4毫秒/步（8卡）                                                                     | 76毫秒/步（8卡）                                                                   |
| 总时间（按75.9%计算） | 10h25m55s                                                                              | 9h28m34s                                                                     |
| 参数(M) | 25.5                                                                             | 25.5                                                                         |
| checkpoint | 1.05G（.ckpt file）                                                                | 198M（.ckpt file）                                                             |


### 推理性能

| 参数                 | Ascend 910             | GPU           |
| ------------------- |------------------------|---------------|
| 模型版本             | Multigrain             | Multigrain |
| 资源                 | Ascend 910；系统 Euler2.8 | GPU           |
| 上传日期              | 2022-11-13             | 2022-11-13    |
| MindSpore版本        | 1.5.0                  | 1.5.0         |
| 数据集               | ImageNet2012           | ImageNet2012  |
| 批大小               | 30                     | 30            |
| 输出                 | 概率                     | 概率            |
| 精度                | 76.26%                 | 76.01%        |
| 推理模型             | 198M (.air file)       |               |


## ModelZoo首页

 请查看官方[主页](https://gitee.com/mindspore/models)
 。  
