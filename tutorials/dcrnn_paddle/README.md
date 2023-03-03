# 扩散卷积神经网络

时空预测是动态环境学习系统中的一个关键任务，其在神经科学，气候，交通运输等领域中都有着广泛应用。交通预测是这类学习任务的一个典型例子。交通预测的目标是在历史交通速度和潜在道路网络的基础上预测一个传感器网络的未来交通速度。由于复杂的时空依赖性和长期预测的固有困难，这类任务十分具有挑战性。

扩散卷积神经网络（DCRNN）将交通流的动力学建模为一个扩散过程，并使用扩散卷积操作来捕获空间依赖性。具体来说，DCRNN使用图上的双向随机游动来捕获空间依赖性，使用具有计划采样的解码器-编码器架构来捕获时间依赖性。并且，DCRNN并不局限于交通运输，其也可很容易地适用于其他时空预测任务。

# 环境配置

开发环境

- Windows10
- Python=3.8.16
- pip=22.3.1
- conda=22.9.0

nvidia-smi

```
NVIDIA-SMI 528.49       Driver Version: 528.49       CUDA Version: 12.0
```

nvcc -V

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Fri_Dec_17_18:28:54_Pacific_Standard_Time_2021
Cuda compilation tools, release 11.6, V11.6.55
Build cuda_11.6.r11.6/compiler.30794723_0
```

主要依赖于PaddlePaddle开发，相关依赖版本为：

```
paddle==2.4.1
scipy==1.10.1
numpy==1.23.2
pandas==1.5.3
pyyaml==5.4.1
tables==3.8.0
visualdl==2.5.0
```

# 代码结构

流程：

- 依据交通数据文件(h5格式)，生成 train/test/val 数据集。

- 基于预先计算的传感器之间的道路网络距离和传感器ID，进行图构建。

- 使用上述得到的数据集，基于DCRNN模型进行训练。

# 运行方式

## 数据准备

训练使用的数据为洛杉矶和湾区的交通数据文件，即`metr-la.h5`和`pems-bay.h5`。这些数据可从[谷歌云盘](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) 或[百度云](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g)中获得，然后将这些输入放入`data/`。

运行以下命令即可依据对应的交通数据文件，生成train/test/val 数据集。

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

## 图构建

由于图构建的实现需要基于预先计算的传感器之间的道路网络距离和传感器ID，且目前只提供洛杉矶的相关数据，因此该流程目前只支持洛杉矶的传感器ID。

```
python gen_adj_mx.py  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1\
       --output_pkl_filename=data/sensor_graph/adj_mx.pkl
```

此外，洛杉矶的传感器位置可在`data/sensor_graph/graph_sensor_locations.csv`中获得。

## 模型训练

以下分别是在`metr-la`和`pems-bay`上训练模型的命令。

```
# METR-LA
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml

# PEMS-BAY
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay.yaml
```

`dcrnn_la.yaml`和`dcrnn_bay.yaml`中包含了训练所需要的相关超参数设定。

# PyTorch转Paddle注意事项

相关API转换和注意事项可参考：

[PyTorch 1.8 与 Paddle 2.0 API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html)

[Torch 转 PaddlePaddle 实战(避坑指南)](https://aistudio.baidu.com/aistudio/projectdetail/4470683)
