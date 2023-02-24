--------------------------------------------------------------------------------

English | [简体中文](./README_cn.md)

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-1ca0f1.svg?logo=twitter&logoColor=white)](https://twitter.com/PaddlePaddle_)


# 扩散卷积神经网络





# 环境配置

开发环境





主要依赖于PaddlePaddle开发，相关依赖版本为：

```plain
paddle==2.4.1
scipy
numpy
pandas
pyyaml==5.4.1
tables
visualdl
```

# 代码结构

流程：

1.依据交通数据文件(h5格式)，生成 train/test/val 数据集。

2.基于预先计算的传感器之间的道路网络距离和传感器ID，进行图构建。

3.使用上述得到的数据集，基于DCRNN模型进行训练。



模块(包含pytorch转换paddle相关重要操作）：

数据准备模块



图构建模块



模型模块



​     模型参数相关转换



训练模块



​     模型训练相关转换



# 运行方式

### (1) 数据准备

环境需求：numpy pandas tables

输入数据：metr-la.h5(洛杉矶), pems-bay.h5(湾区)  这些数据由源代码作者提供

输出数据：test.npz, train.npz, val.npz

调用代码：

```plain
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

### (2) 图构建

环境需求：numpy pandas tables

输入数据：distance_la_2012.csv, graph_sensor_ids.txt

输出数据：adj_mx.pkl

调用代码：

```plain
python gen_adj_mx.py  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1\
       --output_pkl_filename=data/sensor_graph/adj_mx.pkl
```

这部分目前原作者只提供了洛杉矶地区的原始数据，湾区的数据直接提供了adj_mx_bay.pkl文件

### （1）（2）步均不涉及tensorflow，pytorch相关模块，可直接调用。

### (3) 模型训练

环境需求：paddle scipy numpy pandas pyyaml==5.4.1 tables visualdl

由于训练过程中涉及到稀疏矩阵相关操作，所以paddle≥2.4 CUDA >11，且需要支持GPU训练。

调用代码：

```plain
# METR-LA
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml

# PEMS-BAY
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay.yaml
```

dcrnn_la.yaml文件中包含了训练所需要的相关超参数设定。

# 数据介绍





# pytorch转paddle注意事项

部分API转换和注意事项可参考：

https://aistudio.baidu.com/aistudio/projectdetail/4470683

https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html

