### README

# 深度学习课程作业: 使用CNN和LSTM进行DNA序列分类

本仓库包含了使用卷积神经网络（CNN）和长短期记忆网络（LSTM）对DNA序列进行分类的代码，以及进行消融实验和与经典机器学习方法比较的代码。

## 仓库结构

- `train.py`: 训练CNN-LSTM模型的主代码。
- `exp.py`: 进行消融实验的代码，通过训练和评估仅使用CNN或仅使用LSTM的模型来比较性能。
- `ml_exp.py`: 与经典机器学习方法（如随机森林和XGBoost）进行比较的代码。
- `figures/`: 存储精确率-召回率曲线和其他结果可视化的目录。
- `dm3.kc167.example.h5`: 包含DNA序列数据集的数据文件。

## 环境依赖

运行本仓库中的代码需要安装以下Python包：

- `h5py`
- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `xgboost`

可以使用以下命令安装所需的包：

```bash
pip install h5py torch numpy scikit-learn matplotlib xgboost
```
