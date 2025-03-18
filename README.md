# Clustering-Algorithm-and-Application-Based-on-Quantum-Machine-Learning
项目利用K-means的改造算法——量子K-means（QK-means）算法完成MTEB(Massive Text Embedding Benchmark)的10个真实的文本聚类任务和随机的自建数据集的聚类任务。通过v-measure和绘图法评价QK-means的聚类能力，试图证明QK-means能够与经典K-means算法有同样的聚类能力，在解决实际聚类问题中也有不错的表现。
### README 说明

#### 项目概述
本项目基于 Jupyter Notebook 开发，旨在实现量子 K-means（QK-means）聚类算法，并结合经典机器学习和量子计算技术对文本数据进行聚类分析。代码使用了 Qiskit 进行量子计算模拟，结合 `sentence-transformers` 对文本数据生成嵌入向量，最终通过聚类评估指标（如 V-measure）量化聚类效果。

---

#### 文件结构
- **`Pasted_Text_1742280999973.txt`**: 包含主要代码逻辑的文本文件。
- **数据集目录**: 数据集存储在路径 `D:\HF dataset\mteb\` 下，包含多个子数据集。
- **模型缓存目录**: 预训练模型存储在路径 `D:\HF-model\sentence-t5-base` 中。

---

#### 代码功能分析

1. **导入必要的库**
   ```python
   from sklearn.datasets import make_blobs
   import numpy as np
   import matplotlib.pyplot as plt
   import math
   import time
   ```
   - `numpy`: 提供高效的数值计算功能。
   - `math`: 用于数学运算。
   - `matplotlib`: 用于可视化。
   - `time`: 计算运行时间。

2. **量子计算相关库**
   ```python
   from qiskit import Aer, execute
   from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
   from qiskit.tools.visualization import plot_histogram
   ```
   - 使用 Qiskit 模拟器 `Aer` 进行量子电路模拟。
   - 定义量子寄存器、经典寄存器和量子电路。

3. **数据加载与预处理**
   ```python
   def get_data(dataset_name, model_name, data_num):
       ...
   ```
   - 从本地路径加载数据集。
   - 使用 `SentenceTransformer` 将文本数据转换为嵌入向量。
   - 随机采样指定数量的数据点（`data_num`）。

4. **量子距离计算**
   ```python
   def get_theta(d):
       ...
   def get_Distance(x, y):
       ...
   ```
   - `get_theta`: 根据输入数据计算角度参数。
   - `get_Distance`: 构建量子电路，利用量子态叠加和干涉原理计算两点之间的距离。

5. **聚类算法**
   ```python
   def find_nearest_neighbour(points, centroids):
       ...
   def find_centroids(points, centers):
       ...
   ```
   - `find_nearest_neighbour`: 找到每个点最近的聚类中心。
   - `find_centroids`: 根据分配结果重新计算聚类中心。

6. **聚类评价**
   ```python
   from mteb.evaluation.evaluators import ClusteringEvaluator
   clusterer = ClusteringEvaluator(sentences=sentences, labels=labels)
   result = clusterer(model)
   ```
   - 使用 `ClusteringEvaluator` 评估聚类效果，输出 V-measure 等指标。

7. **主循环**
   ```python
   for qq in range(10):
       ...
   ```
   - 遍历多个数据集，执行 QK-means 聚类。
   - 输出聚类结果和运行时间。

---

#### 运行环境
- Python 版本：建议使用 Python 3.8 或更高版本。
- 依赖库：运行代码前，请确保安装以下库：
  ```bash
  pip install numpy matplotlib scikit-learn qiskit sentence-transformers datasets mteb
  ```

---

#### 使用方法
1. 克隆或下载本项目代码。
2. 安装所需的依赖库。
3. 将数据集放置在 `D:\HF dataset\mteb\` 目录中。
4. 将预训练模型缓存到 `D:\HF-model\sentence-t5-base`。
5. 打开 Jupyter Notebook 并运行代码单元。
6. 查看生成的 CSV 文件，分析聚类结果和运行时间。

---

#### 注意事项
- **内存不足问题**:
  - 如果数据集较大，可能导致内存不足（`RuntimeError: not enough memory`）。建议减少 `data_num` 或优化数据加载方式。
- **量子计算模拟**:
  - 当前代码使用 Qiskit 的模拟器运行量子电路，实际硬件可能需要进一步调整。
- **数据集路径**:
  - 确保数据集路径正确，反斜杠 `\` 是必需的，否则可能导致路径解析错误。

---

#### 示例输出
代码会在每次迭代后生成一个 CSV 文件，文件名格式为：
```
20组重复 arxiv-clustering-p2p sentence-t5-base.csv
```
文件内容示例：
```
v_measure,标签种类(k)
0.85,5
0.87,5
...
```
