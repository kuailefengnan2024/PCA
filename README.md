# PCA 实战项目：Eigenfaces（特征脸）

这是一个使用 Python 实现主成分分析（PCA）并应用于真实人脸数据集的实战项目。

## 项目结构

```
/
├── pca/
│   ├── __init__.py
│   └── pca.py                  # 自定义PCA算法实现
├── eigenfaces_analysis.py      # 主程序脚本
└── requirements.txt            # 项目依赖
```

## 环境搭建

1.  **克隆或下载项目**
    将项目文件保存到您的本地计算机。

2.  **创建虚拟环境 (推荐)**
    在项目根目录打开终端，运行以下命令：
    ```bash
    python -m venv venv
    ```
    激活虚拟环境：
    *   Windows: `.\venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`

3.  **安装依赖**
    在激活虚拟环境的终端中，运行：
    ```bash
    pip install -r requirements.txt
    ```

## 如何运行

项目的所有逻辑都封装在 `eigenfaces_analysis.py` 中。要运行此项目，只需在终端中执行：

```bash
python eigenfaces_analysis.py
```

**首次运行注意事项：**
程序会从 scikit-learn 的服务器自动下载 LFW (Labeled Faces in the Wild) 人脸数据集。根据您的网络情况，这可能需要几分钟时间。下载过程中，您会在终端看到相应的日志信息。数据集下载完成后会自动缓存到本地，后续运行将不再需要下载。

## 预期输出

程序运行后，会弹出两个 `matplotlib` 绘图窗口：

1.  **第一个窗口：Eigenfaces**
    这个窗口会展示从数据集中提取出的最重要的12个主成分，也就是“特征脸”。它们看起来像一张张模糊、诡异的人脸，代表了数据集中人脸变化的主要模式。

2.  **第二个窗口：Original vs Reconstructed Faces**
    这个窗口会展示6组成对的图像，左边是数据集中原始的人脸照片，右边是仅使用前150个“特征脸”重建出来的对应人脸。您可以直观地看到PCA是如何用少量核心特征来逼近原始图像的。

关闭弹出的所有绘图窗口后，程序即结束运行。
