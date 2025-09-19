import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from pca.pca import PCA
import logging

# 配置日志，以便了解数据下载情况
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """
    一个辅助函数，用于在一个窗口中绘制多张图片
    """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        # 将一维向量重新整形为二维图像
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def run_eigenfaces_analysis():
    """
    运行 Eigenfaces 分析案例
    """
    # 1. 下载并加载LFW人脸数据集
    # min_faces_per_person=70: 只保留数据集中至少有70张照片的人
    # resize=0.4: 缩小图片尺寸以加快计算速度
    logging.info("正在下载LFW人脸数据集...")
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    logging.info("数据集下载完成.")

    # 获取图像的维度
    n_samples, h, w = lfw_people.images.shape
    
    # 获取图像数据（像素矩阵）和标签（人名）
    X = lfw_people.data
    n_features = X.shape[1]
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    logging.info(f"总数据集大小:")
    logging.info(f"样本数 (n_samples): {n_samples}")
    logging.info(f"特征数/像素数 (n_features): {n_features}")
    logging.info(f"类别数/人数 (n_classes): {n_classes}")

    # 2. 运行PCA，提取前150个主成分
    n_components = 150
    logging.info(f"正在从 {n_features} 个特征中提取前 {n_components} 个主成分...")
    
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    logging.info("PCA计算完成.")

    # 3. 可视化主成分 (Eigenfaces)
    eigenfaces = pca.components_.T
    eigenface_titles = [f"eigenface {i}" for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)

    # 4. 将原始人脸投影到PCA子空间，并重建人脸
    X_transformed = pca.transform(X)
    X_reconstructed = np.dot(X_transformed, pca.components_.T) + pca.mean_

    # 5. 可视化原始人脸和重建后的人脸
    # 我们只展示前12张图片进行对比
    reconstruction_titles = ["original", "reconstructed"]
    n_images_to_show = 6
    
    fig, axes = plt.subplots(n_images_to_show, 2, figsize=(4, 8))
    fig.suptitle("Original vs Reconstructed Faces", fontsize=16)
    
    for i in range(n_images_to_show):
        # 绘制原始图像
        axes[i, 0].imshow(X[i].reshape(h, w), cmap=plt.cm.gray)
        axes[i, 0].set_title(f"Original ({target_names[y[i]]})")
        axes[i, 0].set_xticks(())
        axes[i, 0].set_yticks(())
        
        # 绘制重建图像
        axes[i, 1].imshow(X_reconstructed[i].reshape(h, w), cmap=plt.cm.gray)
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].set_xticks(())
        axes[i, 1].set_yticks(())

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    run_eigenfaces_analysis()
