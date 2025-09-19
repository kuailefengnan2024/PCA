import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        初始化PCA模型
        :param n_components: int, 需要保留的主成分数量
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        """
        根据输入数据X计算主成分
        :param X: array-like, shape (n_samples, n_features)
        """
        # 1. 数据中心化 (Mean centering)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. 计算协方差矩阵 (Covariance matrix)
        # (n_features, n_samples) @ (n_samples, n_features) -> (n_features, n_features)
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. 对特征值进行降序排序，并获取对应特征向量的索引
        sorted_indices = np.argsort(eigenvalues)[::-1]
        
        # 5. 根据排序后的索引，重新排列特征向量
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # 6. 选取前 n_components 个特征向量作为主成分
        self.components_ = sorted_eigenvectors[:, :self.n_components]

        return self

    def transform(self, X):
        """
        将数据X投影到计算好的主成分上
        :param X: array-like, shape (n_samples, n_features)
        :return: array-like, shape (n_samples, n_components)
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA has not been fitted yet. Call fit() first.")
        
        # 中心化数据
        X_centered = X - self.mean_
        
        # 投影到主成分
        # (n_samples, n_features) @ (n_features, n_components) -> (n_samples, n_components)
        X_transformed = np.dot(X_centered, self.components_)
        
        return X_transformed

    def fit_transform(self, X):
        """
        计算主成分并对数据X进行转换
        """
        self.fit(X)
        return self.transform(X)
