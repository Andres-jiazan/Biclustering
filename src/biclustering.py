"""
双向聚类模块 (Bidirectional Clustering)

核心功能：
  - 对训练数据矩阵 X ∈ R^{n×d} 同步聚类行（样本）和列（特征）
  - 返回行标签 ρ 和列标签 γ，以及各子块数据字典

方法：
  SpectralBiclustering（谱双聚类）—— sklearn 实现
  可选：基于 K-Means 的迭代双向聚类（自定义）
"""

import numpy as np
from sklearn.cluster import SpectralBiclustering, KMeans
from sklearn.preprocessing import normalize
from typing import Optional


class BidirectionalClustering:
    """
    对矩阵 X 同步进行行聚类（样本维）和列聚类（特征维）。

    Parameters
    ----------
    n_row_clusters : int
        样本维聚类数 k_r
    n_col_clusters : int
        特征维聚类数 k_c
    method : str
        'spectral'（默认）或 'kmeans'
    random_state : int
    """

    def __init__(
        self,
        n_row_clusters: int = 4,
        n_col_clusters: int = 4,
        method: str = "spectral",
        random_state: int = 42,
    ):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.method = method
        self.random_state = random_state

        # 拟合后填充
        self.row_labels_: Optional[np.ndarray] = None   # shape (n,)
        self.col_labels_: Optional[np.ndarray] = None   # shape (d,)
        self.row_centroids_: Optional[np.ndarray] = None  # shape (k_r, d)

    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "BidirectionalClustering":
        """
        对 X ∈ R^{n×d} 执行双向聚类。

        Parameters
        ----------
        X : ndarray, shape (n, d)
        """
        if self.method == "spectral":
            self._fit_spectral(X)
        elif self.method == "kmeans":
            self._fit_kmeans(X)
        else:
            raise ValueError(f"Unknown method: {self.method!r}. Use 'spectral' or 'kmeans'.")

        # 计算每个行簇的质心（在原始特征空间中）
        self.row_centroids_ = np.zeros((self.n_row_clusters, X.shape[1]))
        for a in range(self.n_row_clusters):
            mask = self.row_labels_ == a
            if mask.sum() > 0:
                self.row_centroids_[a] = X[mask].mean(axis=0)

        return self

    # ------------------------------------------------------------------ #

    def _fit_spectral(self, X: np.ndarray) -> None:
        """谱双聚类（sklearn SpectralBiclustering）。"""
        model = SpectralBiclustering(
            n_clusters=(self.n_row_clusters, self.n_col_clusters),
            method="bistochastic",
            n_components=max(self.n_row_clusters, self.n_col_clusters) + 2,
            random_state=self.random_state,
        )
        # SpectralBiclustering 需要正值矩阵，做最小平移
        X_pos = X - X.min() + 1e-6
        model.fit(X_pos)

        # sklearn 直接提供 row_labels_ / column_labels_（0..k_r-1 / 0..k_c-1）
        self.row_labels_ = model.row_labels_.astype(int)
        self.col_labels_ = model.column_labels_.astype(int)

    def _fit_kmeans(self, X: np.ndarray) -> None:
        """
        基于 K-Means 的迭代双向聚类。
          - 行聚类：对样本向量做 KMeans
          - 列聚类：对特征向量（转置）做 KMeans
        """
        # 行聚类（样本）
        km_row = KMeans(
            n_clusters=self.n_row_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.row_labels_ = km_row.fit_predict(X)

        # 列聚类（特征——对 X^T 聚类）
        km_col = KMeans(
            n_clusters=self.n_col_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.col_labels_ = km_col.fit_predict(X.T)

    @staticmethod
    def _bicluster_to_labels(
        membership: np.ndarray, n: int, n_clusters: int
    ) -> np.ndarray:
        """
        将 SpectralBiclustering 的 rows_/columns_ 布尔矩阵
        （shape: n_biclusters × n）转换为唯一整数标签向量（shape: n,）。

        sklearn 的 SpectralBiclustering(n_clusters=(k_r,k_c)) 生成
        k_r*k_c 个 bicluster，前 k_r 行对应行聚类，后 k_c 行对应列聚类。
        直接读取 row_labels_ / column_labels_ 属性更简洁。
        """
        # SpectralBiclustering 内部存储了 row_labels_ 和 column_labels_
        # 但这里传入的是 model.rows_ / model.columns_，需要做如下转换：
        # 将每个样本分配到它隶属度最多的那个"行 bicluster 组"
        #
        # 更直接：SpectralBiclustering 设置 n_clusters=(k_r,k_c) 时
        # model.row_labels_ 直接就是 0..k_r-1 的标签向量
        # 所以这个方法备而不用，见 _fit_spectral 里的改进。
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            row_memberships = membership[:, i]
            if row_memberships.any():
                labels[i] = int(np.argmax(row_memberships) % n_clusters)
        return labels

    # ------------------------------------------------------------------ #

    def get_blocks(self, X: np.ndarray) -> dict:
        """
        根据行/列标签将 X 切分为子块字典。

        Returns
        -------
        blocks : dict { (a, b): X_ab }
            X_ab ∈ R^{n_a × d_b}，保留原始行/列索引信息。
        block_row_idx : dict { (a, b): row_indices }
        block_col_idx : dict { (a, b): col_indices }
        """
        self._check_fitted()
        blocks = {}
        row_idx_map = {}
        col_idx_map = {}

        for a in range(self.n_row_clusters):
            row_mask = self.row_labels_ == a
            row_idx = np.where(row_mask)[0]
            for b in range(self.n_col_clusters):
                col_mask = self.col_labels_ == b
                col_idx = np.where(col_mask)[0]
                block = X[np.ix_(row_mask, col_mask)]
                blocks[(a, b)] = block
                row_idx_map[(a, b)] = row_idx
                col_idx_map[(a, b)] = col_idx

        return blocks, row_idx_map, col_idx_map

    def assign_row_cluster(self, X: np.ndarray) -> np.ndarray:
        """
        将新样本分配到最近的训练行簇质心。

        Parameters
        ----------
        X : ndarray, shape (m, d)

        Returns
        -------
        labels : ndarray, shape (m,), 值域 0..k_r-1
        """
        self._check_fitted()
        # 欧氏距离到各质心
        dists = np.linalg.norm(
            X[:, np.newaxis, :] - self.row_centroids_[np.newaxis, :, :],
            axis=2
        )  # (m, k_r)
        return np.argmin(dists, axis=1)

    # ------------------------------------------------------------------ #

    def _check_fitted(self):
        if self.row_labels_ is None or self.col_labels_ is None:
            raise RuntimeError("BidirectionalClustering has not been fitted yet.")

    def summary(self):
        """打印聚类结果统计。"""
        self._check_fitted()
        print(f"双向聚类结果 (method={self.method})")
        print(f"  行聚类数 k_r = {self.n_row_clusters}")
        for a in range(self.n_row_clusters):
            cnt = (self.row_labels_ == a).sum()
            print(f"    簇 {a}: {cnt} 个样本")
        print(f"  列聚类数 k_c = {self.n_col_clusters}")
        for b in range(self.n_col_clusters):
            cnt = (self.col_labels_ == b).sum()
            print(f"    簇 {b}: {cnt} 个特征")
