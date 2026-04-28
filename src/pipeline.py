"""
BCNLT 完整流水线 (Full Pipeline)

BCNLTPreprocessor 提供 sklearn 风格的 API：
    preprocessor = BCNLTPreprocessor(n_row_clusters=4, n_col_clusters=4)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

内部流程：
    fit(X_train):
        1. 双向聚类 → row_labels, col_labels
        2. 获取矩阵块字典
        3. 对角矩阵优化与块重构 → X_hat_train
        4. 非线性变换学习 → transform_params

    transform(X):
        1. 分配测试样本到最近的行簇（用训练质心）
        2. 应用对应变换参数
        → X_transformed
"""

import numpy as np
from .biclustering import BidirectionalClustering
from .reconstruction import reconstruct_all_blocks, reconstruction_error
from .transform import fit_all_transforms, apply_all_transforms


class BCNLTPreprocessor:
    """
    基于双向聚类与非线性变换的高维数据预处理器。

    Parameters
    ----------
    n_row_clusters : int
        样本维聚类数 k_r（默认 4）
    n_col_clusters : int
        特征维聚类数 k_c（默认 4）
    cluster_method : str
        'spectral'（默认）或 'kmeans'
    poly_degree : int
        多项式变换次数（默认 2）
    use_nonlinear : bool
        False（默认）：使用多项式最小二乘（线性参数）
        True：使用幂次变换（非线性参数，scipy.optimize.least_squares）
    eps : float
        对角矩阵优化中的数值稳定性常数
    random_state : int
    verbose : bool
        是否打印进度信息
    """

    def __init__(
        self,
        n_row_clusters: int = 3,
        n_col_clusters: int = 3,
        cluster_method: str = "spectral",
        poly_degree: int = 2,
        use_nonlinear: bool = False,
        eps: float = 1e-8,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.cluster_method = cluster_method
        self.poly_degree = poly_degree
        self.use_nonlinear = use_nonlinear
        self.eps = eps
        self.random_state = random_state
        self.verbose = verbose

        # 拟合后填充
        self._clusterer: BidirectionalClustering = None
        self._block_params: dict = None
        self._transform_params: dict = None
        self._X_hat_train: np.ndarray = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------ #
    #  主要接口
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "BCNLTPreprocessor":
        """
        在训练集 X 上拟合 BCNLT 预处理器。

        Parameters
        ----------
        X : ndarray, shape (n, d)
        """
        self._log("Step 1/3: 双向聚类...")
        clusterer = BidirectionalClustering(
            n_row_clusters=self.n_row_clusters,
            n_col_clusters=self.n_col_clusters,
            method=self.cluster_method,
            random_state=self.random_state,
        )
        clusterer.fit(X)
        if self.verbose:
            clusterer.summary()

        blocks, row_idx_map, col_idx_map = clusterer.get_blocks(X)

        self._log("Step 2/3: 对角矩阵优化与块重构...")
        X_hat, block_params = reconstruct_all_blocks(
            X, blocks, row_idx_map, col_idx_map, eps=self.eps
        )
        err = reconstruction_error(X, X_hat)
        self._log(
            f"  重构相对误差: {err['relative']:.4f}  PSNR: {err['psnr']:.2f} dB"
        )

        self._log("Step 3/3: 非线性变换学习...")
        transform_params = fit_all_transforms(
            X_train=X,
            X_hat_train=X_hat,
            row_labels=clusterer.row_labels_,
            col_labels=clusterer.col_labels_,
            n_row_clusters=self.n_row_clusters,
            n_col_clusters=self.n_col_clusters,
            degree=self.poly_degree,
            use_nonlinear=self.use_nonlinear,
            block_params=block_params,   # 传入精确 gamma 值
        )

        # 保存状态
        self._clusterer = clusterer
        self._block_params = block_params
        self._transform_params = transform_params
        self._X_hat_train = X_hat
        self.is_fitted_ = True
        self._log("拟合完成。")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将已学习的变换应用于新数据（训练集或测试集）。

        Parameters
        ----------
        X : ndarray, shape (m, d)

        Returns
        -------
        X_transformed : ndarray, shape (m, d)
        """
        self._check_fitted()

        # 为测试样本分配最近的行簇
        row_labels = self._clusterer.assign_row_cluster(X)

        X_transformed = apply_all_transforms(
            X=X,
            row_labels=row_labels,
            col_labels=self._clusterer.col_labels_,
            transform_params=self._transform_params,
            n_row_clusters=self.n_row_clusters,
            n_col_clusters=self.n_col_clusters,
            use_direct_gamma=True,   # 默认使用精确 Gamma
        )
        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """在训练集上拟合并返回变换后的训练集。"""
        self.fit(X)
        # 训练集直接使用重构矩阵（不再经过 transform，避免二次误差积累）
        # 注意：也可以改为 self.transform(X) 以使训练/测试路径完全一致
        return self._X_hat_train

    def get_train_reconstruction(self) -> np.ndarray:
        """返回训练集重构结果（对角矩阵优化后，变换前）。"""
        self._check_fitted()
        return self._X_hat_train

    # ------------------------------------------------------------------ #
    #  辅助方法
    # ------------------------------------------------------------------ #

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError(
                "BCNLTPreprocessor 尚未拟合，请先调用 fit(X_train)。"
            )

    def _log(self, msg: str):
        if self.verbose:
            print(f"[BCNLT] {msg}")

    @property
    def row_labels_(self) -> np.ndarray:
        self._check_fitted()
        return self._clusterer.row_labels_

    @property
    def col_labels_(self) -> np.ndarray:
        self._check_fitted()
        return self._clusterer.col_labels_

    @property
    def row_centroids_(self) -> np.ndarray:
        self._check_fitted()
        return self._clusterer.row_centroids_
