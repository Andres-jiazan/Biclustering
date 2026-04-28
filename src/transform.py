"""
非线性最小二乘变换模块 (Nonlinear Least-squares Transform)

数学原理（见论文第4章）：

  在完成双向聚类与 Gamma 矫正后，训练集已得到重构矩阵 X̂（Gamma 变换后的版本）。

  本模块学习一个非线性映射 f，使得 f(x_i) ≈ x̂_i 对所有训练样本成立。

  === 两种参数化方式 ===

  方式 A：分段多项式（默认）
      f_{ab,j}(z; θ) = θ_0 + θ_1·z + θ_2·z²
      由于目标 x̂_ij = x_ij^γ 是 x_ij 的真正非线性函数，
      多项式在 [0,1] 上能很好地近似 Gamma 函数。

  方式 B：幂次变换（scipy.optimize.least_squares，真正的非线性参数）
      f_{ab,j}(z; α_j, γ_j) = α_j · z^γ_j
      直接拟合每个特征的 Gamma 值，与 Gamma 矫正理论完全对应。

  方式 C：精确 Gamma（推荐用于实际预处理，最准确）
      直接使用训练时学习到的 gamma_{ab} 值：x̂ = x^gamma_{ab}
      这是最准确的测试集预处理方法，也是论文中 BCNLT 的核心贡献。

  === 测试集预处理 ===
  1. 将测试样本分配到最近的样本簇 a*（用训练簇质心）
  2. 对每个特征簇 b，应用 gamma_{a*b} 做 Gamma 矫正
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Optional


# --------------------------------------------------------------------------- #
#  多项式特征展开
# --------------------------------------------------------------------------- #

def poly_features(z: np.ndarray, degree: int = 2) -> np.ndarray:
    """将向量 z 展开为多项式特征矩阵 [1, z, z², ...]。"""
    return np.column_stack([z ** k for k in range(degree + 1)])


# --------------------------------------------------------------------------- #
#  单个 (a, b) 组合的变换拟合
# --------------------------------------------------------------------------- #

def fit_block_transform(
    X_orig: np.ndarray,
    X_hat: np.ndarray,
    degree: int = 2,
    use_nonlinear: bool = False,
) -> np.ndarray:
    """
    对矩阵块拟合每维度多项式或幂次变换。

    Parameters
    ----------
    X_orig : ndarray, shape (n_a, d_b)  — 原始特征
    X_hat  : ndarray, shape (n_a, d_b)  — 目标（Gamma 矫正后）
    degree : int                         — 多项式次数（默认 2）
    use_nonlinear : bool
        False（默认）：多项式（θ_0 + θ_1·z + θ_2·z²），参数对 θ 线性
        True：幂次变换（α·z^γ），非线性参数，用 scipy.optimize.least_squares

    Returns
    -------
    params : ndarray
        use_nonlinear=False: shape (d_b, degree+1)，每行是多项式系数
        use_nonlinear=True:  shape (d_b, 2)，每行是 [α_j, γ_j]
    """
    n_a, d_b = X_orig.shape
    if n_a < degree + 1:
        degree = max(1, n_a - 1)

    if use_nonlinear:
        return _fit_power_transform(X_orig, X_hat)
    else:
        return _fit_poly_transform(X_orig, X_hat, degree)


def _fit_poly_transform(
    X_orig: np.ndarray,
    X_hat: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    最小二乘拟合逐特征多项式：min_{θ} ||Φ·θ - y||²
    Φ = [1, z, z², ...]（特征矩阵），θ 线性出现 → 直接 lstsq。
    """
    n_a, d_b = X_orig.shape
    params = np.zeros((d_b, degree + 1))

    for j in range(d_b):
        z = X_orig[:, j]
        y = X_hat[:, j]
        Phi = poly_features(z, degree)
        theta, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
        params[j] = theta

    return params


def _fit_power_transform(
    X_orig: np.ndarray,
    X_hat: np.ndarray,
) -> np.ndarray:
    """
    使用 scipy.optimize.least_squares 拟合幂次变换：
        f(z; α, γ) = α · (z + ε)^γ

    这是真正的"非线性最小二乘"—— α 和 γ 以非线性方式出现在残差中。
    由于目标 x̂ = x^gamma_{ab}，最优解接近 α=1, γ=gamma_{ab}。
    但逐特征拟合允许更精细的局部调整。
    """
    n_a, d_b = X_orig.shape
    params = np.zeros((d_b, 2))
    EPS = 1e-8

    for j in range(d_b):
        z = X_orig[:, j]
        y = X_hat[:, j]

        def residuals(theta):
            a_j, g_j = theta
            pred = a_j * np.power(z + EPS, np.abs(g_j))
            return pred - y

        x0 = np.array([1.0, 1.0])
        try:
            result = least_squares(
                residuals, x0,
                method="trf",
                bounds=([0.01, 0.1], [10.0, 5.0]),
                max_nfev=100,
                ftol=1e-4,
            )
            params[j] = result.x
        except Exception:
            params[j] = x0

    return params


# --------------------------------------------------------------------------- #
#  批量变换（向量化）
# --------------------------------------------------------------------------- #

def apply_block_transform_batch(
    X_block: np.ndarray,
    params: np.ndarray,
    use_nonlinear: bool = False,
    gamma_direct: Optional[float] = None,
) -> np.ndarray:
    """
    批量对矩阵块 X_block ∈ R^{m × d_b} 应用变换。

    Parameters
    ----------
    gamma_direct : float | None
        若提供，则直接用精确 Gamma 矫正（忽略 params）。
        这是测试集预处理的推荐方式。
    """
    m, d_b = X_block.shape
    EPS = 1e-8

    # 优先：精确 Gamma 矫正
    if gamma_direct is not None:
        return np.clip(np.power(X_block + EPS, gamma_direct), 0.0, 1.0)

    X_hat = np.zeros_like(X_block)

    if use_nonlinear:
        for j in range(d_b):
            a_j, g_j = params[j]
            X_hat[:, j] = a_j * np.power(X_block[:, j] + EPS, np.abs(g_j))
    else:
        degree = params.shape[1] - 1
        for j in range(d_b):
            theta = params[j]
            z = X_block[:, j]
            Phi = poly_features(z, degree)
            X_hat[:, j] = Phi @ theta

    return X_hat


# --------------------------------------------------------------------------- #
#  完整训练：学习所有 (a,b) 块的变换参数
# --------------------------------------------------------------------------- #

def fit_all_transforms(
    X_train: np.ndarray,
    X_hat_train: np.ndarray,
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    n_row_clusters: int,
    n_col_clusters: int,
    degree: int = 2,
    use_nonlinear: bool = False,
    block_params: Optional[dict] = None,
) -> dict:
    """
    对所有 (a, b) 块学习变换参数，同时存储精确 Gamma 值。

    Parameters
    ----------
    block_params : dict | None
        若提供 reconstruct_all_blocks 返回的 block_params，
        则同时存储精确 gamma 值（用于测试集直接应用）。

    Returns
    -------
    transform_params : dict {(a, b): {'poly'/'power': params_array, 'gamma': float}}
    """
    transform_params = {}

    for a in range(n_row_clusters):
        row_mask = row_labels == a
        row_idx = np.where(row_mask)[0]
        if len(row_idx) == 0:
            continue

        for b in range(n_col_clusters):
            col_mask = col_labels == b
            col_idx = np.where(col_mask)[0]
            if len(col_idx) == 0:
                continue

            X_orig_block = X_train[np.ix_(row_idx, col_idx)]
            X_hat_block  = X_hat_train[np.ix_(row_idx, col_idx)]

            params = fit_block_transform(
                X_orig_block, X_hat_block,
                degree=degree, use_nonlinear=use_nonlinear
            )

            entry = {"params": params, "use_nonlinear": use_nonlinear}

            # 存储精确 Gamma（若 block_params 提供）
            if block_params is not None and (a, b) in block_params:
                entry["gamma"] = block_params[(a, b)]["gamma"]
            else:
                entry["gamma"] = None

            transform_params[(a, b)] = entry

    return transform_params


def apply_all_transforms(
    X: np.ndarray,
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    transform_params: dict,
    n_row_clusters: int,
    n_col_clusters: int,
    use_direct_gamma: bool = True,
) -> np.ndarray:
    """
    将学习到的变换应用于数据集（训练集或测试集）。

    Parameters
    ----------
    use_direct_gamma : bool（默认 True）
        True：使用精确 Gamma 矫正（推荐，最准确）
        False：使用多项式/幂次拟合的近似变换

    Returns
    -------
    X_out : ndarray, shape (m, d)
    """
    m, d = X.shape
    X_out = np.copy(X)

    for a in range(n_row_clusters):
        row_mask = row_labels == a
        row_idx = np.where(row_mask)[0]
        if len(row_idx) == 0:
            continue

        for b in range(n_col_clusters):
            col_mask = col_labels == b
            col_idx = np.where(col_mask)[0]
            if len(col_idx) == 0:
                continue

            if (a, b) not in transform_params:
                continue

            entry = transform_params[(a, b)]
            X_block = X[np.ix_(row_idx, col_idx)]

            if use_direct_gamma and entry.get("gamma") is not None:
                # 精确 Gamma 矫正（首选）
                X_out[np.ix_(row_idx, col_idx)] = apply_block_transform_batch(
                    X_block, entry["params"],
                    gamma_direct=entry["gamma"]
                )
            else:
                # 多项式/幂次近似
                X_out[np.ix_(row_idx, col_idx)] = apply_block_transform_batch(
                    X_block, entry["params"],
                    use_nonlinear=entry.get("use_nonlinear", False)
                )

    return X_out
