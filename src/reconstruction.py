"""
对角矩阵优化与块重构模块 (Block Reconstruction)

数学原理（见论文第3章）：

  对每个矩阵块 X_{ab} ∈ R^{n_a × d_b}，构造左对角阵 L_a = diag(l_a) 和
  右对角阵 R_b = diag(r_b)，使得重构后的矩阵 X̂_{ab} 逐元素满足：

      X̂_{ab,ij} = l_{a,i} · x_{ab,ij} · r_{b,j}

  即 X̂_{ab} = diag(l_a) · X_{ab} · diag(r_b)。

  === 权重向量定义（块内双向 Gamma 归一化）===

  传统光照归一化存在跨样本比较困难的问题。本模块采用块级 Gamma 矫正：

  (1) 计算块级最优 Gamma：
      gamma_{ab} = log(target_mean) / log(mean(X_{ab}) + ε)
      使得 X_{ab}^{gamma} 的均值归一化到 target_mean（默认 0.5）

  (2) 逐元素 Gamma 变换：
      X̂_{ab} = X_{ab}^{gamma_{ab}}

  (3) 对角矩阵的物理解释：
      l_{a,i} = mean(X_{ab,i,:}^γ) / mean(X_{ab,i,:})   — 样本级 Gamma 效应
      r_{b,j} = mean(X_{ab,:,j}^γ) / mean(X_{ab,:,j})   — 特征级 Gamma 效应

  这两个向量形成了显式的左右对角矩阵，捕获了 Gamma 矫正对样本强度和
  特征强度的非线性效应。

  === 非线性最小二乘的意义 ===
  由于 x̂_{ij} = x_{ij}^γ 是 x_{ij} 的真正非线性函数，多项式拟合
      f_j(z) = θ_0 + θ_1·z + θ_2·z²  ≈  z^γ
  在 z ∈ [0,1] 上给出良好的 Gamma 函数近似。

  幂次拟合（scipy.optimize.least_squares）：
      f_j(z; α, γ) = α·z^γ  ←→ 直接求解最优 γ
"""

import numpy as np
from typing import Optional


# --------------------------------------------------------------------------- #
#  单个矩阵块的 Gamma 矫正与对角权重
# --------------------------------------------------------------------------- #

def optimize_diagonal(
    X_block: np.ndarray,
    target_mean: float = 0.5,
    gamma_min: float = 0.2,
    gamma_max: float = 5.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    对矩阵块 X_block 执行 Gamma 矫正，返回左右对角权重。

    Parameters
    ----------
    X_block     : ndarray, shape (n_a, d_b)
    target_mean : float  — 归一化目标均值（默认 0.5）
    gamma_min/max : float — Gamma 范围限制
    eps         : float

    Returns
    -------
    l_a   : ndarray, shape (n_a,) — 左对角权重（样本级 Gamma 效应）
    r_b   : ndarray, shape (d_b,) — 右对角权重（特征级 Gamma 效应）
    mu    : ndarray, shape (d_b,) — 块均值（参考用）
    gamma : float                 — 最优 Gamma 值
    """
    n_a, d_b = X_block.shape

    if n_a == 0 or d_b == 0:
        return np.ones(n_a), np.ones(d_b), np.zeros(d_b), 1.0

    # 块均值
    mu = X_block.mean(axis=0)

    # 计算最优 Gamma（使全局均值归一化到 target_mean）
    block_mean = X_block.mean()
    if block_mean < eps:
        gamma = 1.0
    else:
        gamma = np.log(target_mean) / np.log(block_mean + eps)
        gamma = float(np.clip(gamma, gamma_min, gamma_max))

    # Gamma 变换（元素级）
    X_gamma = np.power(X_block + eps, gamma)

    # 左权重：样本均值的 Gamma 效应比率
    row_mean_orig  = X_block.mean(axis=1)       # (n_a,)
    row_mean_gamma = X_gamma.mean(axis=1)       # (n_a,)
    l_a = row_mean_gamma / (row_mean_orig + eps)

    # 右权重：特征均值的 Gamma 效应比率
    col_mean_orig  = X_block.mean(axis=0)       # (d_b,)
    col_mean_gamma = X_gamma.mean(axis=0)       # (d_b,)
    r_b = col_mean_gamma / (col_mean_orig + eps)

    return l_a, r_b, mu, gamma


def reconstruct_block(
    X_block: np.ndarray,
    gamma: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    对矩阵块执行 Gamma 矫正，返回重构块。

    X̂_{ab,ij} = x_{ab,ij}^gamma  （元素级幂次变换）

    Parameters
    ----------
    X_block : ndarray, shape (n_a, d_b)
    gamma   : float  — Gamma 值（由 optimize_diagonal 返回）

    Returns
    -------
    X_hat : ndarray, shape (n_a, d_b)，值域 [0, 1]
    """
    X_hat = np.power(X_block + eps, gamma)
    X_hat = np.clip(X_hat, 0.0, 1.0)
    return X_hat


# --------------------------------------------------------------------------- #
#  全局重构：处理所有块并组装回完整矩阵
# --------------------------------------------------------------------------- #

def reconstruct_all_blocks(
    X: np.ndarray,
    blocks: dict,
    row_idx_map: dict,
    col_idx_map: dict,
    target_mean: float = 0.5,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """
    对所有矩阵块执行 Gamma 矫正与重构，组装完整重构矩阵 X̂。

    Parameters
    ----------
    X           : ndarray, shape (n, d)
    blocks      : dict {(a,b): X_ab}
    row_idx_map : dict {(a,b): row_indices}
    col_idx_map : dict {(a,b): col_indices}
    target_mean : float
    eps         : float

    Returns
    -------
    X_hat        : ndarray, shape (n, d)
    block_params : dict {(a,b): {'l','r','mu','gamma','row_idx','col_idx'}}
    """
    n, d = X.shape
    X_hat = np.zeros_like(X)
    block_params = {}

    for (a, b), X_block in blocks.items():
        row_idx = row_idx_map[(a, b)]
        col_idx = col_idx_map[(a, b)]

        if len(row_idx) == 0 or len(col_idx) == 0:
            continue

        l_a, r_b, mu, gamma = optimize_diagonal(
            X_block, target_mean=target_mean, eps=eps
        )
        X_block_hat = reconstruct_block(X_block, gamma, eps=eps)

        # 散布回原始位置
        X_hat[np.ix_(row_idx, col_idx)] = X_block_hat

        block_params[(a, b)] = {
            "l": l_a,
            "r": r_b,
            "mu": mu,
            "gamma": gamma,
            "row_idx": row_idx,
            "col_idx": col_idx,
        }

    return X_hat, block_params


# --------------------------------------------------------------------------- #
#  直接对新样本应用 Gamma 矫正（用于测试集）
# --------------------------------------------------------------------------- #

def apply_gamma_to_block(
    x_block: np.ndarray,
    gamma: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    对单个样本的特征子集（一个特征簇）应用已学习的 Gamma。

    Parameters
    ----------
    x_block : ndarray, shape (d_b,) 或 (m, d_b)
    gamma   : float

    Returns
    -------
    x_hat_block : 与输入同形状
    """
    return np.clip(np.power(x_block + eps, gamma), 0.0, 1.0)


# --------------------------------------------------------------------------- #
#  重构质量评估
# --------------------------------------------------------------------------- #

def reconstruction_error(X: np.ndarray, X_hat: np.ndarray) -> dict:
    """计算重构误差（Frobenius、相对误差、MSE、PSNR）。"""
    diff = X - X_hat
    frob = np.linalg.norm(diff, "fro")
    rel  = frob / (np.linalg.norm(X, "fro") + 1e-12)
    mse  = np.mean(diff ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-12)) if mse > 0 else float("inf")
    return {"frobenius": frob, "relative": rel, "mse": mse, "psnr": psnr}
