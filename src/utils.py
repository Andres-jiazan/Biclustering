"""
工具函数：评估、可视化、基线方法

包含：
  - 分类实验（SVM / KNN）
  - 基线预处理方法（PCA, LDA, NMF）
  - 可视化工具（脸图展示、双聚类热力图、结果对比柱状图）
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 无头环境兼容；Notebook 中会被 %matplotlib inline 覆盖
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from typing import Optional


# ============================================================ #
#  评估框架
# ============================================================ #

def evaluate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clf_type: str = "svm",
) -> float:
    """
    在预处理后的特征上训练并评估分类器。

    Parameters
    ----------
    clf_type : str  'svm' 或 'knn'

    Returns
    -------
    accuracy : float
    """
    if clf_type == "svm":
        clf = SVC(kernel="rbf", C=10.0, gamma="scale", decision_function_shape="ovr")
    elif clf_type == "knn":
        clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    else:
        raise ValueError(f"Unknown clf_type: {clf_type!r}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    methods: dict,
    clf_types: list = ("svm", "knn"),
    n_repeats: int = 1,
    verbose: bool = True,
) -> dict:
    """
    对多种预处理方法批量评估。

    Parameters
    ----------
    methods : dict { name: (X_train_proc, X_test_proc) }
        每个方法预处理后的训练/测试特征对。

    Returns
    -------
    results : dict { name: { clf_type: accuracy } }
    """
    results = {}
    for name, (Xtr, Xte) in methods.items():
        results[name] = {}
        for clf_type in clf_types:
            acc = evaluate_classifier(Xtr, y_train, Xte, y_test, clf_type)
            results[name][clf_type] = acc
            if verbose:
                print(f"  [{name}] {clf_type.upper()}: {acc * 100:.2f}%")
    return results


# ============================================================ #
#  基线预处理方法
# ============================================================ #

def apply_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 80,
    whiten: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA 降维。"""
    pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
    Xtr = pca.fit_transform(X_train)
    Xte = pca.transform(X_test)
    return Xtr, Xte


def apply_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """LDA 降维（监督）。"""
    # LDA 输出维度最多为 n_classes - 1
    max_comp = len(np.unique(y_train)) - 1
    n_comp = min(n_components or max_comp, max_comp)
    # 先 PCA 降到足够小再 LDA，避免奇异
    pca = PCA(n_components=min(n_comp * 4, X_train.shape[1], X_train.shape[0] - 1))
    Xtr_pca = pca.fit_transform(X_train)
    Xte_pca = pca.transform(X_test)
    lda = LDA(n_components=n_comp)
    Xtr = lda.fit_transform(Xtr_pca, y_train)
    Xte = lda.transform(Xte_pca)
    return Xtr, Xte


def apply_nmf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """NMF 分解（要求非负输入）。"""
    X_tr_pos = np.clip(X_train, 0, None)
    X_te_pos = np.clip(X_test, 0, None)
    nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
    Xtr = nmf.fit_transform(X_tr_pos)
    Xte = nmf.transform(X_te_pos)
    return Xtr, Xte


def prepare_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 80,
) -> dict:
    """
    生成所有基线方法的预处理结果。

    Returns
    -------
    methods : dict { name: (X_train_proc, X_test_proc) }
    """
    methods = {}
    # Raw（无预处理）
    methods["Raw"] = (X_train, X_test)
    # PCA
    methods["PCA"] = apply_pca(X_train, X_test, n_components=n_components)
    # LDA（监督方法，仅作参考）
    try:
        methods["LDA"] = apply_lda(X_train, y_train, X_test)
    except Exception as e:
        print(f"  LDA 跳过: {e}")
    # NMF
    try:
        methods["NMF"] = apply_nmf(X_train, X_test, n_components=n_components)
    except Exception as e:
        print(f"  NMF 跳过: {e}")
    return methods


# ============================================================ #
#  可视化工具
# ============================================================ #

def show_faces(
    X: np.ndarray,
    y: np.ndarray,
    img_shape: tuple = (112, 92),
    n_subjects: int = 5,
    n_per_subject: int = 3,
    title: str = "Face Samples",
    save_path: Optional[str] = None,
):
    """
    展示部分人脸图像。

    Parameters
    ----------
    img_shape : (H, W)
    """
    fig, axes = plt.subplots(n_subjects, n_per_subject, figsize=(n_per_subject * 1.5, n_subjects * 1.8))
    classes = np.unique(y)[:n_subjects]
    for row, cls in enumerate(classes):
        idx = np.where(y == cls)[0][:n_per_subject]
        for col, i in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(X[i].reshape(img_shape), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"S{cls+1}", fontsize=8)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def show_reconstruction_comparison(
    X_orig: np.ndarray,
    X_hat: np.ndarray,
    y: np.ndarray,
    img_shape: tuple = (112, 92),
    n_samples: int = 5,
    title: str = "Original vs Reconstructed",
    save_path: Optional[str] = None,
):
    """展示原始图像与重构图像对比。"""
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.8, 4))
    idx = np.arange(n_samples)
    for col, i in enumerate(idx):
        axes[0, col].imshow(X_orig[i].reshape(img_shape), cmap="gray", vmin=0, vmax=1)
        axes[0, col].axis("off")
        axes[0, col].set_title(f"S{y[i]+1}", fontsize=8)

        axes[1, col].imshow(X_hat[i].reshape(img_shape), cmap="gray", vmin=0, vmax=1)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def show_bicluster_heatmap(
    X: np.ndarray,
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    title: str = "Biclustered Data (Checkerboard)",
    subsample: int = 200,
    save_path: Optional[str] = None,
):
    """
    可视化双聚类重排后的矩阵热力图（Checkerboard 结构）。

    为避免内存问题，仅展示前 subsample 列特征。
    """
    row_order = np.argsort(row_labels)
    col_order = np.argsort(col_labels)

    X_sorted = X[np.ix_(row_order, col_order[:subsample])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 原始矩阵（随机顺序）
    sns.heatmap(X[:, :subsample], ax=axes[0], cmap="viridis",
                xticklabels=False, yticklabels=False, cbar=False)
    axes[0].set_title("Original Order")
    axes[0].set_xlabel("Features (first 200)")
    axes[0].set_ylabel("Samples")

    # 重排后的矩阵
    sns.heatmap(X_sorted, ax=axes[1], cmap="viridis",
                xticklabels=False, yticklabels=False, cbar=True)
    axes[1].set_title("After Biclustering (sorted)")
    axes[1].set_xlabel("Features (sorted by cluster)")
    axes[1].set_ylabel("Samples (sorted by cluster)")

    # 绘制簇边界线
    row_boundaries = np.where(np.diff(row_labels[row_order]))[0] + 1
    for rb in row_boundaries:
        axes[1].axhline(rb, color="red", linewidth=1.0, linestyle="--")

    col_sub = col_labels[col_order[:subsample]]
    col_boundaries = np.where(np.diff(col_sub))[0] + 1
    for cb in col_boundaries:
        axes[1].axvline(cb, color="red", linewidth=1.0, linestyle="--")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_accuracy_comparison(
    results: dict,
    clf_type: str = "svm",
    title: str = "Recognition Accuracy Comparison",
    save_path: Optional[str] = None,
):
    """
    绘制各方法识别率柱状图。

    Parameters
    ----------
    results : dict { method_name: { clf_type: accuracy } }
    """
    methods = list(results.keys())
    accs = [results[m].get(clf_type, 0) * 100 for m in methods]

    colors = ["#4c72b0"] * len(methods)
    # 高亮 BCNLT
    if "BCNLT" in methods:
        colors[methods.index("BCNLT")] = "#dd8452"

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.4), 5))
    bars = ax.bar(methods, accs, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"{title} ({clf_type.upper()})", fontsize=12)
    ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=9
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_transform_fit(
    X_orig_block: np.ndarray,
    X_hat_block: np.ndarray,
    X_transformed_block: np.ndarray,
    feature_idx: int = 0,
    title: str = "Nonlinear Transform Fitting",
    save_path: Optional[str] = None,
):
    """
    可视化单个特征维度上的变换拟合效果（散点图）。
    """
    z = X_orig_block[:, feature_idx]
    y_target = X_hat_block[:, feature_idx]
    y_fitted = X_transformed_block[:, feature_idx]

    sort_idx = np.argsort(z)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(z, y_target, alpha=0.5, s=20, color="#4c72b0", label="Target (reconstructed)")
    ax.plot(z[sort_idx], y_fitted[sort_idx], color="#dd8452", linewidth=2, label="Fitted transform")
    ax.set_xlabel("Original pixel value", fontsize=10)
    ax.set_ylabel("Target pixel value", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================ #
#  超参数搜索
# ============================================================ #

def grid_search_bcnlt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kr_range: list = (2, 3, 4, 5),
    kc_range: list = (2, 3, 4, 5),
    clf_type: str = "svm",
    verbose: bool = True,
) -> tuple[int, int, float, np.ndarray]:
    """
    网格搜索最优的 (k_r, k_c) 超参数组合。

    Returns
    -------
    best_kr, best_kc, best_acc, acc_matrix
    """
    from .pipeline import BCNLTPreprocessor

    acc_matrix = np.zeros((len(kr_range), len(kc_range)))
    best_acc = 0.0
    best_kr, best_kc = kr_range[0], kc_range[0]

    for i, kr in enumerate(kr_range):
        for j, kc in enumerate(kc_range):
            try:
                prep = BCNLTPreprocessor(
                    n_row_clusters=kr, n_col_clusters=kc, verbose=False
                )
                Xtr = prep.fit_transform(X_train)
                Xte = prep.transform(X_test)
                acc = evaluate_classifier(Xtr, y_train, Xte, y_test, clf_type)
                acc_matrix[i, j] = acc
                if acc > best_acc:
                    best_acc = acc
                    best_kr, best_kc = kr, kc
                if verbose:
                    print(f"  k_r={kr}, k_c={kc}: {acc*100:.2f}%")
            except Exception as e:
                if verbose:
                    print(f"  k_r={kr}, k_c={kc}: FAILED ({e})")

    return best_kr, best_kc, best_acc, acc_matrix


def plot_hyperparam_heatmap(
    acc_matrix: np.ndarray,
    kr_range: list,
    kc_range: list,
    title: str = "BCNLT Accuracy vs (k_r, k_c)",
    save_path: Optional[str] = None,
):
    """绘制超参数搜索热力图。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        acc_matrix * 100,
        annot=True, fmt=".1f",
        xticklabels=kc_range,
        yticklabels=kr_range,
        cmap="YlOrRd",
        ax=ax,
    )
    ax.set_xlabel("n_col_clusters (k_c)", fontsize=10)
    ax.set_ylabel("n_row_clusters (k_r)", fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
