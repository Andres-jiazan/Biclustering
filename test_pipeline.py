"""
快速验证脚本：端到端测试 BCNLT 完整流水线
运行方式：python test_pipeline.py
"""
import sys
import time
sys.path.insert(0, '.')

import numpy as np

print("=" * 55)
print("  BCNLT 端到端验证")
print("=" * 55)

# ── 1. 数据加载 ──────────────────────────────────────────
print("\n[1/5] 加载 ORL 数据集...")
from data.load_dataset import load_orl, train_test_split_orl, dataset_info
X, y = load_orl('data/ORL')
dataset_info(X, y, 'ORL')
X_train, X_test, y_train, y_test = train_test_split_orl(X, y, n_train=5, random_state=0)
print(f"训练集: {X_train.shape}  测试集: {X_test.shape}")

# ── 2. 双向聚类 ──────────────────────────────────────────
print("\n[2/5] 双向聚类...")
from src.biclustering import BidirectionalClustering
bc = BidirectionalClustering(n_row_clusters=4, n_col_clusters=4, random_state=42)
bc.fit(X_train)
bc.summary()
blocks, row_idx_map, col_idx_map = bc.get_blocks(X_train)
print(f"  生成 {len(blocks)} 个矩阵块")

# ── 3. 对角矩阵优化与块重构 ──────────────────────────────
print("\n[3/5] 对角矩阵优化与块重构...")
from src.reconstruction import reconstruct_all_blocks, reconstruction_error
X_hat_train, block_params = reconstruct_all_blocks(
    X_train, blocks, row_idx_map, col_idx_map
)
err = reconstruction_error(X_train, X_hat_train)
print(f"  相对重构误差: {err['relative']:.4f}")
print(f"  PSNR: {err['psnr']:.2f} dB")
print(f"  Xhat range: [{X_hat_train.min():.3f}, {X_hat_train.max():.3f}]")
assert X_hat_train.shape == X_train.shape, "重构矩阵尺寸不匹配"

# ── 4. 非线性变换学习与应用 ──────────────────────────────
print("\n[4/5] 非线性变换学习...")
from src.transform import fit_all_transforms, apply_all_transforms
transform_params = fit_all_transforms(
    X_train, X_hat_train,
    bc.row_labels_, bc.col_labels_,
    4, 4, degree=2
)
print(f"  学习了 {len(transform_params)} 个块的变换参数")

test_row_labels = bc.assign_row_cluster(X_test)
X_test_proc = apply_all_transforms(
    X_test, test_row_labels, bc.col_labels_,
    transform_params, 4, 4
)
assert X_test_proc.shape == X_test.shape, "变换后测试集尺寸不匹配"
print(f"  测试集变换完成: {X_test_proc.shape}")

# ── 5. 完整流水线 + 分类评估 ────────────────────────────
print("\n[5/5] 完整流水线评估...")
from src.pipeline import BCNLTPreprocessor
from src.utils import evaluate_classifier, apply_pca

t0 = time.time()
prep = BCNLTPreprocessor(n_row_clusters=4, n_col_clusters=4, verbose=True)
X_tr_proc = prep.fit_transform(X_train)
X_te_proc = prep.transform(X_test)
t_total = time.time() - t0

# 分类对比
acc_raw_svm  = evaluate_classifier(X_train, y_train, X_test, y_test, 'svm')
acc_raw_knn  = evaluate_classifier(X_train, y_train, X_test, y_test, 'knn')
Xtr_pca, Xte_pca = apply_pca(X_train, X_test, n_components=80)
acc_pca_svm  = evaluate_classifier(Xtr_pca, y_train, Xte_pca, y_test, 'svm')
acc_pca_knn  = evaluate_classifier(Xtr_pca, y_train, Xte_pca, y_test, 'knn')
acc_bcnlt_svm = evaluate_classifier(X_tr_proc, y_train, X_te_proc, y_test, 'svm')
acc_bcnlt_knn = evaluate_classifier(X_tr_proc, y_train, X_te_proc, y_test, 'knn')

print("\n" + "=" * 55)
print("  识别率汇总 (ORL, 5-train/5-test per class)")
print("=" * 55)
print(f"{'方法':<15} {'SVM':>8} {'KNN':>8}")
print("-" * 35)
print(f"{'Raw':<15} {acc_raw_svm*100:>7.2f}% {acc_raw_knn*100:>7.2f}%")
print(f"{'PCA(80)':<15} {acc_pca_svm*100:>7.2f}% {acc_pca_knn*100:>7.2f}%")
print(f"{'BCNLT':<15} {acc_bcnlt_svm*100:>7.2f}% {acc_bcnlt_knn*100:>7.2f}%")
print("=" * 55)
print(f"  总耗时: {t_total:.2f}s")
print("\n所有测试通过!")
