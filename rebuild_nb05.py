import json

def code_cell(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}

def md_cell(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [src]}

c0 = """import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import csv
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 100
from data.load_dataset import load_orl, train_test_split_orl
from src.pipeline import BCNLTPreprocessor
from src.utils import evaluate_classifier, apply_pca, apply_lda, apply_nmf, grid_search_bcnlt
X, y = load_orl(data_dir='../data/ORL')
print(f'Dataset: {X.shape}, classes={len(np.unique(y))}')"""

c1_trial = """N_TRIALS = 10
results = {m: {'svm':[], 'knn':[]} for m in ['Raw','PCA80','LDA','NMF','BCNLT']}
for trial in range(N_TRIALS):
    X_tr, X_te, y_tr, y_te = train_test_split_orl(X, y, n_train=5, random_state=trial)
    results['Raw']['svm'].append(evaluate_classifier(X_tr, y_tr, X_te, y_te, 'svm'))
    results['Raw']['knn'].append(evaluate_classifier(X_tr, y_tr, X_te, y_te, 'knn'))
    Xtr_p, Xte_p = apply_pca(X_tr, X_te, n_components=80)
    results['PCA80']['svm'].append(evaluate_classifier(Xtr_p, y_tr, Xte_p, y_te, 'svm'))
    results['PCA80']['knn'].append(evaluate_classifier(Xtr_p, y_tr, Xte_p, y_te, 'knn'))
    Xtr_l, Xte_l = apply_lda(X_tr, y_tr, X_te)
    results['LDA']['svm'].append(evaluate_classifier(Xtr_l, y_tr, Xte_l, y_te, 'svm'))
    results['LDA']['knn'].append(evaluate_classifier(Xtr_l, y_tr, Xte_l, y_te, 'knn'))
    Xtr_n, Xte_n = apply_nmf(X_tr, X_te, n_components=80)
    results['NMF']['svm'].append(evaluate_classifier(Xtr_n, y_tr, Xte_n, y_te, 'svm'))
    results['NMF']['knn'].append(evaluate_classifier(Xtr_n, y_tr, Xte_n, y_te, 'knn'))
    prep = BCNLTPreprocessor(n_row_clusters=3, n_col_clusters=3, random_state=trial)
    Xtr_b = prep.fit_transform(X_tr)
    Xte_b = prep.transform(X_te)
    results['BCNLT']['svm'].append(evaluate_classifier(Xtr_b, y_tr, Xte_b, y_te, 'svm'))
    results['BCNLT']['knn'].append(evaluate_classifier(Xtr_b, y_tr, Xte_b, y_te, 'knn'))
    if (trial+1) % 2 == 0:
        print(f'Trial {trial+1}/{N_TRIALS}')
print('Done')"""

c2_print = """print(f"{'Method':<12} {'SVM Acc':>20} {'KNN Acc':>20}")
print('-' * 55)
for method, data in results.items():
    sa = np.array(data['svm']) * 100
    ka = np.array(data['knn']) * 100
    print(f"{method:<12} {sa.mean():.2f} +/- {sa.std():.2f}%   {ka.mean():.2f} +/- {ka.std():.2f}%")
with open('../results/tables/comparison_10trials.csv', 'w', newline='', encoding='utf-8') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(['Method', 'SVM_mean', 'SVM_std', 'KNN_mean', 'KNN_std'])
    for method, data in results.items():
        sa = np.array(data['svm']) * 100
        ka = np.array(data['knn']) * 100
        writer.writerow([method, f'{sa.mean():.2f}', f'{sa.std():.2f}', f'{ka.mean():.2f}', f'{ka.std():.2f}'])
print('Saved CSV')"""

c3_bar = """methods = list(results.keys())
svm_means = [np.array(results[m]['svm']).mean()*100 for m in methods]
knn_means = [np.array(results[m]['knn']).mean()*100 for m in methods]
svm_stds  = [np.array(results[m]['svm']).std()*100 for m in methods]
knn_stds  = [np.array(results[m]['knn']).std()*100 for m in methods]
x = np.arange(len(methods))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, svm_means, width, yerr=svm_stds, capsize=4, label='SVM', color='#4c72b0', alpha=0.85)
bars2 = ax.bar(x + width/2, knn_means, width, yerr=knn_stds, capsize=4, label='KNN', color='#dd8452', alpha=0.85)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('ORL Face Recognition: 10-Trial Comparison (5 train/5 test)', fontsize=11)
ax.set_ylim(0, 115)
ax.legend()
plt.tight_layout()
plt.savefig('../results/figures/05_comparison_10trials.png', dpi=150, bbox_inches='tight')
plt.show()"""

c4_ablation = """abl = {f'k={k}x{k}': {'svm':[], 'knn':[]} for k in [1,2,3,4,5]}
abl['Raw'] = {'svm':[], 'knn':[]}
for trial in range(5):
    X_tr, X_te, y_tr, y_te = train_test_split_orl(X, y, n_train=5, random_state=trial)
    abl['Raw']['svm'].append(evaluate_classifier(X_tr, y_tr, X_te, y_te, 'svm'))
    abl['Raw']['knn'].append(evaluate_classifier(X_tr, y_tr, X_te, y_te, 'knn'))
    for k in [1, 2, 3, 4, 5]:
        key = f'k={k}x{k}'
        prep = BCNLTPreprocessor(n_row_clusters=k, n_col_clusters=k, random_state=trial)
        Xtr_b = prep.fit_transform(X_tr)
        Xte_b = prep.transform(X_te)
        abl[key]['svm'].append(evaluate_classifier(Xtr_b, y_tr, Xte_b, y_te, 'svm'))
        abl[key]['knn'].append(evaluate_classifier(Xtr_b, y_tr, Xte_b, y_te, 'knn'))
print('Ablation (5 trials):')
for method, data in abl.items():
    sa = np.array(data['svm'])*100
    ka = np.array(data['knn'])*100
    print(f"  {method:<10}: SVM={sa.mean():.2f}+/-{sa.std():.2f}  KNN={ka.mean():.2f}+/-{ka.std():.2f}")"""

c5_grid = """X_tr0, X_te0, y_tr0, y_te0 = train_test_split_orl(X, y, n_train=5, random_state=0)
grid_results = grid_search_bcnlt(X_tr0, y_tr0, X_te0, y_te0, k_row_list=[2,3,4,5], k_col_list=[2,3,4,5])
print(f"{'k_r':>4} {'k_c':>4} {'SVM':>8} {'KNN':>8}")
print('-'*30)
for entry in sorted(grid_results, key=lambda x: -x['knn'])[:8]:
    print(f"{entry['k_row']:>4} {entry['k_col']:>4} {entry['svm']*100:>7.2f}% {entry['knn']*100:>7.2f}%")"""

c6_heatmap = """k_vals = [2,3,4,5]
knn_mat = np.zeros((len(k_vals), len(k_vals)))
for entry in grid_results:
    i = k_vals.index(entry['k_row'])
    j = k_vals.index(entry['k_col'])
    knn_mat[i,j] = entry['knn']*100
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(knn_mat, cmap='YlOrRd', vmin=knn_mat.min()-1, vmax=knn_mat.max()+1)
plt.colorbar(im, ax=ax, label='KNN Accuracy (%)')
ax.set_xticks(range(len(k_vals)))
ax.set_yticks(range(len(k_vals)))
ax.set_xticklabels(k_vals)
ax.set_yticklabels(k_vals)
ax.set_xlabel('k_col')
ax.set_ylabel('k_row')
ax.set_title('KNN Accuracy vs Cluster Count (k_r x k_c)')
for i in range(len(k_vals)):
    for j in range(len(k_vals)):
        ax.text(j, i, f'{knn_mat[i,j]:.1f}', ha='center', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('../results/figures/05_hyperparam_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()"""

cells = [
    md_cell('# 05 - Comparison Experiments\n\n1. 10-trial comparison\n2. Ablation by k\n3. Hyperparameter heatmap'),
    code_cell(c0),
    md_cell('## 1. 10-Trial Comparison (Raw, PCA, LDA, NMF, BCNLT)'),
    code_cell(c1_trial),
    code_cell(c2_print),
    code_cell(c3_bar),
    md_cell('## 2. Ablation: Effect of Cluster Count k'),
    code_cell(c4_ablation),
    md_cell('## 3. Hyperparameter Grid Search (k_r x k_c)'),
    code_cell(c5_grid),
    code_cell(c6_heatmap),
]

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python (bcnlt)', 'language': 'python', 'name': 'bcnlt'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('notebooks/05_experiments.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Notebook 05 OK')
