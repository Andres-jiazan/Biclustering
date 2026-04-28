import json

def code_cell(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}

def md_cell(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [src]}

c0 = ("import sys\n"
      "sys.path.insert(0, '..')\n"
      "import numpy as np\n"
      "import matplotlib.pyplot as plt\n"
      "import csv\n"
      "import warnings\n"
      "warnings.filterwarnings('ignore')\n"
      "get_ipython().run_line_magic('matplotlib', 'inline')\n"
      "plt.rcParams['figure.dpi'] = 100\n"
      "from data.load_dataset import load_orl, train_test_split_orl\n"
      "from src.pipeline import BCNLTPreprocessor\n"
      "from src.utils import evaluate_classifier, apply_pca, apply_lda, apply_nmf, grid_search_bcnlt\n"
      "X, y = load_orl(data_dir='../data/ORL')\n"
      "print(f'Dataset: {X.shape}, classes={len(np.unique(y))}')")

c1_trial = ("N_TRIALS = 10\n"
            "results = {m: {'svm':[], 'knn':[]} for m in ['Raw','PCA80','LDA','NMF','BCNLT']}\n"
            "for trial in range(N_TRIALS):\n"
            "    X_tr, X_te, y_tr, y_te = train_test_split_orl(X, y, n_train=5, random_state=trial)\n"
            "    results['Raw']['svm'].append(evaluate_classifier(X_tr, y_tr, X_te, y_te, 'svm'))\n"
            "    results['Raw']['knn'].append(evaluate_classifier(X_tr, y_tr, X_te, y_te, 'knn'))\n"
            "    Xtr_p, Xte_p = apply_pca(X_tr, X_te, n_components=80)\n"
            "    results['PCA80']['svm'].append(evaluate_classifier(Xtr_p, y_tr, Xte_p, y_te, 'svm'))\n"
            "    results['PCA80']['knn'].append(evaluate_classifier(Xtr_p, y_tr, Xte_p, y_te, 'knn'))\n"
            "    Xtr_l, Xte_l = apply_lda(X_tr, y_tr, X_te)\n"
            "    results['LDA']['svm'].append(evaluate_classifier(Xtr_l, y_tr, Xte_l, y_te, 'svm'))\n"
            "    results['LDA']['knn'].append(evaluate_classifier(Xtr_l, y_tr, Xte_l, y_te, 'knn'))\n"
            "    Xtr_n, Xte_n = apply_nmf(X_tr, X_te, n_components=80)\n"
            "    results['NMF']['svm'].append(evaluate_classifier(Xtr_n, y_tr, Xte_n, y_te, 'svm'))\n"
            "    results['NMF']['knn'].append(evaluate_classifier(Xtr_n, y_tr, Xte_n, y_te, 'knn'))\n"
            "    prep = BCNLTPreprocessor(n_row_clusters=3, n_col_clusters=3, random_state=trial)\n"
            "    Xtr_b = prep.fit_transform(X_tr)\n"
            "    Xte_b = prep.transform(X_te)\n"
            "    results['BCNLT']['svm'].append(evaluate_classifier(Xtr_b, y_tr, Xte_b, y_te, 'svm'))\n"
            "    results['BCNLT']['knn'].append(evaluate_classifier(Xtr_b, y_tr, Xte_b, y_te, 'knn'))\n"
            "    if (trial+1) % 2 == 0:\n"
            "        print(f'Trial {trial+1}/{N_TRIALS}')\n"
            "print('Done')")

c2_print = ("print('Method         SVM Acc              KNN Acc')\n"
            "print('-' * 55)\n"
            "for method, data in results.items():\n"
            "    sa = np.array(data['svm']) * 100\n"
            "    ka = np.array(data['knn']) * 100\n"
            "    print(f'{method:<13} {sa.mean():.2f} +/- {sa.std():.2f}%   {ka.mean():.2f} +/- {ka.std():.2f}%')\n"
            "with open('../results/tables/comparison_10trials.csv', 'w', newline='', encoding='utf-8') as csvf:\n"
            "    writer = csv.writer(csvf)\n"
            "    writer.writerow(['Method', 'SVM_mean', 'SVM_std', 'KNN_mean', 'KNN_std'])\n"
            "    for method, data in results.items():\n"
            "        sa = np.array(data['svm']) * 100\n"
            "        ka = np.array(data['knn']) * 100\n"
            "        writer.writerow([method, f'{sa.mean():.2f}', f'{sa.std():.2f}', f'{ka.mean():.2f}', f'{ka.std():.2f}'])\n"
            "print('Saved CSV')")

c3_bar = ("methods = list(results.keys())\n"
          "svm_means = [np.array(results[m]['svm']).mean()*100 for m in methods]\n"
          "knn_means = [np.array(results[m]['knn']).mean()*100 for m in methods]\n"
          "svm_stds  = [np.array(results[m]['svm']).std()*100 for m in methods]\n"
          "knn_stds  = [np.array(results[m]['knn']).std()*100 for m in methods]\n"
          "x = np.arange(len(methods))\n"
          "width = 0.35\n"
          "fig, ax = plt.subplots(figsize=(9, 5))\n"
          "bars1 = ax.bar(x - width/2, svm_means, width, yerr=svm_stds, capsize=4, label='SVM', color='#4c72b0', alpha=0.85)\n"
          "bars2 = ax.bar(x + width/2, knn_means, width, yerr=knn_stds, capsize=4, label='KNN', color='#dd8452', alpha=0.85)\n"
          "for bar in list(bars1) + list(bars2):\n"
          "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,\n"
          "            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)\n"
          "ax.set_xticks(x)\n"
          "ax.set_xticklabels(methods, fontsize=11)\n"
          "ax.set_ylabel('Accuracy (%)', fontsize=11)\n"
          "ax.set_title('ORL Face Recognition: 10-Trial Comparison (5 train/5 test)', fontsize=11)\n"
          "ax.set_ylim(0, 115)\n"
          "ax.legend()\n"
          "plt.tight_layout()\n"
          "plt.savefig('../results/figures/05_comparison_10trials.png', dpi=150, bbox_inches='tight')\n"
          "plt.show()")

c4_ablation = ("abl_k = {f'k={k}x{k}': {'svm':[], 'knn':[]} for k in [1,2,3,4,5]}\n"
               "abl_k['Raw'] = {'svm':[], 'knn':[]}\n"
               "for trial in range(5):\n"
               "    X_tr, X_te, y_tr, y_te = train_test_split_orl(X, y, n_train=5, random_state=trial)\n"
               "    abl_k['Raw']['svm'].append(evaluate_classifier(X_tr, y_tr, X_te, y_te, 'svm'))\n"
               "    abl_k['Raw']['knn'].append(evaluate_classifier(X_tr, y_tr, X_te, y_te, 'knn'))\n"
               "    for k in [1, 2, 3, 4, 5]:\n"
               "        key = f'k={k}x{k}'\n"
               "        prep = BCNLTPreprocessor(n_row_clusters=k, n_col_clusters=k, random_state=trial)\n"
               "        Xtr_b = prep.fit_transform(X_tr)\n"
               "        Xte_b = prep.transform(X_te)\n"
               "        abl_k[key]['svm'].append(evaluate_classifier(Xtr_b, y_tr, Xte_b, y_te, 'svm'))\n"
               "        abl_k[key]['knn'].append(evaluate_classifier(Xtr_b, y_tr, Xte_b, y_te, 'knn'))\n"
               "print('Ablation (5 trials):')\n"
               "for method, data in abl_k.items():\n"
               "    sa = np.array(data['svm'])*100\n"
               "    ka = np.array(data['knn'])*100\n"
               "    print(f'  {method:<10}: SVM={sa.mean():.2f}+/-{sa.std():.2f}  KNN={ka.mean():.2f}+/-{ka.std():.2f}')")

c5_grid = ("# grid_search_bcnlt(X_tr, y_tr, X_te, y_te, kr_range, kc_range)\n"
           "# returns (best_kr, best_kc, best_acc, acc_matrix)\n"
           "X_tr0, X_te0, y_tr0, y_te0 = train_test_split_orl(X, y, n_train=5, random_state=0)\n"
           "kr_range = [2, 3, 4, 5]\n"
           "kc_range = [2, 3, 4, 5]\n"
           "best_kr, best_kc, best_acc, acc_matrix = grid_search_bcnlt(\n"
           "    X_tr0, y_tr0, X_te0, y_te0,\n"
           "    kr_range=kr_range, kc_range=kc_range, clf_type='knn', verbose=True\n"
           ")\n"
           "print(f'Best: k_r={best_kr}, k_c={best_kc}, KNN={best_acc*100:.2f}%')")

c6_heatmap = ("fig, ax = plt.subplots(figsize=(5, 4))\n"
              "im = ax.imshow(acc_matrix*100, cmap='YlOrRd',\n"
              "               vmin=acc_matrix.min()*100-1, vmax=acc_matrix.max()*100+1)\n"
              "plt.colorbar(im, ax=ax, label='KNN Accuracy (%)')\n"
              "ax.set_xticks(range(len(kc_range)))\n"
              "ax.set_yticks(range(len(kr_range)))\n"
              "ax.set_xticklabels(kc_range)\n"
              "ax.set_yticklabels(kr_range)\n"
              "ax.set_xlabel('k_col (feature clusters)')\n"
              "ax.set_ylabel('k_row (sample clusters)')\n"
              "ax.set_title('KNN Accuracy vs Cluster Count (k_r x k_c)')\n"
              "for i in range(len(kr_range)):\n"
              "    for j in range(len(kc_range)):\n"
              "        ax.text(j, i, f'{acc_matrix[i,j]*100:.1f}', ha='center', va='center', fontsize=9)\n"
              "plt.tight_layout()\n"
              "plt.savefig('../results/figures/05_hyperparam_heatmap.png', dpi=150, bbox_inches='tight')\n"
              "plt.show()")

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
print('Notebook 05 v2 OK')
