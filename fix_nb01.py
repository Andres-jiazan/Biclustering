import json

with open('notebooks/01_biclustering.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Rewrite cell 9 with simple title (no tricky f-string escapes)
lines = [
    "# Mean face per row cluster",
    "n_clusters = bicluster.n_row_clusters",
    "fig, axes = plt.subplots(1, n_clusters, figsize=(n_clusters * 2.5, 3))",
    "",
    "for a in range(n_clusters):",
    "    mask = bicluster.row_labels_ == a",
    "    n_a = int(mask.sum())",
    "    cluster_mean = X_train[mask].mean(axis=0)",
    "    axes[a].imshow(cluster_mean.reshape(IMG_SHAPE), cmap='gray', vmin=0, vmax=1)",
    "    axes[a].set_title(f'Cluster {a} (n={n_a})', fontsize=9)",
    "    axes[a].axis('off')",
    "",
    "fig.suptitle('Mean Face per Row Cluster', fontsize=11)",
    "plt.tight_layout()",
    "plt.savefig('../results/figures/01_cluster_mean_faces.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
]
nb['cells'][9]['source'] = "\n".join(lines)

with open('notebooks/01_biclustering.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Saved')
