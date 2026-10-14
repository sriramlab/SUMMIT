"""Scale-free focal-context x polygenic-background epistasis (class-score U-statistics).

See docs/wiki/Epistasis-scale-free.md. ``design`` forms outcome columns,
``stream`` runs the single genotype traversal on SUMMIT's shared stream and
protected products, ``stats`` turns kernels into estimates with Hoeffding and
SNP-block standard errors (null-restricted for the degenerate statistics), and
``rank_pairs`` holds the rank-based pairwise tests.
"""
