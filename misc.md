# In here, I will paste random things to come back to

* HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise):How it works: It builds a hierarchy of connected points based on density. It assumes that clusters are "islands" of high density separated by "oceans" of low density.Why it's the gold standard: Unlike K-Means, you don't need to specify the number of clusters (*k*) beforehand. It also explicitly labels "noise" points (outliers) that don't fit well anywhere, which is crucial for cleaning up your data.+1Best for: Exploratory data analysis where you don't know how many themes exist.

---
## how to cluster
1. Dimensionality Reduction: UMAP (to compress vectors from 768d to 5d for easier clustering).

Dimensionality must be reduced for clustering. In 786d what we think of as "related" concepts will be incredibly far apart because so much detail / information is capture. When we use UMAP to reduce dimensionality, it essentially forces the embedding model to "summarize" the information into clusters.

2. Clustering: HDBSCAN (to find the groups).
3. Theme Extraction: c-TF-IDF (to extract keywords).
4. Labeling: GenAI (to turn keywords into a human-readable title).
