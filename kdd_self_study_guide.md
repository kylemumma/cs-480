# Knowledge Discovery & Data Mining: A Self-Study Guide

A practical roadmap for learning modern KDD — from fundamentals to the cutting edge. Work through the phases in order; each builds on the last.

---

## Before You Start

You'll need a working foundation in these areas. If you're rusty, the linked resources can get you up to speed.

| Prerequisite | What You Need | Free Resource |
|---|---|---|
| **Python** | NumPy, pandas, basic scikit-learn | [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) |
| **Probability & Stats** | Distributions, Bayes' theorem, hypothesis testing | [StatQuest (YouTube)](https://www.youtube.com/@statquest) |
| **Linear Algebra** | Vectors, matrices, eigenvalues, SVD | [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) |
| **SQL** | Joins, aggregation, subqueries | [SQLBolt](https://sqlbolt.com/) |
| **Calculus** | Derivatives, chain rule, gradients | [Khan Academy Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus) |

**Tools to install:** Python 3.10+, Jupyter, scikit-learn, pandas, matplotlib, XGBoost, PyTorch. A free Google Colab account works if you don't want to set up locally.

---

## Phase 1: Foundations

### 1. The KDD Process

Data mining isn't just running algorithms — it's a full lifecycle.

**Learn:** Problem formulation → data collection → preprocessing → modeling → evaluation → deployment → monitoring. Understand that this cycle is iterative and messy in practice.

**Read:** Chapters 1–2 of *Data Mining: Concepts and Techniques* (Han, Kamber, Pei) or the free [KDD Process overview on Wikipedia](https://en.wikipedia.org/wiki/Knowledge_discovery_in_databases).

**Do:** Pick any Kaggle dataset. Before touching a model, spend an hour just *understanding* the data — what's missing, what's noisy, what the columns actually mean.

---

### 2. Data Preprocessing & Feature Engineering

This is where most real-world time goes. Get good at it early.

**Key topics:**
- Exploratory data analysis (histograms, correlations, summary stats)
- Handling missing data (imputation strategies: mean, KNN, multiple imputation)
- Encoding categorical variables (one-hot, ordinal, target encoding)
- Feature scaling (standardization vs. normalization)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Dealing with class imbalance (SMOTE, undersampling)

**Do:** Work through the [Titanic](https://www.kaggle.com/c/titanic) or [Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) Kaggle competitions. Focus on preprocessing and feature engineering more than model tuning.

---

### 3. Classification

The most widely used data mining task.

**Work through these models in order** (simplest → most powerful):

| Model | Why Learn It | Key Idea |
|---|---|---|
| Logistic Regression | Baseline for everything | Linear decision boundary + sigmoid |
| Naive Bayes | Fast, works with little data | Bayes' theorem + independence assumption |
| k-Nearest Neighbors | No training step, intuitive | Classify by neighbors |
| Decision Trees (CART) | Interpretable, basis for ensembles | Recursive splitting on features |
| Random Forest | Powerful, hard to mess up | Bagging + random feature subsets |
| XGBoost / LightGBM | State-of-the-art for tabular data | Gradient-boosted decision trees |
| SVM | Elegant theory | Maximum margin + kernel trick |

**Evaluation — don't skip this:**
- Train/validation/test splits and k-fold cross-validation
- Metrics: accuracy, precision, recall, F1, AUC-ROC
- Confusion matrices, overfitting, bias-variance tradeoff
- Regularization (L1, L2) and hyperparameter tuning

**Read:** Chapters 8–9 of *Introduction to Statistical Learning* (ISLR) — [free PDF](https://www.statlearning.com/).

**Do:** Build a classifier on a real dataset. Try at least 3 different models, compare them properly with cross-validation, and write up which one you'd deploy and why.

---

### 4. Regression

**Key topics:**
- Linear and polynomial regression, Ridge, Lasso, Elastic Net
- Tree-based regression (XGBoost for regression)
- Metrics: RMSE, MAE, R²
- Prediction intervals and uncertainty (conformal prediction)

**Do:** The Kaggle Housing Prices competition is ideal for practicing regression.

---

### 5. Clustering

Finding structure without labels.

**Key algorithms:**
- **k-Means** — fast and simple; learn k-means++ initialization and the elbow method
- **DBSCAN / HDBSCAN** — density-based, handles arbitrary shapes and noise
- **Hierarchical clustering** — dendrograms, different linkage methods
- **Gaussian Mixture Models** — soft clustering via the EM algorithm

**Validation:** Silhouette score, adjusted Rand index (when you have ground truth), visual inspection.

**Do:** Cluster a customer dataset or the [Mall Customers dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python). Visualize with t-SNE or UMAP.

---

### 6. Association & Pattern Mining

Less trendy than it used to be, but still important conceptually.

**Key topics:** Support, confidence, lift. Apriori algorithm. FP-Growth. Sequential pattern mining (PrefixSpan).

**Do:** Run association rule mining on the [Online Retail dataset](https://archive.ics.uci.edu/ml/datasets/online+retail). Look for interesting product co-purchase patterns.

---

### 7. Anomaly Detection

Critical in fraud, security, manufacturing, and healthcare.

**Key methods:**
- Statistical (z-scores, IQR)
- Isolation Forest (fast, effective, scalable)
- Local Outlier Factor (LOF)
- One-Class SVM
- Autoencoders (deep learning approach)

**Do:** Try anomaly detection on the [Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Pay attention to evaluation under extreme class imbalance.

---

### 8. Responsible Data Mining

Not optional. Build these habits now.

**Key topics:**
- Fairness definitions (demographic parity, equalized odds) and why you can't satisfy them all simultaneously
- Explainability tools: SHAP values, LIME
- Privacy: differential privacy basics, anonymization pitfalls
- Bias sources: selection bias, measurement bias, historical bias
- Documentation: model cards, datasheets for datasets

**Read:** [Fairness and Machine Learning](https://fairmlbook.org/) (free online textbook by Barocas, Hardt, Narayanan).

**Do:** Audit a classifier you've built for demographic bias. Use the `fairlearn` or `aif360` Python libraries.

---

## Phase 2: Deep Learning Essentials

Before tackling advanced KDD topics, you need a working knowledge of deep learning.

### 9. Neural Networks & Deep Learning

**Learn in this order:**
1. **Fundamentals** — Neurons, layers, loss functions, backpropagation, SGD and Adam
2. **Practical skills** — PyTorch basics, training loops, learning rate schedules, dropout, batch norm
3. **CNNs** — Convolution, pooling, ResNet. Applications: images, spatial data
4. **Sequence models** — RNNs, LSTMs, GRUs for sequential and temporal data
5. **Transformers & Attention** — Self-attention, positional encoding, the architecture behind LLMs

**Read:** [Practical Deep Learning for Coders](https://course.fast.ai/) (free course by fast.ai) — the best way to learn by doing.

**Do:** Fine-tune a pre-trained image model (e.g., ResNet on a custom dataset). Train a simple text classifier using Hugging Face Transformers.

---

## Phase 3: Advanced Topics

Pick the modules most relevant to your interests and goals. Each is roughly 1–3 weeks of study.

### 10. Graph Mining & GNNs

Graphs are everywhere: social networks, molecules, knowledge bases, supply chains.

**Key topics:** Graph neural networks (GCN, GraphSAGE, GAT), knowledge graphs and embeddings (TransE), community detection, link prediction.

**Tools:** PyTorch Geometric (PyG), NetworkX, DGL.

**Start with:** [Stanford CS224W](http://web.stanford.edu/class/cs224w/) (free lectures and assignments).

---

### 11. NLP & Text Mining

**Key topics:** Word embeddings (word2vec, GloVe) → BERT and Transformer models → LLM prompting for data mining tasks → RAG (retrieval-augmented generation).

**Tools:** Hugging Face Transformers, spaCy, sentence-transformers.

**Do:** Build a text classification pipeline with BERT fine-tuning. Then try the same task with zero-shot prompting of an LLM and compare.

---

### 12. Time Series & Forecasting

**Key topics:** ARIMA, exponential smoothing → deep approaches (N-BEATS, PatchTST, temporal CNNs) → time series classification (ROCKET) → anomaly and change point detection.

**Do:** Forecast on the [M5 competition dataset](https://www.kaggle.com/c/m5-forecasting-accuracy) or any real time series you care about.

---

### 13. Recommender Systems

**Key topics:** Collaborative filtering (matrix factorization, ALS) → deep recommendation models (neural CF, two-tower retrieval) → evaluation (offline metrics vs. A/B tests, popularity bias, diversity).

**Start with:** [Google's Recommendation Systems course](https://developers.google.com/machine-learning/recommendation) (free).

---

### 14. Foundation Models & LLMs

**Key topics:** Self-supervised learning (contrastive, masked, generative) → scaling laws → fine-tuning vs. prompting → parameter-efficient fine-tuning (LoRA) → RAG → AI agents for data analysis.

**Do:** Fine-tune a small LLM (e.g., via Hugging Face PEFT) on a domain-specific task. Build a simple RAG pipeline with a vector database.

---

### 15. Causal Inference

Going beyond correlation to understand *why*.

**Key topics:** Potential outcomes framework, causal DAGs, confounding, propensity score matching, treatment effect estimation, causal forests.

**Read:** [Causal Inference: The Mixtape](https://mixtape.scunning.com/) (free) or Brady Neal's [Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course) (free lectures).

---

### 16. Scalable & Production Data Mining

**Key topics:** Distributed computing (Spark MLlib), GPU-accelerated mining (RAPIDS, FAISS), MLOps (experiment tracking with MLflow, model registries, CI/CD for ML, data drift monitoring), feature stores, model serving.

**Do:** Deploy a model as an API endpoint. Set up experiment tracking with MLflow. Practice with Spark on a dataset too large for pandas.

---

## Suggested Study Plans

### 🏃 Fast Track (3–4 months, ~10 hrs/week)
Phases 1–2 (sections 1–9). You'll have a strong applied foundation.

### 🎯 Comprehensive (6–9 months, ~10 hrs/week)
Phases 1–3, picking 3–4 advanced modules that match your goals.

### 🔬 Research-Oriented (9–12 months)
All phases. Supplement with paper reading — start with best paper awards from recent KDD, NeurIPS, and ICML conferences.

---

## Key Textbooks & Resources

| Resource | Coverage | Cost |
|---|---|---|
| *Introduction to Statistical Learning* (James et al.) | Classical ML foundations | Free |
| *Data Mining: Concepts and Techniques* (Han, Kamber, Pei) | Comprehensive KDD | Paid |
| *Deep Learning* (Goodfellow, Bengio, Courville) | Deep learning theory | Free online |
| *Dive into Deep Learning* (d2l.ai) | Hands-on deep learning | Free |
| fast.ai courses | Practical deep learning | Free |
| Stanford CS224W | Graph ML | Free |
| Hugging Face NLP Course | NLP & Transformers | Free |
| *Fairness and Machine Learning* (Barocas et al.) | Responsible AI | Free |

---

## One Last Piece of Advice

The best way to learn data mining is to *mine data*. Theory is important, but this is fundamentally a practical discipline. Find a dataset you're genuinely curious about — something from your domain, your hobby, your city — and work through the full KDD process end to end. You'll learn more from one messy real-world project than from ten clean textbook exercises.

Good luck. 🚀
