note: this is the opus 4.6's revised version of the 12-week in depth syllabus. it has not been review by a human yet.

# Latent Space Analysis & Engineering: A 12-Week Self-Study Curriculum

**Curator's Note:** This program combines the intuition of **3Blue1Brown**, the technical depth of **Andrej Karpathy**, and the clarity of **Jay Alammar**. It is designed for disciplined self-study — every week ends with a concrete checkpoint so you can verify you've actually learned something before moving on.

## 📋 Course Overview

- **Goal:** Build a "Semantic Librarian" — a RAG-powered tool that visualizes a PDF as a galaxy of concepts and lets you query individual clusters.
- **Prerequisites:** Python proficiency, basic linear algebra (or willingness to learn it in Week 1).
- **Tools:** `Python`, `PyTorch`, `Sentence Transformers`, `Streamlit`, `UMAP`, `ChromaDB`.
- **Time commitment:** ~6–10 hours per week depending on the phase.

---

## Phase I: The Physics of Meaning (Weeks 1–2)

*The mathematical foundation of how "meaning" becomes geometry.*

---

### Week 1: The Vector Space Model

**Topic:** How we turn words into numbers without losing their soul.

**📺 Watch (Intuition):** [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- Watch Chapters 1–3 (Vectors, Linear Combinations, Matrices). This is non-negotiable.

**📖 Read (Theory):** [Google Machine Learning Crash Course: Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)
- Focus on "Translating to a Lower-Dimensional Space" and "Motivation."

**📖 Read (Classic):** [Jay Alammar: The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- The blog post that the industry actually read to understand embeddings.

**💻 Lab:** [PyTorch: Word Embeddings Tutorial](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)
- Run the N-Gram Language Modeling example.

**✅ Checkpoint:** In a notebook, generate embeddings for 20+ words across 3–4 categories using `sentence-transformers`. Compute pairwise cosine similarities. Write a brief paragraph: which pairs were closest? Which surprised you? Were there any that *should* have been close but weren't? Save this notebook — you'll use these embeddings again in Week 8.

---

### Week 2: The Manifold Hypothesis

**Topic:** Why high-dimensional data lives on "thin sheets" and how that explains why dimensionality reduction works at all.

**📖 Read (Deep Dive):** [Colah's Blog: Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
- A classic. It visualizes how neural networks "fold" space to separate data.

**💻 Lab:** [Scikit-Learn: Manifold Learning](https://scikit-learn.org/stable/modules/manifold.html)
- Run the "S-Curve" reduction example. Compare how IsoMap, LLE, and t-SNE each flatten a 3D shape into 2D. Notice what each method preserves and distorts.

**✅ Checkpoint:** Take the 20-word embeddings from Week 1. Reduce them to 2D using PCA, t-SNE, and UMAP (install `umap-learn`). Plot all three. Write a paragraph comparing them: do the category clusters hold? Which method separates them most cleanly?

---

## Phase II: The Encoder (Weeks 3–4)

*Understanding the engines that generate the embeddings.*

---

### Week 3: Transformers & Attention

**Topic:** The architecture that conquered NLP.

**📖 Read:** [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- Focus on understanding the "Query, Key, Value" vectors and how attention scores determine which tokens "look at" each other.

**📺 Watch:** [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Watch the first 60–90 minutes, through the self-attention implementation. You do not need to build the full GPT — the goal is to see attention in code, not to train a model.

**✅ Checkpoint:** In your own words (no more than one page), explain: what are Q, K, and V? Why does attention use a dot product? What does "softmax over the keys" actually do to the information flow? If you can't answer these clearly, re-read Alammar before moving on.

---

### Week 4: Sentence Embeddings & Embedding Model Selection

**Topic:** The models people actually use to embed text for retrieval — and how to choose between them.

**📖 Read:** [SBERT.net: Sentence Transformers Documentation](https://www.sbert.net/)
- Focus on the "Training Overview" and "Pretrained Models" pages. Understand the bi-encoder architecture: two texts go through the same model independently, and you compare their output vectors.

**📖 Read:** [Hugging Face: MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- Browse the leaderboard. Notice how different models excel at different tasks (retrieval vs. classification vs. clustering). This is how practitioners choose embedding models.

**💻 Lab:** Pick 3 embedding models of different sizes from the MTEB leaderboard. Embed the same 20 sentences with each. Compare their cosine similarity matrices. Is the most expensive model noticeably better for your use case?

**✅ Checkpoint:** Write a half-page recommendation: which embedding model would you choose for a RAG system over 10,000 document chunks, and why? Consider model size, speed, and retrieval quality.

---

## Phase III: Engineering the Void (Weeks 5–8)

*How to store, search, retrieve, and evaluate vectors at scale.*

---

### Week 5: Naive RAG from Scratch

**Topic:** Build the dumbest possible RAG pipeline so you feel the pain that proper tools solve.

["Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer" by freeCodeCamp.org](https://www.youtube.com/watch?v=sVcwVQRHIc8)
or maybe
Daniel Bourke's "Local RAG From Scratch"
* Do this [https://github.com/mrdbourke/simple-local-rag](https://github.com/mrdbourke/simple-local-rag)
* if you get stuck refer to the video [https://www.youtube.com/watch?v=qN_2fnOPY-M](https://www.youtube.com/watch?v=qN_2fnOPY-M)

**✅ Checkpoint:** Your pipeline answers questions about your PDF. Write down 3 queries where it worked well and 3 where it failed. For the failures, look at what chunks were actually retrieved — was the problem bad retrieval, bad chunking, or bad generation?

---

### Week 6: Chunking, Indexing, & Vector Databases

**Topic:** The two biggest levers in RAG quality — how you split the text and how you search it.

**📖 Read (Chunking):** Work through Greg Kamradt's [5 Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) notebook. This covers fixed-size, recursive, document-specific, semantic, and agentic chunking.

**📖 Read (Indexing):** [Pinecone: Hierarchical Navigable Small Worlds (HNSW)](https://www.pinecone.io/learn/series/faiss/hnsw/)
- Focus on the "Skip List" analogy. This is the best available explanation of the algorithm that makes vector search fast.

**💻 Lab (Vector DB — Choose One):**
- **Option A (Local):** [ChromaDB Getting Started](https://docs.trychroma.com/getting-started). Run it locally in a notebook.
- **Option B (Cloud):** [Pinecone Quickstart](https://docs.pinecone.io/guides/get-started/quickstart). Set up a free-tier index.
- **Task:** Migrate your Week 5 pipeline from NumPy to the vector database. Re-run your 6 test queries. Is retrieval faster? Are results better?

**💻 Lab (Chunking Experiment):** Take your PDF from Week 5 and re-chunk it using recursive and semantic splitting (use LangChain's text splitters or roll your own). Load each chunking strategy into your vector DB. Compare retrieval quality on the same queries.

**✅ Checkpoint:** A comparison table: for each of your 6 test queries, show which chunking strategy retrieved the best chunks. Which strategy won overall?

---

### Week 7: Advanced RAG — Re-ranking, Query Expansion, & Failure Modes

**Topic:** The techniques that separate a demo from a system that actually works.

**📺 Watch:** [DeepLearning.AI: Advanced Retrieval for AI with Chroma](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/)
- Covers Query Expansion, Re-ranking, and the "Lost in the Middle" problem (where LLMs ignore context placed in the middle of a long prompt).

**📖 Read (Re-ranking):** [SBERT: Cross-Encoders for Re-ranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- Understand the key distinction: bi-encoders are fast (used for initial retrieval over millions of chunks) but less accurate. Cross-encoders are slow (they process query + document together) but much more accurate. So you retrieve with a bi-encoder, then re-rank the top results with a cross-encoder.

**💻 Lab:** Add a cross-encoder re-ranking step to your pipeline. Retrieve the top 20 chunks with your bi-encoder, then re-rank them with `cross-encoder/ms-marco-MiniLM-L-6-v2`. Compare the top-5 before and after re-ranking.

**✅ Checkpoint:** For your 6 test queries, show the top-5 retrieval results with and without re-ranking. Did re-ranking fix any of the failures from Week 5?

---

### Week 8: Evaluating RAG

**Topic:** How to measure whether your retrieval system is actually working.

**📖 Read:** [RAGAS Documentation](https://docs.ragas.io/) — focus on the core metrics: context precision, context recall, faithfulness, and answer relevancy.

**💻 Lab:**

1. Write 15–20 question-answer pairs for your PDF (these are your ground truth).
2. For each question, manually identify which chunk(s) contain the answer.
3. Compute Recall@5 (did the correct chunk appear in your top 5?) and Mean Reciprocal Rank (where did it first appear?).
4. Optional: Run RAGAS automated evaluation over your pipeline.

**✅ Checkpoint:** A summary of your pipeline's retrieval and generation quality. Recall@5, MRR, and 2–3 sentences on where the system still struggles and what you'd try next.

---

## Phase IV: The Cartography (Weeks 9–10)

*Visualizing the high-dimensional space.*

---

### Week 9: Dimensionality Reduction — PCA, t-SNE, UMAP

**Topic:** Smashing 1500 dimensions down to 2 — three different philosophies.

**📺 Watch:** [StatQuest: PCA Main Ideas](https://www.youtube.com/watch?v=HMOI_lkzW08)
- The best explanation of eigenvectors that won't make you cry. PCA preserves global variance but can crush local neighborhoods.

**📖 Read (Interactive):** [Google PAIR: Understanding UMAP](https://pair-code.github.io/understanding-umap/)
- Play with the hyperparameters. Pay attention to how `n_neighbors` (local vs. global focus) and `min_dist` (tight vs. spread clusters) reshape the output.

**📖 Read:** [Distill.pub: How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- Critical for avoiding common misinterpretations — cluster sizes and distances between clusters in t-SNE are often meaningless.

**💻 Lab:** Take your document chunk embeddings from Week 6. Run PCA, t-SNE, and UMAP. Plot all three with Plotly, coloring points by source section or page. Hover over points to see chunk text.

**✅ Checkpoint:** Three side-by-side plots of the same embeddings. Write a paragraph: which method produces the most meaningful clusters for your document? Why?

---

### Week 10: Clustering & Interactive Visualization

**Topic:** Going from a scatter plot to a tool someone can actually use.

**📖 Read:** [Scikit-Learn: Clustering — HDBSCAN](https://scikit-learn.org/stable/modules/clustering.html#hdbscan)
- You need this for the final project. HDBSCAN works well on UMAP output because it handles variable-density clusters and identifies noise points.

**📺 Watch:** [Nomic AI: Atlas Demo](https://atlas.nomic.ai/)
- Inspiration: see what a production-grade semantic map looks like.

**💻 Lab:**

1. Run UMAP on your chunk embeddings, then cluster the 2D output with HDBSCAN.
2. For each cluster, send its chunk texts to an LLM and ask for a 1-sentence summary. These are your cluster labels.
3. Build a Plotly scatter plot where each cluster is a different color, with the LLM-generated label in the legend.
4. Wrap it in a basic Streamlit app.

**✅ Checkpoint:** A working Streamlit app that shows your document as a labeled cluster map. You can hover over points to see chunk text and identify the topics. This is 80% of your final project — if this works, you're in good shape.

---

## Phase V: Final Project (Weeks 11–12)

---

### 🚀 The Semantic Librarian

**The Prompt:**
> Build a tool that allows a user to upload a PDF. The tool visualizes the PDF as a "Galaxy of Concepts" (a UMAP scatter plot with labeled clusters). When the user clicks a cluster on the map, the LLM summarizes only that cluster's content. The user can also ask free-text questions and see which chunks were retrieved, highlighted on the map.

**Recommended Tech Stack:**
- **Ingestion:** `PyMuPDF` or `pdfplumber` (PDF reading) + recursive or semantic chunking.
- **Embedding:** A sentence-transformers model you chose in Week 4 (or OpenAI embeddings).
- **Storage:** `ChromaDB` (local vector DB).
- **Re-ranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (optional but recommended).
- **Visualization:** `UMAP` + `HDBSCAN` + `Plotly`.
- **Interface:** `Streamlit`.

**Week 11 Milestone:** The full pipeline works end-to-end in a Jupyter notebook. You can upload a PDF, chunk it, embed it, store it, visualize it, cluster it, and query it. No UI polish yet — just prove the pipeline works.

**Week 12 Milestone:** Wrap it in Streamlit. The user can upload a PDF, see the galaxy, click clusters for summaries, and ask questions with retrieved chunks highlighted on the map.

**Deliverables:**
1. A GitHub repository with your code and a README explaining your design choices (which embedding model, which chunking strategy, why).
2. A 2-minute screen recording demonstrating you navigating the map and asking questions about specific clusters.

---

## Extra Credit

Read the papers. These are more accessible than you'd expect, especially after completing the curriculum:

- **UMAP:** [McInnes, Healy, & Melville (2018)](https://arxiv.org/abs/1802.03426) — formalizes the intuition you built in Week 9.
- **HNSW:** [Malkov & Yashunin (2016)](https://arxiv.org/abs/1603.09320) — explains why your vector search was fast. Skim Section 3 for the core algorithm.
- **Sentence-BERT:** [Reimers & Gurevych (2019)](https://arxiv.org/abs/1908.10084) — the paper behind the embedding models you've been using since Week 4.
