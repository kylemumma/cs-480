# Self-Structured Course: Embeddings, RAG, and Vector Visualization

---

## Module 1: The Intuition (The "Explorer" Hook)

*Instead of a dry lecture, start with the people who visualize this best.*

**Goal:** "Get the vibe" — and then prove you got it by generating your own embeddings.

### Demo: TensorFlow Embedding Projector

[projector.tensorflow.org](https://projector.tensorflow.org/)

On the right side, search for a word (e.g., "computer"). Notice how words like "software," "hardware," and "computing" float near it in 3D space. Rotate the view. Get a feel for what "closeness" means in this space.

### Read: Cohere LLM University — The Embeddings Chapter

[cohere.com/llmu](https://cohere.com/llmu)

Specific lessons: [Introduction to Text Embeddings](https://cohere.com/llmu/text-embeddings) and [Semantic Search](https://cohere.com/llmu/introduction-semantic-search). This is arguably the most visual, non-math-heavy explanation of what an embedding is currently on the web.

### Read: Jay Alammar's "The Illustrated Word2Vec"

[jalammar.github.io/illustrated-word2vec](https://jalammar.github.io/illustrated-word2vec/)

The legendary blog post that everyone in the industry actually read to understand this.

---

## Module 2: The Engineering (The "Builder" Skills)

*Build a RAG pipeline from scratch first, then learn the tools that scale it.*

**Goal:** Build a working RAG bot — first naively, then properly.

### 🛠️ Exercise: Naive RAG from Scratch (Do This First)

Before touching any vector database, build the dumbest possible RAG pipeline in a notebook:

1. Load a short PDF (5–10 pages) using `PyMuPDF` or `pdfplumber`.
2. Split it into chunks (start with fixed-size, ~300 tokens each, with ~50 token overlap).
3. Embed each chunk using `sentence-transformers` or the OpenAI API.
4. Store the embeddings in a plain NumPy array.
5. Given a user query, embed it, compute cosine similarity against all chunks, and retrieve the top 5.
6. Pass those chunks as context to an LLM (OpenAI, Anthropic, or a local model) and get an answer.

**Why this matters:** You need to feel the pain of brute-force search, bad chunking, and irrelevant retrieval *before* you'll appreciate what Pinecone and HNSW solve.

### Read & Build: Chunking Strategies

This is the single biggest practical lever in RAG quality, and most courses skip it entirely.

- Read Greg Kamradt's [chunking evaluation notebook](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) covering 5 levels of text splitting.
- Experiment: Take your PDF from the exercise above and try fixed-size, recursive, and semantic chunking. Compare retrieval quality for the same set of queries.

### Watch & Build: Vector Databases with Pinecone

[DeepLearning.AI: Vector Databases — Embeddings & Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)

Free, 1 hour, taught by the people who built the database. It covers HNSW (the index algorithm) and RAG end-to-end. As you go through it, notice how much faster and cleaner this is than your NumPy approach — that's the point.

### Watch & Build: Advanced Retrieval with Chroma

[DeepLearning.AI: Advanced Retrieval for AI with Chroma](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/)

This covers the "messy" real-world parts: Query Expansion, Re-ranking, and the "Lost in the Middle" problem.

### 🛠️ Exercise: Evaluate Your Pipeline

Before moving on, add basic evaluation to your RAG system:

1. Write 10 question-answer pairs based on your PDF (these are your "ground truth").
2. For each question, check: Did the correct chunk appear in your top-5 retrieval? (This is Recall@5.)
3. Compute Mean Reciprocal Rank (MRR): where in the ranked list did the correct chunk first appear?
4. Optional: Try the [RAGAS](https://docs.ragas.io/) library for automated RAG evaluation.

**Save your pipeline and your embedded chunks. You'll need them in Module 3.**

---

## Module 3: The Visualization (The "Art" Project)

*Most courses stop at "PCA." You won't.*

**Goal:** Visualize the embeddings from your own RAG pipeline. This is the final project.

### Read: SciKit-Learn Manifold Learning Documentation

[scikit-learn.org/stable/modules/manifold.html](https://scikit-learn.org/stable/modules/manifold.html)

Surprisingly, the official docs are better than most courses. They have a visual gallery comparing t-SNE, UMAP, and Isomap on the "S-Curve" dataset.

### Explore: Google PAIR — Understanding UMAP

[pair-code.github.io/understanding-umap](https://pair-code.github.io/understanding-umap/)

An interactive article that lets you tweak the hyperparameters and see the clusters move in real-time. Pay attention to how `n_neighbors` and `min_dist` change the shape of the output.

### 🛠️ Final Project: Visualize Your RAG Embeddings

Take the chunk embeddings you saved from Module 2 and do the following:

1. Reduce them to 2D using UMAP (`pip install umap-learn`).
2. Plot them with `matplotlib` or `plotly`, coloring each point by its source section, page number, or topic.
3. Hover over the points (if using Plotly) to see the chunk text.
4. Now re-run with t-SNE and PCA. Compare the three plots — what does each method preserve or distort?
5. Search for a query, highlight the retrieved chunks on the plot, and see if "close in the visualization" matches "close in retrieval."

If the clusters make sense visually and your retrieval results land in the right neighborhoods, you understand the full stack.

---

## Extra Credit

Read the original papers. These are more accessible than you'd expect:

- **UMAP paper:** [McInnes, Healy, & Melville (2018)](https://arxiv.org/abs/1802.03426) — Read this. If you did Module 3, you already have the intuition. The paper will formalize it.
- **HNSW paper:** [Malkov & Yashunin (2016)](https://arxiv.org/abs/1603.09320) — Explains why your Pinecone queries were fast. Skim Section 3 for the core algorithm.
