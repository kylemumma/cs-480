# Self-Structured Course: Embeddings, RAG, and Vector Visualization

### Module 1: The Intuition (The "Explorer" Hook)
*Instead of a dry lecture, start with the people who visualize this best.*

**The Resource:** [Cohere LLM University (The Embeddings Chapter)](https://cohere.com/llmu)
* **The specific lesson:** [Introduction to Text Embeddings](https://cohere.com/llmu/text-embeddings) & [Semantic Search](https://cohere.com/llmu/introduction-semantic-search)
* **Why:** It is arguably the most visual, non-math-heavy explanation of what an embedding is currently on the web.

**The Resource:** [Jay Alammar’s "The Illustrated Word2Vec"](https://jalammar.github.io/illustrated-word2vec/)
* **Why:** This is the legendary blog post that everyone in the industry actually read to understand this.

---

### Module 2: The Engineering (The "Builder" Skills)
*For the RAG/Vector DB part, don't guess. Use the industry standard.*

**The Resource:** [DeepLearning.AI: Vector Databases (with Pinecone)](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)
* **Why:** It’s free, it’s 1 hour long, and it’s taught by the people who built the database. It covers "HNSW" (the index) and "RAG" perfectly.

**The Resource:** [DeepLearning.AI: Advanced Retrieval for AI with Chroma](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/)
* **Why:** This covers the "messy" parts—Query Expansion, Re-ranking, and "Lost in the Middle."

---

### Module 3: The Visualization (The "Art" Project)
*This is the hardest part to find. Most courses stop at "PCA."*

**The Resource:** [SciKit-Learn "Manifold Learning" Documentation](https://scikit-learn.org/stable/modules/manifold.html)
* **Why:** Surprisingly, the official docs are better than most courses. They have a visual gallery comparing t-SNE, UMAP (via libraries), and Isomap on the "S-Curve" dataset.

**The Resource:** [Pair Code (Google): Understanding UMAP](https://pair-code.github.io/understanding-umap/)
* **Why:** An interactive article that lets you tweak the "hyperparameters of gravity" and see the clusters move in real-time.

---

## The "CS 480" Assembly Instructions
*If I were you, I would not sign up for a generic Coursera course. I would do this:*

### Week 1: Visuals
* **Action:** Read Jay Alammar and play with the TensorFlow Projector.
* **Goal:** "Get the vibe."

### Week 2: The Tool
* **Action:** Take the DeepLearning.AI "Vector Databases" short course (1 hour).
* **Goal:** Build a working RAG bot in a notebook.

### Week 3: The Map
* **Action:** Clone the RAGxplorer repo.
* **Goal:** Run it on your own PDF. **This is the final project.**

### Week 4: The Deep Dive
* **Action:** If (and only if) you are still interested, read the HNSW and UMAP papers to understand why Week 3 worked.
