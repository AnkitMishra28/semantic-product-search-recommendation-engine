# Models Directory

This directory contains configuration, checkpoints, and cached weights for neural embedding models and rerankers.

## Supported Models

### 1. Dense Semantic Embedder
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Max Sequence Length**: 256
- **Pooling**: Mean pooling with normalized embeddings
- **Usage**: Encodes product text (title, features, category) into dense unit vectors for first-stage FAISS similarity search.

### 2. Cross-Encoder Reranker
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Architecture**: Cross-attention over concatenated `(query, product)` text
- **Output**: Uncalibrated logit / sigmoid relevance score
- **Usage**: Stage 2 reranking of Top-K candidates retrieved by FAISS.

## Storage Conventions

- Pre-trained model weights from Hugging Face Hub are automatically cached into `models/weights/` or Hugging Face cache.
- Fine-tuned domain adapters and custom heads will be saved under `models/custom/`.
