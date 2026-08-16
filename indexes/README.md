# Indexes Directory

This directory stores serialized vector index artifacts (FAISS binary files and mapping metadata).

## Index Topologies

1. **Flat Index (`IndexFlatIP`)**:
   - Exact inner product (cosine similarity on normalized vectors).
   - Serves as the ground-truth recall reference.

2. **Inverted File Index (`IndexIVFFlat`)**:
   - Approximate Nearest Neighbors (ANN) using Voronoi cells.
   - Recommended for dataset sizes 50K–1M items.

3. **Hierarchical Navigable Small World (`IndexHNSWFlat`)**:
   - Graph-based ANN offering high recall with sub-millisecond query latency.

## File Naming Convention

Indexes follow the naming format:
`<dataset>_<model>_<index_type>_<dimension>.index`

Example:
`electronics_all_minilm_l6_v2_hnsw_384.index`
`electronics_all_minilm_l6_v2_hnsw_384_id_map.json`
