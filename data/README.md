# Data Directory — Amazon Reviews 2023 (Electronics)

This directory stores raw downloads and processed datasets for the semantic search and recommendation engine.

---

## 1. Subdirectories

- `raw/`: Raw downloaded JSONL streams (`meta_Electronics.jsonl`, `Electronics_reviews.jsonl`) fetched from the official McAuley Lab Hugging Face repository. (Git-ignored). See [data/raw/README.md](file:///C:/Users/am280/OneDrive/Desktop/Amazon-AppliedScientist/Amazon-Scale%20Semantic%20Product%20Search%20&%20Recommendation%20Engine/data/raw/README.md).
- `processed/`: Validated Parquet datasets, catalog-grounded evaluation queries, and statistical profiling reports.

---

## 2. Processed Datasets & Schemas

### A. Product Catalog (`data/processed/products.parquet`)
Parquet file containing 60,000 deduplicated, information-rich product records.

| Field | Type | Description |
| :--- | :--- | :--- |
| `parent_asin` | `string` | **Canonical product identifier** (Primary Key) |
| `title` | `string` | Cleaned product title |
| `brand` | `string \| null` | Normalized brand name |
| `categories` | `list<string>` | Hierarchical category path |
| `description` | `string` | Cleaned description text |
| `features` | `list<string>` | Bullet specifications |
| `price` | `double \| null` | Cleaned numeric price in USD |
| `average_rating` | `double \| null` | Average review star rating (1.0 - 5.0) |
| `rating_number` | `int64` | Total count of ratings/reviews |
| `image_url` | `string \| null` | Primary image URL |
| `images` | `list<string>` | Complete list of product image URLs |
| `bought_together` | `list<string>` | Co-purchased product ASINs |
| `embedding_text` | `string` | Formatted label-delimited document representation |

### B. User Interactions (`data/processed/interactions.parquet`)
Parquet file containing 31,286 user ratings/reviews filtered for catalog referential integrity.

| Field | Type | Description |
| :--- | :--- | :--- |
| `user_id` | `string` | Unique customer identifier |
| `parent_asin` | `string` | Canonical product identifier (Foreign Key -> `products.parent_asin`) |
| `rating` | `double` | Star rating (1.0 to 5.0) |
| `timestamp` | `int64` | Interaction timestamp (epoch milliseconds) |
| `verified_purchase` | `boolean` | Verified purchase indicator |
| `helpful_vote` | `int64` | Helpfulness upvotes |
| `split` | `string` | Temporal split assignment (`train` [70%], `val` [15%], `test` [15%]) |

### C. Ground-Truth Evaluation Queries (`data/processed/evaluation_queries.json`)
JSON array containing 30 intent-diverse search queries grounded strictly against real catalog products.

---

## 3. Preprocessing & Validation CLI

1. **Acquire Raw Data**:
   ```bash
   python scripts/download_data.py --max-products 75000 --max-reviews 250000
   ```

2. **Execute Ingestion & Preprocessing**:
   ```bash
   python scripts/preprocess_data.py --target-products 60000 --seed 42
   ```

3. **Run Validation Suite**:
   ```bash
   python scripts/validate_dataset.py
   ```

4. **Generate Statistical Profile**:
   ```bash
   python scripts/profile_dataset.py
   ```
