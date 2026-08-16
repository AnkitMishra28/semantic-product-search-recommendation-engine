# Raw Dataset Documentation — Amazon Reviews 2023 (Electronics)

## 1. Dataset Overview
- **Dataset Name**: Amazon Reviews 2023 (Electronics)
- **Origin / Creators**: McAuley Lab, University of California San Diego (UCSD)
- **Primary Reference**: Hou et al., *"Bridging Language and Items for Retrieval and Recommendation: An Evaluation Benchmark for E-Commerce"*, 2024.
- **Repository**: [Hugging Face: McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Domain**: Consumer Electronics (`Electronics` category)

---

## 2. Source Files & Schemas

| Raw File | Direct Source Endpoint | Full File Size | Raw Format | Content Description |
| :--- | :--- | :--- | :--- | :--- |
| `meta_Electronics.jsonl` | `https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Electronics.jsonl` | ~5.00 GB | JSON Lines | Product metadata: titles, brands, categories, descriptions, bullet features, prices, ratings, and image links. |
| `Electronics_reviews.jsonl` | `https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Electronics.jsonl` | ~21.57 GB | JSON Lines | Customer interaction data: user IDs, product ASINs, star ratings (1.0-5.0), timestamps, and verified purchase flags. |

### Raw Product Record Schema (`meta_Electronics.jsonl`)
```json
{
  "main_category": "All Electronics",
  "title": "FS-1051 FATSHARK TELEPORTER V3 HEADSET",
  "average_rating": 3.5,
  "rating_number": 6,
  "features": ["UPC: 662774021904", "Weight: 0.600 lbs"],
  "description": ["Teleporter V3 kit sets a new level of value..."],
  "price": 199.99,
  "images": [
    {
      "thumb": "https://m.media-amazon.com/images/I/41qrX56lsYL._AC_US40_.jpg",
      "large": "https://m.media-amazon.com/images/I/41qrX56lsYL._AC_.jpg",
      "variant": "MAIN",
      "hi_res": "https://m.media-amazon.com/images/I/71YN+Qk3kCL._SL1500_.jpg"
    }
  ],
  "store": "Fat Shark",
  "categories": ["Electronics", "Television & Video", "Video Glasses"],
  "details": {"Brand": "Fat Shark", "Date First Available": "August 2, 2014"},
  "parent_asin": "B00MCW7G9M",
  "bought_together": null
}
```

### Raw Interaction Record Schema (`Electronics_reviews.jsonl`)
```json
{
  "rating": 5.0,
  "title": "Best Headphones in the Fifties price range!",
  "text": "I've bought these headphones three times...",
  "images": [],
  "asin": "B013J7WUGC",
  "parent_asin": "B07CJYMRWM",
  "user_id": "AG2L7H23R5LLKDKLBEF2Q3L2MVDA",
  "timestamp": 1676601581238,
  "helpful_vote": 0,
  "verified_purchase": true
}
```

---

## 3. Automated Acquisition Instructions

The project provides an automated streaming downloader that fetches official data without requiring manual browser downloads.

### A. Download Development Subset (Recommended for fast local experimentation)
```bash
python scripts/download_data.py --max-products 75000 --max-reviews 250000
```

### B. Download Complete Uncompressed Corpus
```bash
python scripts/download_data.py --full
```

---

## 4. Expected Directory Layout

```
data/
├── raw/
│   ├── .gitkeep
│   ├── README.md                      # (This document)
│   ├── meta_Electronics.jsonl         # (Git-ignored raw product metadata stream)
│   └── Electronics_reviews.jsonl      # (Git-ignored raw user review stream)
└── processed/
    ├── .gitkeep
    ├── products.parquet               # (Cleaned, deduplicated, embedding-ready product catalog)
    ├── interactions.parquet           # (Referentially-validated, temporally-split interactions)
    ├── evaluation_queries.json        # (Catalog-grounded evaluation queries)
    ├── dataset_profile.json           # (Computed empirical statistics)
    └── dataset_profile.md             # (Human-readable statistical profile)
```

---

## 5. Licensing & Usage Notes
- The Amazon Reviews 2023 dataset is released for academic research and non-commercial educational benchmarking by the McAuley Lab at UC San Diego.
- Raw dataset files are strictly **git-ignored** via `.gitignore` and must **never** be committed to version control.
