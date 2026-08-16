# Amazon Reviews 2023 (Electronics) — Dataset Profile Report

*Generated on: 2026-08-14T05:31:56.826686+00:00*

---

## 1. Executive Summary

| Dimension | Processed Metric |
| :--- | :--- |
| **Product Catalog Size** | **60,000** unique products |
| **User Interactions** | **31,286** ratings/reviews |
| **Unique Customer Users** | **16,841** users |
| **Unique Categories** | **875** categories |
| **Temporal Span** | 2002-09-27 01:01:30 UTC — 2023-03-18 14:29:57 UTC |
| **Catalog Review Coverage** | **14.44%** |

---

## 2. Product Catalog Characteristics

### Metadata Quality & Completeness
| Attribute | Missing Count | Missing Percentage |
| :--- | :--- | :--- |
| **Price** | 28,976 | 48.29% |
| **Brand** | 213 | 0.36% |
| **Description** | 17,310 | 28.85% |
| **Features** | 2,241 | 3.74% |

### Pricing Distribution (USD)
| Statistic | Value |
| :--- | :--- |
| **Min Price** | $0.01 |
| **25th Percentile (p25)** | $11.99 |
| **Median Price (p50)** | $20.17 |
| **Mean Price** | $86.22 |
| **75th Percentile (p75)** | $52.91 |
| **95th Percentile (p95)** | $349.00 |
| **Max Price** | $12998.00 |

### Top Product Categories
| Category | Product Count |
| :--- | :--- |
| Electronics | 58,198 |
| Computers & Accessories | 26,008 |
| Camera & Photo | 9,225 |
| Accessories | 7,714 |
| Computer Accessories & Peripherals | 6,722 |
| Bags, Cases & Sleeves | 6,636 |
| Laptop Accessories | 5,373 |
| Cases | 5,147 |
| Tablet Accessories | 5,074 |
| Television & Video | 4,718 |

---

## 3. User Interaction & Rating Dynamics

### Rating Distribution
| Rating Score | Count | Percentage |
| :--- | :--- | :--- |
| 1.0 ★ | 2,330 | 7.45% |
| 2.0 ★ | 1,276 | 4.08% |
| 3.0 ★ | 2,119 | 6.77% |
| 4.0 ★ | 4,253 | 13.59% |
| 5.0 ★ | 21,308 | 68.11% |

### Temporal Partitioning (Train / Val / Test)
| Split Partition | Interactions | Share | Temporal Date Range |
| :--- | :--- | :--- | :--- |
| **train** | 21,900 | 70.0% | 2002-09-27 01:01:30 UTC to 2020-12-17 12:55:18 UTC |
| **val** | 4,693 | 15.0% | 2020-12-17 19:11:58 UTC to 2022-01-21 18:17:06 UTC |
| **test** | 4,693 | 15.0% | 2022-01-21 18:27:24 UTC to 2023-03-18 14:29:57 UTC |

### Interaction Density Statistics
| Metric | Mean | Median | p75 | p95 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Interactions per User** | 1.86 | 1.0 | 2.0 | 5.0 | 96 |
| **Interactions per Product** | 3.61 | 1.0 | 3.0 | 12.0 | 839 |

---

## 4. Methodology & Research Integrity Notes
1. **Source Authenticity**: All product metadata and user interactions originate strictly from the official McAuley Lab Amazon Reviews 2023 Electronics dataset.
2. **Deterministic Sampling**: The development subset was selected via quality-weighted stratified scoring with a deterministic random seed (`seed=42`).
3. **Temporal Evaluation**: Partitions use global chronological quantile splits to prevent historical leakage.
