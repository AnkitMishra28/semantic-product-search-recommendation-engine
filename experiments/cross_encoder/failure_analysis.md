# Phase 9: Cross-Encoder Reranking Diagnostic Failure Analysis

## 1. Diagnostic Taxonomy of Second-Stage Neural Reranking Errors

Systematic analysis of Cross-Encoder reranking across 30 catalog-grounded queries identifies four distinct error modes:

### 1.1 First-Stage Recall Miss (Candidate Pool Exclusion)
- **Mechanism**: Ground-truth product was missing exact lexical keywords for BM25 and was outside the Top-100 FAISS HNSW embedding neighborhood.
- **Impact**: The Cross-Encoder never receives the item in its candidate pool; second-stage reranking cannot score what first-stage retrieval fails to capture.
- **Mitigation**: First-stage query expansion (synonym enrichment, Query Understanding relaxation) and hybrid candidate union.

### 1.2 Second-Stage Reranking Exclusion (False Negatives)
- **Mechanism**: Ground-truth product entered the candidate pool at rank 20–80, but Cross-Encoder assigned higher relevance to competing distractor items.
- **Impact**: Item drops below the top-20 final result threshold.
- **Mitigation**: Soft business signal blending (ratings, review volume) and domain-specific fine-tuning of cross-encoder weights.

### 1.3 Fine-Grained Generation / Spec Ambiguity (Regression Cases)
- **Mechanism**: Adjacent hardware generations or specifications (e.g. Cat6 vs Cat8, 65W vs 100W) have subtle lexical distinctions that generic MS-MARCO pretraining occasionally misranks.
- **Impact**: A correct spec product is demoted in favor of a superficially matching brand sibling.
- **Mitigation**: Attribute-aware structured text serialization (`variant=title_brand_category_features`).

---

## 2. Representative Qualitative Case Studies

### 2.1 Representative Cross-Encoder Improvements (Promotions)
#### Case 1.1: laptop cooling pad with quiet fans
- **Target Product**: [B0BGGR6K4M] Kootek Laptop Cooling Pad 12"-17" Cooler Pad Chill Mat 5 Quiet Fans LED Lights and 2 USB 2.0 Ports Adjustable Mounts Laptop Stand Height Angle, Red
- **First-Stage Retriever Rank**: 7
- **Final Cross-Encoder Rank**: **2**
- **Cross-Encoder Score**: `8.6617`
- **Diagnosis**: Cross-attention captured deep contextual feature relevance between query 'laptop cooling pad with quiet fans' and product specifications, promoting the item from first-stage rank 7 up to rank 2.

#### Case 1.2: surge protector power strip with USB ports
- **Target Product**: [B0BWMXQZ32] Surge Protector Power Strip with USB, TROND Ultra Thin Flat Plug 10ft Long Extension Cord 1625W, 3 USB A & 1 Type C, 4 AC Outlets 1440J Surge Protection Wall Mount for Home Office Dorm Room, Black
- **First-Stage Retriever Rank**: 28
- **Final Cross-Encoder Rank**: **3**
- **Cross-Encoder Score**: `9.3856`
- **Diagnosis**: Cross-attention captured deep contextual feature relevance between query 'surge protector power strip with USB ports' and product specifications, promoting the item from first-stage rank 28 up to rank 3.

#### Case 1.3: wifi range extender booster for home coverage
- **Target Product**: [B0C65RFZ5J] WiFi Extender Internet Signal Booster and Amplifier up to 8500 sq.ft - Long Range Coverage Wi-Fi Repeater for Home, with Ethernet Port & Access Point Mode, Support 40 Devices,1 Touch Easy Setup
- **First-Stage Retriever Rank**: 7
- **Final Cross-Encoder Rank**: **1**
- **Cross-Encoder Score**: `8.6529`
- **Diagnosis**: Cross-attention captured deep contextual feature relevance between query 'wifi range extender booster for home coverage' and product specifications, promoting the item from first-stage rank 7 up to rank 1.

### 2.2 Representative Cross-Encoder Regressions (Demotions)
#### Case 2.1: wifi range extender booster for home coverage
- **Target Product**: [B0C6LCJ95V] WiFi Extenders Signal Booster for Home, WiFi Extender 300Mbps, WiFi Booster, Wireless Signal Range Amplifier Covers Up to 3000 Sq.ft, WiFi Repeater, Internet Booster, Internet Extender
- **First-Stage Retriever Rank**: 3
- **Final Cross-Encoder Rank**: **7**
- **Cross-Encoder Score**: `7.9604`
- **Diagnosis**: First-stage retrieval ranked the item at rank 3, but cross-encoder assigned a lower score (7.9604), causing it to drop to rank 7.

### 2.3 Representative First-Stage Recall Misses
#### Case 3.1: noise cancelling bluetooth headphones for travel
- **Target Product**: [B0BW4PFM58] OontZ Angle 3 Bluetooth Speaker, Portable Wireless Bluetooth 5.0 Speaker, 10 Watts, Crystal Clear Stereo Sound, Rich Bass, IPX5 Water Resistant, Loud Portable Bluetooth Speaker (Black)
- **First-Stage Retriever Rank**: `Not in Top 100 (>100)`
- **Diagnosis**: The relevant product lacked strong lexical keyword overlap with the query for BM25 and was outside the Top-100 FAISS HNSW dense neighborhood, preventing the Cross-Encoder from seeing it.

#### Case 3.2: noise cancelling bluetooth headphones for travel
- **Target Product**: [B07S764D9V] Panasonic ErgoFit Wired Earbuds, In-Ear Headphones with Microphone and Call Controller, Ergonomic Custom-Fit Earpieces (S/M/L), 3.5mm Jack for Phones and Laptops - RP-TCM125-A (Blue)
- **First-Stage Retriever Rank**: `Not in Top 100 (>100)`
- **Diagnosis**: The relevant product lacked strong lexical keyword overlap with the query for BM25 and was outside the Top-100 FAISS HNSW dense neighborhood, preventing the Cross-Encoder from seeing it.

#### Case 3.3: noise cancelling bluetooth headphones for travel
- **Target Product**: [B07V6CGGW2] R-fun AirPods Case Cover, Soft Silicone Protective Cover with Keychain for Women Men Compatible with Apple AirPods 2nd 1st Generation Charging Case, Front LED Visible-Pink Sand
- **First-Stage Retriever Rank**: `Not in Top 100 (>100)`
- **Diagnosis**: The relevant product lacked strong lexical keyword overlap with the query for BM25 and was outside the Top-100 FAISS HNSW dense neighborhood, preventing the Cross-Encoder from seeing it.

### 2.4 Representative Second-Stage Reranking Exclusions
#### Case 4.1: wireless earbuds with charging case for running
- **Target Product**: [B0BHWYQ47Y] kurdene Bluetooth Wireless Earbuds, S8 Deep Bass Sound 38H Playtime IPX8 Waterproof Earphones Call Clear with Microphone in-Ear Bluetooth Headphones Comfortable for iPhone, Android
- **First-Stage Retriever Rank**: 87
- **Final Cross-Encoder Rank**: `> 20 (Dropped)`
- **Diagnosis**: Product was captured in the candidate pool at rank 87, but Cross-Encoder scored other items higher, pushing it beyond the top-20 cutoff.

#### Case 4.2: portable bluetooth speaker waterproof with deep bass
- **Target Product**: [B099V8GPR4] JBL Flip 4, Black - Waterproof, Portable & Durable Bluetooth Speaker - Up to 12 Hours of Wireless Streaming - Includes Noise-Cancelling Speakerphone, Voice Assistant & JBL Connect+
- **First-Stage Retriever Rank**: 47
- **Final Cross-Encoder Rank**: `> 20 (Dropped)`
- **Diagnosis**: Product was captured in the candidate pool at rank 47, but Cross-Encoder scored other items higher, pushing it beyond the top-20 cutoff.

#### Case 4.3: portable bluetooth speaker waterproof with deep bass
- **Target Product**: [B09MMWM2Y8] JBL FLIP 5, Waterproof Portable Bluetooth Speaker, Gray
- **First-Stage Retriever Rank**: 42
- **Final Cross-Encoder Rank**: `> 20 (Dropped)`
- **Diagnosis**: Product was captured in the candidate pool at rank 42, but Cross-Encoder scored other items higher, pushing it beyond the top-20 cutoff.

