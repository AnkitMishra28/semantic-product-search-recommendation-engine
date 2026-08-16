# BM25 Lexical Baseline — Failure Analysis & Diagnostic Report

*Generated on: 2026-08-14 07:15:08 UTC*

---

## 1. Baseline Performance Summary

| Metric | Measured Value |
| :--- | :--- |
| **Total Evaluation Queries** | **30** |
| **Recall@10** | **0.0500** (5.00%) |
| **Recall@50** | **0.1167** (11.67%) |
| **Recall@100** | **0.1875** (18.75%) |
| **MRR@10** | **0.1140** |
| **NDCG@10** | **0.0512** |
| **Query Latency (p50)** | **234.13 ms** |
| **Query Latency (p95)** | **377.75 ms** |
| **Query Latency (p99)** | **416.63 ms** |

---

## 2. Zero-Hit Query Distribution

| Threshold | Zero-Hit Query Count | Failure Rate |
| :--- | :--- | :--- |
| **Zero Relevant in Top 10** | **21** / 30 | **70.0%** |
| **Zero Relevant in Top 50** | **17** / 30 | **56.7%** |
| **Zero Relevant in Top 100** | **16** / 30 | **53.3%** |

---

## 3. Detailed Failure Case Studies

### Query `q_001`: *"noise cancelling bluetooth headphones for travel"*
- **Intent Type**: `semantic_use_case`
- **Identified Failure Category**: **Vocabulary Mismatch / Contextual Intent Blindness**
- **Ground-Truth Relevant Products (Expected)**:
  - `B0BW4PFM58`: OontZ Angle 3 Bluetooth Speaker, Portable Wireless Bluetooth 5.0 Speaker, 10 Watts, Crystal Clear Stereo Sound, Rich Bass, IPX5 Water Resistant, Loud Portable Bluetooth Speaker (Black)
  - `B07S764D9V`: Panasonic ErgoFit Wired Earbuds, In-Ear Headphones with Microphone and Call Controller, Ergonomic Custom-Fit Earpieces (S/M/L), 3.5mm Jack for Phones and Laptops - RP-TCM125-A (Blue)
  - `B07V6CGGW2`: R-fun AirPods Case Cover, Soft Silicone Protective Cover with Keychain for Women Men Compatible with Apple AirPods 2nd 1st Generation Charging Case, Front LED Visible-Pink Sand
- **Top Retreived by BM25 (Actual)**:
  1. `B0B3LXYHVH`: Active Noise Cancelling Headphones,Wireless Bluetooth Headphones Built-in Mic 40 Hours Playtime Wireless Noise Cancelling Headphone 3D Low Bass Tone Fast Charge for Cellphone/Work/Gym/TravelComputers
  2. `B0B9SC3MHY`: Noise Cancelling Headphones Bluetooth 5.0, 35Hrs Playtime Wireless Headphones Over-Ear w/Hi-Fi Stereo Deep Bass, CVC 8.0 Noise Cancelling, Bluetooth Headphones for Online Course Airplane Travel
  3. `B0BN8J8XD5`: Soundcore by Anker Life Q35 Multi Mode Active Noise Cancelling Headphones, Bluetooth Headphones with LDAC for Hi Res Wireless Audio, 40H Playtime, Comfortable Fit, Clear Calls, for Home, Work, Travel
  4. `B0BRZ2BFVK`: E7 Active Noise Cancelling Headphones Bluetooth Headphones Over Ear Wireless Headphones with Microphone Deep Bass, Comfortable Protein Earpads, 30HPlaytime for Travel/Work
  5. `B09G2F6G6L`: Vsonus H51 Active Noise Cancelling Headphones, Wireless Over Ear Bluetooth Headphones, Heavy Bass, BT 5.0, 30H Playtime, Metal Cover, Comfortable Protein Earpads for Travel Home Office
- **Root Cause Analysis**:
  The user query contains colloquial use-case terms (e.g. *'noise cancelling bluetooth headphones for travel'*). Relevant items emphasize technical specs without necessarily repeating the exact use-case descriptor, leading BM25 to score irrelevant accessory items with partial keyword overlap higher.

### Query `q_002`: *"mechanical keyboard for programming with quiet switches"*
- **Intent Type**: `attribute_use_case`
- **Identified Failure Category**: **Vocabulary Mismatch / Contextual Intent Blindness**
- **Ground-Truth Relevant Products (Expected)**:
  - `B07KG7BLFD`: MOSISO Compatible with MacBook Air 13 inch Case 2022, 2021-2018 Release A2337 M1 A2179 A1932 Retina Display Touch ID, Plastic Hard Shell&Keyboard Cover&Screen Protector&Storage Bag, Lavender Gray
  - `B077TPFXZV`: MOSISO Compatible with MacBook Air 13 inch Case (Models: A1369 & A1466, Older Version 2010-2017 Release), Protective Plastic Hard Shell Case & Keyboard Cover & Screen Protector, Deep Teal
  - `B07J38VMPN`: MOSISO Compatible with MacBook Pro 13 inch Case M2 2023, 2022, 2021-2016 A2338 M1 A2251 A2289 A2159 A1989 A1708 A1706, Plastic Hard Shell&Keyboard Cover&Screen Protector&Storage Bag, Capulet Olive
- **Top Retreived by BM25 (Actual)**:
  1. `B07D5JM51Q`: CORSAIR K70 LUX Mechanical Gaming Keyboard - Backlit Red LED - USB Passthrough & Media Controls - Linear & Quiet - Cherry MX Red (Renewed)
  2. `B01NAZ6RE3`: iKBC New Poker II Mechanical Keyboard with Cherry MX Black Switch, PBT Keycaps, Macro Programming, 6 DIP Switch, Win/Mac compatible, NKRO, Detachable USB Type-C Cable, Black Case, ANSI Layout
  3. `B0BFKCS22P`: Logitech MX Mechanical Mini for Mac Wireless Illuminated Keyboard, Low-Profile Performance Switches, Tactile Quiet Keys, Backlit, Bluetooth, USB-C, Apple, iPad - Space Grey
  4. `B09CTNCBFG`: Wireless 60% Mechanical Gaming Keyboard, 3 Connection Mode Keyboard with Linear Red Switches, Portabl Compact Grey&Black Keyboard, 68 Keys Mini Keyboard for Mac/Windows, Gamer,Typist,Travel,Trip
  5. `B094C5ZN1M`: luo Cherry MX RGB Switch Mute Pink Switch Mute Gray Switch Genuine German Cherry Switch (10PCS, Cherry RGB Mute Gray Switch)
- **Root Cause Analysis**:
  The user query contains colloquial use-case terms (e.g. *'mechanical keyboard for programming with quiet switches'*). Relevant items emphasize technical specs without necessarily repeating the exact use-case descriptor, leading BM25 to score irrelevant accessory items with partial keyword overlap higher.

### Query `q_003`: *"wireless earbuds with charging case for running"*
- **Intent Type**: `semantic_use_case`
- **Identified Failure Category**: **Vocabulary Mismatch / Contextual Intent Blindness**
- **Ground-Truth Relevant Products (Expected)**:
  - `B07S764D9V`: Panasonic ErgoFit Wired Earbuds, In-Ear Headphones with Microphone and Call Controller, Ergonomic Custom-Fit Earpieces (S/M/L), 3.5mm Jack for Phones and Laptops - RP-TCM125-A (Blue)
  - `B07V6CGGW2`: R-fun AirPods Case Cover, Soft Silicone Protective Cover with Keychain for Women Men Compatible with Apple AirPods 2nd 1st Generation Charging Case, Front LED Visible-Pink Sand
  - `B0BJCYP6H5`: BRG Compatible with Airpods Pro Case,Soft Silicone Skin Case Cover Shock-Absorbing Protective Case with Keychain [Front LED Visible] (Black)
- **Top Retreived by BM25 (Actual)**:
  1. `B07DD2RSSL`: Rowkin Ascent Charge True Wireless Earbuds Headphones: 50+ Hours, Bluetooth 5 Smallest Earphones & Qi Charging Case. Deep Bass Headset, Mic & Noise Reduction for Android Samsung & iPhone (Slate Gray)
  2. `B09W5DX5JH`: KOSETON E9 True Wireless Earbuds, Baby Blue – Wireless in-Ear Headphones for Running and Sport – Bluetooth Earbuds with a Comfortable, Secure Fit, 30 Hour Battery, Great Sound
  3. `B0B55Z76XM`: comiso Wireless Earbuds, True Wireless in Ear Bluetooth 5.0 with Microphone, Deep Bass, IPX7 Waterproof Loud Voice Sport Earphones with Charging Case for Outdoor Running Gym Workout (All Black)
  4. `B08Q85HL1X`: Headphone Sports Headset Earbuds Running
  5. `B0C6KGG6N7`: Wireless Earbuds Bluetooth Headphones 130Hrs Playtime with 2500mAh Wireless Charging Case LED Diaplay Hi-Fi Waterproof Over Ear Earphones for Sports Running Workout Gaming
- **Root Cause Analysis**:
  The user query contains colloquial use-case terms (e.g. *'wireless earbuds with charging case for running'*). Relevant items emphasize technical specs without necessarily repeating the exact use-case descriptor, leading BM25 to score irrelevant accessory items with partial keyword overlap higher.

### Query `q_004`: *"4K HDMI 2.1 cable high speed for gaming console"*
- **Intent Type**: `attribute_compatibility`
- **Identified Failure Category**: **Compatibility / Paraphrase Semantic Gap**
- **Ground-Truth Relevant Products (Expected)**:
  - `B0BGNG1294`: Amazon Basics HDMI Cable, 18Gbps High-Speed, 4K@60Hz, 2160p, Ethernet Ready, 10 Foot, Black
  - `B0756HHHL2`: Amazon Basics Nylon Braided Lightning to USB A Cable, MFi Certified Apple iPhone Charger, Silver, 3-Foot - Pack of 10
  - `B0BGS23YKX`: JSAUX USB-C to USB A Cable 3.1A Fast Charging [2-Pack 6.6ft], USB Type C Charger Cord Compatible with Samsung Galaxy S10 S9 S8 S20 Plus A51 A12 A11, Note 10 9 8, PS5 Controller USB C Charger-Green
- **Top Retreived by BM25 (Actual)**:
  1. `B09X4R7WH7`: Fusion8K White HDR HDMI 2.1 Cable Supports 8K @60Hz and 4K @120Hz Compatible with Dolby Vision and All TVs, BluRay, Xbox Series X, PS5 (10 Feet)
  2. `B0BH8BXTKK`: 8K HDMI Cable 15 FT, Capshi Ultra 48Gbps High Speed 15 FT HDMI Cable, Long HDMI Cable 15 FT -4K@120Hz 8K@150Hz, eARC, HDR10, DTS:X, HDCP 2.2 & 2.3, Compatible with PS4/5,Blu-ray, Monitor,PC and More
  3. `B0BKZT9QR3`: huaham 8K Fiber Optic HDMI Cable 25ft, 48Gbps Ultra High Speed HDMI 2.1 Cable 8K@60Hz 4K@120Hz, Support eARC RTX 3090 HDCP 2.2&2.3 Dolby Compatible with PS5, Xbox Series X, Roku/Fire/Sony/LG CX TV
  4. `B0C1NTC6M1`: BIFALE 8K HDMI Cable 10ft 2Pack, HDMI Cable 2.1 Support 8K@60Hz,4K@120Hz, Ultra-high Speed 48Gbps, Dynamic HDR, eARC Compatible with Apple TV, Switch, Xbox, PS4, Projector-3M2P
  5. `B09YFZ9SHC`: Pacorban Extension Cable (6ft 2pack) 8K HDMI 2.1 Male to Female HDMI Cable Ultra High Speed 8K 60Hz, 4K 120Hz, 3D Ultra HDR 48Gbps, eARC Dolby Atmos HDMI Extension
- **Root Cause Analysis**:
  The query specifies cross-device compatibility (e.g. *'4K HDMI 2.1 cable high speed for gaming console'*). BM25 fails to perform relational reasoning between the host device and peripheral product.

### Query `q_005`: *"USB C multiport hub adapter for MacBook Pro"*
- **Intent Type**: `compatibility_hardware`
- **Identified Failure Category**: **Compatibility / Paraphrase Semantic Gap**
- **Ground-Truth Relevant Products (Expected)**:
  - `B07PWCN4LC`: Syntech USB C to USB Adapter Pack of 2 USB C Male to USB3 Female Adapter Compatible with MacBook Pro 2021 iMac iPad Mini 6/Pro MacBook Air 2022 and Other Type C or Thunderbolt 4/3 Devices Midnight
  - `B0BRZPCQNJ`: SanDisk 64GB 2-Pack Ultra microSDXC UHS-I Memory Card (2x64GB) with Adapter - SDSQUAB-064G-GN6MT
  - `B07LG5WBTS`: Micro Center 64GB Class 10 MicroSDXC Flash Memory Card with Adapter for Mobile Device Storage Phone, Tablet, Drone & Full HD Video Recording - 80MB/s UHS-I, C10, U1 (1 Pack)
- **Top Retreived by BM25 (Actual)**:
  1. `B0BYSDFT45`: USB C Hub 10Gbps – Minisopuru USB C Hub for Laptop, USB C Hub with 100W Power Delivery, USB C to USB C Hub (Not Support Video), USB C Splitter, USB C Multiport Adapter for MacBook Air/Pro, etc.
  2. `B0BZ43P36B`: USB C Hub Docking Station - iDsonix 8 in 1 USB C Hub Aluminum Multiport Adapter with HDMI 4K@60Hz, PD 100W, 1Gigabit Ethernet, SD/TF Card Reader for MacBook Air/Pro iPad Dell/Hp Laptop and More
  3. `B0BZP7PJ9G`: Minisopuru 4 Ports USB C Hub – 10Gbps USB Hub for Laptop, USB Hub Multiport Adapter, USB C Adapter for MacBook Pro, MacBook, MacBook Air, iPad, Surface Pro, Chromebook. (No Charging/Video Transfer)
  4. `B0B72PSF14`: Minisopuru USB C Hub,7-in-1 USB-C Hub for Laptop & Tablet, USB C to USB C Multiport Adapter with 3 USB 3.0, 4K HDMI,100W Charging, SD/TF, USB C Dongle for MacBook Pro, Surface & More(Gun Black)
  5. `B08CV6P5BD`: USB C Hub HDMI Adapter, 8-in-1 USB C Adapter with 4K HDMI, 3 USB 3.0, 100W PD, VGA and Audio, USB C Multiport Adapter Hub for MacBook Pro 2019/2018/2017, MacBook Air, Dell XPS More USB Type C Devices
- **Root Cause Analysis**:
  The query specifies cross-device compatibility (e.g. *'USB C multiport hub adapter for MacBook Pro'*). BM25 fails to perform relational reasoning between the host device and peripheral product.

### Query `q_006`: *"portable bluetooth speaker waterproof with deep bass"*
- **Intent Type**: `semantic_attribute`
- **Identified Failure Category**: **Lexical Synonymy / Term Sparsity**
- **Ground-Truth Relevant Products (Expected)**:
  - `B0BW4PFM58`: OontZ Angle 3 Bluetooth Speaker, Portable Wireless Bluetooth 5.0 Speaker, 10 Watts, Crystal Clear Stereo Sound, Rich Bass, IPX5 Water Resistant, Loud Portable Bluetooth Speaker (Black)
  - `B07PGL2N7J`: Echo Dot (3rd Gen) - Smart speaker with Alexa - Sandstone
  - `B099V8GPR4`: JBL Flip 4, Black - Waterproof, Portable & Durable Bluetooth Speaker - Up to 12 Hours of Wireless Streaming - Includes Noise-Cancelling Speakerphone, Voice Assistant & JBL Connect+
- **Top Retreived by BM25 (Actual)**:
  1. `B0BY8N7HHG`: W-KING Bluetooth Speaker, 50W Powerful Bluetooth Speaker Loud IPX6 Waterproof, Large Outdoor Portable Speaker Wireless for Deep Bass/Bluetooth 5.0/Power Bank/40H Playtime/TF-Card/AUX/NFC/EQ (Black)
  2. `B0BNLMG9VT`: Speaqua – Bluetooth Speaker-Waterproof, Floatable, Portable Speaker Beach Accessory - Dual Portable Speakers Bluetooth Wireless Pairing - Removable Suction - Barnacle Vibe 2.0 (Manta Ray Black)
  3. `B09ZKW7DZV`: SFABF Bluetooth Speaker,Speakers,Outdoor, Portable,Waterproof,Wireless Speaker,Dual Pairing, 5.0,Loud Stereo,Booming Bass,300 Mins Playtime for Home,Party, Compatible with iOS/Android/PC
  4. `B09BCDY9TH`: MEGUO Bluetooth Speaker with Bass+, IPX7 Waterproof Wireless Portable Speaker, Outdoor Waterproof with TWS, HD Sound, Built in Mic, 24-Hour Playtime for Pool Beach Travel and More
  5. `B09CKDCZ4M`: Bluetooth Speaker, BOGASING M4 Speaker with 40W Stereo HD Surround Sound, Deeper Bass, 24H Playtime, IPX7 Waterproof, Bluetooth 5.0 TWS Wireless Dual Pairing Portable Speaker for Home, Outdoor (Black)
- **Root Cause Analysis**:
  Exact keyword overlap is insufficient to bridge lexical variance between query terminology and product metadata.


---

## 4. Key Takeaways & Motivation for Dense Semantic Retrieval (Phase 3)
1. **Vocabulary Mismatch**: BM25 requires exact token overlap and fails when customers search with colloquial intent rather than exact catalog keywords.
2. **Context Blindness**: Modifiers such as *"for travel"*, *"for running"*, or *"for programming"* dilutes the lexical score across irrelevant accessories containing those words.
3. **Control Condition Established**: This empirical BM25 benchmark serves as the rigorous control baseline for dense bi-encoder retrieval comparisons.
