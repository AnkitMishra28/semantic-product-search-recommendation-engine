# Dense Semantic Retrieval — Failure Diagnostic & Query-Level Comparison

*Generated on: 2026-08-14 11:28:03 UTC*
*Evaluation model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, normalized)*

---

## 1. Query-Level Performance Quadrant Distribution

| Quadrant | Count | Description |
| :--- | :--- | :--- |
| **Dense Succeeds & BM25 Fails** | **0** / 30 | Dense retrieval captures semantic intent where keyword match fails |
| **BM25 Succeeds & Dense Fails** | **4** / 30 | Strict exact keyword match is necessary |
| **Both Succeed** | **5** / 30 | Queries with clear keyword and semantic alignment |
| **Both Fail (Top 10)** | **21** / 30 | Complex constraint or ambiguous search intents |

---

## 2. Category Breakdown & Case Studies

### A. Dense Succeeds & BM25 Fails (Semantic Generalization Wins)
*None in this category.*

### B. BM25 Succeeds & Dense Fails (Lexical Specificity Wins)
#### Query `q_011`: *"laptop cooling pad with quiet fans"* (`category_use_case`)
- **Metrics**: BM25 R@10 = `0.12` (R@100 = `0.50`) vs Dense R@10 = `0.00` (R@100 = `0.12`)
- **Target Relevant**:
  -   - `B097RTX8R9`: Thermal Grizzly Kryonaut The High Performance Thermal Paste for Cooling All Processors, Graphics Cards and Heat Sinks in Computers and Consoles (1 Gram)
  - `B07S4KJV6W`: VersionTECH. Mini Handheld Fan, USB Desk Fan, Small Personal Portable Table Fan with USB Rechargeable Battery Operated Cooling Folding Electric Fan for Travel Office Room Household Black
- **Dense Top-3**:
  1. `B001JPRQWK`: USB Business Notebook Cooler Pad with 3 Built-in Fans Laptop Stand ALL Metal Construction
2. `B09PQJZ3BH`: Hiwings Laptop Cooling Pad, RGB Gaming Laptop Cooling Stand with 6 Quiet Cooling Fans, 6 Height Adjustable and Dual USB Ports for 11-16Inch Laptops (Extra Phone Stand) (Blue)
3. `B00LEFU2EY`: Superbpag USB Powered 5 Cooler Fan Laptop Cooling Pad for 12-17 inch Notebook with Adjustable Speed and Quantity Black
- **BM25 Top-3**:
  1. `B09PQJZ3BH`: Hiwings Laptop Cooling Pad, RGB Gaming Laptop Cooling Stand with 6 Quiet Cooling Fans, 6 Height Adjustable and Dual USB Ports for 11-16Inch Laptops (Extra Phone Stand) (Blue)
2. `B0BGGR6K4M`: Kootek Laptop Cooling Pad 12"-17" Cooler Pad Chill Mat 5 Quiet Fans LED Lights and 2 USB 2.0 Ports Adjustable Mounts Laptop Stand Height Angle, Red
3. `B004OBYZUM`: Gear Head Dual-Cool Laptop Cooling Fan - Silver/Black (USB) (ROHS) (CF3100SLV)

#### Query `q_016`: *"cat8 ethernet cable high speed for gigabit network router"* (`attribute_spec`)
- **Metrics**: BM25 R@10 = `0.12` (R@100 = `0.38`) vs Dense R@10 = `0.00` (R@100 = `0.50`)
- **Target Relevant**:
  -   - `B0BMQJYLQV`: Cat 6 Ethernet Cable 1 Ft (6Pack), Outdoor&Indoor, 10Gbps Support Cat7 Network, Heavy Duty Flat Internet LAN Patch Cord, Solid High Speed Weatherproof Cable for Router, Modem, Xbox, PS4, Switch, Black
  - `B09QX2TG87`: UGREEN Cat 7 Ethernet Cable Cat7 High Speed Flat Gigabit RJ45 LAN Cable 10Gbps Shielded Internet Network Patch Cord Compatible for Gaming PS5 PS4 PS3 Xbox PC Laptop Modem Router Computer 50FT
- **Dense Top-3**:
  1. `B0BWLTBC5S`: Cat 8 Ethernet Cable 20 ft,Internet Network LAN Cable,High Speed Gigabit Patch Cable 2000MHz 40Gbps with Gold Plated RJ45 Connector,Cat8 20 Feet Cable for Gaming/Xbox/Modem/Router/Switch
2. `B08FR4G1JZ`: Cat8 Ethernet Cable, Network Cable 60ft, High Speed 40Gbps 2000MHz SFTP Patch Cord, Cat8 Network Cable, Gold Plated RJ45 Connectors, for Router, Modem, Gaming, PC, Xbox, Laptop - White
3. `B08DTXW5K2`: CAT 8 Ethernet Cable, 1.5ft (5 Pack) Ultra High Speed 40Gbps 2000MHz SFTP 26AWG CAT8 Cable LAN Internet Network Cord with Gold Plated RJ45 Connector for Gaming, Router, Modems, PC (1.5ft/5 Pack/Black)
- **BM25 Top-3**:
  1. `B096MD1Q1Q`: Cat 8 Ethernet Cable,50FT Yurnero Gigabit High Speed Cat8 Network Cable 40Gbps/2000Mhz RJ51 Connector Ethernet Cord with Gold Plated SFTP LAN Cable for Gaming/Ethernet Switch/Modem/Router/Xbox
2. `B083ZRZVPY`: CableCreation Flat Cat8 Ethernet Cord Long, 40G High Speed Slim Network LAN Cable Cord Gigabit Internet Router Cable RJ45 Wire for Computer Laptop PS5 PS4, Switch Box PC,TV Box, 16.6ft/5m, Black
3. `B09ZHXLTWS`: Cat8 Ethernet Cable 50ft Shielded,High Speed 40Gbps 2000MHz SSTP Flat Internet Network LAN Cable with Gold Plated RJ45 Connector for Game Console,PS5,Xbox,Router,PC,Laptop,Modem,Switches,Hub...

#### Query `q_018`: *"magnetic wireless car charger mount for iPhone"* (`compatibility_use_case`)
- **Metrics**: BM25 R@10 = `0.12` (R@100 = `0.38`) vs Dense R@10 = `0.00` (R@100 = `0.25`)
- **Target Relevant**:
  -   - `B08TX9GNB3`: Anker Roav SmartCharge F0 Bluetooth FM Transmitter for Car, Audio Adapter and Receiver, Hands-Free Calling, MP3 Car Charger with 2 USB Ports, PowerIQ, and AUX Output (No Dedicated App)
  - `B08LD9NGQ2`: DBPOWER 12" Portable DVD Player with 5-Hour Rechargeable Battery, 10" Swivel Display Screen, SD/USB Port, with 1.8 Meter Car Charger, Power Adaptor and Car Headrest Mount, Region Free- Blue
- **Dense Top-3**:
  1. `B0BJVTD6VK`: Magnetic Phone Holder for Laptop, Car Phone Holder Mount for Tesla Model 3 Y, Magnetic Phone Mount Rotation Foldaway Invisible Phone Bracket for Car Screen/Dashboard, Laptop Phone Holder - Black
2. `B06XYZDS5N`: Cell Phone Car Mount, Magnetic, Universal, Black
3. `B0C1LBWM4Q`: Scosche MAGDM2 MagicMount Magnetic Car Phone Holder Mount - 360 Degree Adjustable Head, Universal with All Devices - Dashboard Mount
- **BM25 Top-3**:
  1. `B072QC8ZQ2`: SoundBot SB360 LITE Bluetooth Wireless 4.0 Car Kit Hands-Free Wireless Talking & Music Streaming Dongle w/ Magnetic Mounts + Built-in 3.5mm Aux Cable
2. `B00PTLNZJI`: Evecase 10.6~11.6 inch Tablet/Laptop/Chromebook/MacBook Multi-Functional Neoprene Messenger Case Tote Bag with Handle and Carrying Strap (Purple)
3. `B01N31AQ68`: SoundBot SB361 FM Radio Wireless Car Kit SB361-BLK/BLK

#### Query `q_025`: *"external DVD drive USB 3.0 portable optical drive"* (`attribute_compatibility`)
- **Metrics**: BM25 R@10 = `0.12` (R@100 = `0.62`) vs Dense R@10 = `0.00` (R@100 = `0.75`)
- **Target Relevant**:
  -   - `B007IIM14A`: TOP CASE - 2 in 1 Signature Bundle Rubberized Hard Case and Keyboard Cover Compatible Old Generation MacBook Pro 13" with DVD Drive/CD-ROM Model: A1278 - Grey
  - `B00W9GC1AK`: Samsung USB 2.0 Ultra Portable External DVD Writer Model SE-218CB/RSBS
- **Dense Top-3**:
  1. `B0886JZY2K`: USB 3.0 Type A and Type-C 2-in-1 External DVD CD Player Burner, for HP Dell Asus Acer Lenovo Alienware MSI Sony Laptop & Desktop Computer, Pop-up Mobile Portable 8X DVD+-R/RW DL 24X CD-R Optical Drive
2. `B0B4Z5K9BZ`: External DVD Drive, Type A/Type C USB3.0 5Gbps Optical Drive Case, 9.5/12.7MM Mobile USB 3.0 CD ROM Optical Player Drive for Laptop Desktop PC for Window
3. `B07L58GSDF`: External DVD Drive, M Way USB 3.0 Type C CD Drive, Dual Port DVD Player, Portable Optical Burner Writer Rewriter, High Speed Data Transfer for Laptop Notebook Desktop PC MAC OS Windows 7/8/10
- **BM25 Top-3**:
  1. `B0B9XMWFLN`: External CD/DVD Drive for Laptop, USB 3.0 CD Burner Portable CD/DVD Optical Drive Player Reader Writer, Compatible with Laptop Desktop PC MacBook Mac Windows Linux OS (Black)
2. `B0BN5MPWV7`: External CD/DVD Drive for Laptop, 7 IN 1 USB 3.0 Ultra-Slim Portable DVD Player, CD ROM Burner Writer External Disk Drive Optical Compatible with Laptop Desktop PC MacBook Windows Mac Linux OS
3. `B0B6NPLY8P`: OWC Mercury Pro 5.25" Optical Drive External Enclosure (NO Drive)


### C. Both Succeed
#### Query `q_007`: *"high capacity power bank fast charging 20000mAh"* (`attribute_spec`)
- **Metrics**: BM25 R@10 = `0.12` (R@100 = `0.25`) vs Dense R@10 = `0.12` (R@100 = `0.12`)
- **Target Relevant**:
  -   - `B0BY8N7HHG`: W-KING Bluetooth Speaker, 50W Powerful Bluetooth Speaker Loud IPX6 Waterproof, Large Outdoor Portable Speaker Wireless for Deep Bass/Bluetooth 5.0/Power Bank/40H Playtime/TF-Card/AUX/NFC/EQ (Black)
  - `B07NV3LSZJ`: Limoss Battery Pack for Reclining Furniture with Charger - Rechargeable Power Pack for Power Sofas/Loveseats/Lift Chairs/Recliners/Sectionals - Recliner Battery Pack - Ashley - Flexstell - ZB-B1800
- **Dense Top-3**:
  1. `B0B4DN17ND`: ONLYNEW 32000mah Power Bank - Portable Charger White...
2. `B01GMZ5NB2`: Race Sport RS03JUMP 12V 6000mAH Travel Size Mini Power Bank
3. `B07T64VNFF`: Renogy 45000mAh/162WH 120W Max AC Outlet Laptop Power Bank, Portable Laptop Charger with 3-Prong AC Outlet, 3 USB Ports and Flashlight, High Capacity Power Bank Compatible with MacBook
- **BM25 Top-3**:
  1. `B0B7JXCH53`: AirPods Strap with Smart Portable Power Supply for AirPods Pro 1, Airpods 1/2/3, Anti-Lost Neck Rope with Charger Power Bank (Not for AirPods Pro 2)
2. `B0BHY8TMT7`: JBL Pulse 4 - Waterproof Portable Bluetooth Speaker with Light Show and InfinityLab InstantGo 10000mAh Wireless Power Bank (White)
3. `B07MDVXVV2`: Rockville TRuRock Bluetooth EarBuds Earphones+Mic For Google Pixel 3 XL Phone

#### Query `q_017`: *"dual monitor arm desk mount adjustable"* (`category_attribute`)
- **Metrics**: BM25 R@10 = `0.12` (R@100 = `0.62`) vs Dense R@10 = `0.12` (R@100 = `0.88`)
- **Target Relevant**:
  -   - `B09ZBJKPX1`: VIVO Extra Tall Single Monitor Desk Mount Stand 39 inch Pole, Features Full Adjustability - Tilt and Articulation, Holds 13 to 32 inch Screens up to 22 lbs with VESA Mounting, White, STAND-V011W
  - `B07VSDF7XT`: Pipishell 25 Inch Webcam Stand - Flexible Desk Mount Clamp Gooseneck Stand for Logitech Webcam C930e,C930,C920, C922x,C922, Brio 4K, C925e,C615-PIWS01
- **Dense Top-3**:
  1. `B085ZVFTLR`: EleTab Dual Monitor Stand - Heavy Duty Dual Arm Monitor Desk Mount Fully Adjustable, Fit 2/Two LCD Screens up to 27 Inch with C-Clamp and Grommet Base, Gray
2. `B08BHK3TFX`: DESINO Dual Monitor Stand- Fully Adjustable Monitor Arm Desk Mount Heavy Duty Fit 2/Two LCD Screens up to 32 Inch, Mechanical Warfare Red
3. `B0017DINGC`: Dualswing Arm Desk Mount
- **BM25 Top-3**:
  1. `B09LS4XJBL`: Joy Seeker Dual Monitor Mount for 15 to 27 Inch,Height Adjustable Monitor Stands for 2 Monitors,Gas Spring Full Motion Swivel Computer Monitor Arm Each Arm Holds up to 14.3lbs
2. `B0BCHP31HD`: Mount-It! Stand Up Workstation with Dual Monitor Mount - Standing Desk Converter with Height Adjustable Keyboard & Counterbalance Monitor Arm
3. `B09SLMD243`: Joy Seeker Dual Monitor Arms Stands, Fully Adjustable Double Monitor Desk Mount with USB, Gas Spring Computer Mount for 2 Screens Fits 15"-27" Max 17.6lbs Per Arm with Clamp, 75x75/100x100mm, Black

#### Query `q_019`: *"bluetooth audio transmitter receiver for TV and airplane"* (`long_tail_intent`)
- **Metrics**: BM25 R@10 = `0.25` (R@100 = `0.50`) vs Dense R@10 = `0.25` (R@100 = `0.25`)
- **Target Relevant**:
  -   - `B09NWGVKZ3`: Avantree DG45 USB Bluetooth Adapter for PC, 5.0 Bluetooth Dongle for PC Computer Desktop Laptop, Wireless Transfer for Bluetooth Headphones Speakers Keyboard Mouse Printers Windows 11/10/8.1/8
  - `B0C33Z5X4B`: USB Bluetooth Adapter 5.3 for Desktop PC, Plug & Play Mini Bluetooth EDR Dongle Receiver & Transmitter only for Laptop Computer Headphones Keyboard Mouse Speakers Printer Windows 11/10/8.1-Grey
- **Dense Top-3**:
  1. `B081T1T73S`: ABS Bluetooth Audio Transmitter TV PC Bluetooth 3.5mm Audio Adapter Transmitter Receiver for TV/Computer/Smartphone
2. `B08H8RKM5D`: [2020 Upgraded] 1Mii Long Range Bluetooth Transmitter Receiver Bluetooth Audio Adapter for TV PC Home Stereo BT Headphones with aptX Low Latency HiFi Sound, Optical RCA AUX 3.5mm - B03 (Renewed)
3. `B0BKSNF4G8`: YMOO Bluetooth 5.3 Transmitter Receiver for TV to 2 Wireless Headphones, 3.5mm Jack in-Flight Bluetooth Audio Adapter for Airplane, Dual Link AptX Adaptive/Low Latency/HD Audio for Home Stereo/PC/Gym
- **BM25 Top-3**:
  1. `B01LTH9QHU`: LyxPro Bluetooth Audio Transmitter for TV, PC, Music, MP3 Player. Small Wireless Portable 3.5mm Audio Bluetooth Dongle, A2DP Stereo Music Transmission
2. `B0BYRS57TC`: Bluetooth Transmitter Receiver for TV Wireless Headphones, ZDMYY 2-in-1 Bluetooth AUX Adapter Pairs 2 Devices, Bluetooth Audio Adapter for Home Stereo/Car/Airplane/Boat/Gym/MP3/MP4/DVD, Built-in Mic
3. `B0BKSNF4G8`: YMOO Bluetooth 5.3 Transmitter Receiver for TV to 2 Wireless Headphones, 3.5mm Jack in-Flight Bluetooth Audio Adapter for Airplane, Dual Link AptX Adaptive/Low Latency/HD Audio for Home Stereo/PC/Gym

#### Query `q_023`: *"stylus pen for touch screen tablet high precision"* (`category_compatibility`)
- **Metrics**: BM25 R@10 = `0.12` (R@100 = `0.25`) vs Dense R@10 = `0.12` (R@100 = `0.62`)
- **Target Relevant**:
  -   - `B0BR3MLGZZ`: Stylus Pen for iPad 9th&10th Generation-2X Fast Charge Active Pencil Compatible with 2018-2023 Apple iPad Pro11&12.9 inch, iPad Air 3/4/5,iPad 6-10,iPad Mini 5/6 Gen-Orange
  - `B09S32P52Q`: XPPen Artist12 Pro 11.6 Inch Drawing Monitor Pen Display Full-Laminated Graphics Drawing Tablet with Tilt Function Battery-Free Stylus and 8 Shortcut Keys(8192 Levels Pen Pressure and 72% NTSC)
- **Dense Top-3**:
  1. `B07FY7DJQ8`: Autones Resistive Hard Tip Stylus Pen for Resistance Touch Screen Game Player Tablet
2. `B07G98G3BV`: Targus Stylus for iPad, iPhone, iPod, Samsung Tablets, Smartphones and Other Touchscreen Devices, Black (AMM01US)
3. `B09L4L3DVC`: Pen for Surface,Pressure Sensitive Stylus Pen for Microsoft Surface ProX/6/5/4/3, Go2/Go1, Surface3, Surface Laptop3/2/1, Surface Studio2/1, Surface Book3/2/1, Right-Click, Silver
- **BM25 Top-3**:
  1. `B07FY7DJQ8`: Autones Resistive Hard Tip Stylus Pen for Resistance Touch Screen Game Player Tablet
2. `B09TR2LGBZ`: Stylus Pen for iPad with Palm Rejection,[2 Pack] Paper Screen Protector Compatible with iPad 9th/8th/7th Generation (2021/2020/2019 Model)
3. `B00AHWL9R4`: Adonit Jot Pro Fine Point Precision Stylus for iPad, iPhone, Android, Kindle, Samsung, and Windows Tablets - Gun Metal [Previous Generation]

#### Query `q_028`: *"wifi range extender booster for home coverage"* (`category_use_case`)
- **Metrics**: BM25 R@10 = `0.38` (R@100 = `1.00`) vs Dense R@10 = `0.38` (R@100 = `1.00`)
- **Target Relevant**:
  -   - `B087D7C96F`: RANGEXTD WiFi Extender with Ethernet Port - WiFi Signal Amplifier Increases Home WiFi Coverage | Single-Band 2.4GHz WiFi Range Booster Fixes Dead Spots | Up to 300Mbps Speed, 10 Devices
  - `B01N1H6MS0`: Amped Wireless High Power Wireless-N 600mW Smart Repeater and Range Extender (SR10000),Black
- **Dense Top-3**:
  1. `B087D7C96F`: RANGEXTD WiFi Extender with Ethernet Port - WiFi Signal Amplifier Increases Home WiFi Coverage | Single-Band 2.4GHz WiFi Range Booster Fixes Dead Spots | Up to 300Mbps Speed, 10 Devices
2. `B0BSFLWVRL`: WiFi Extender WiFi Booster with Ethernet Port, 2022 Upgraded 1200mbps Wi-Fi Range Extender, WiFi Extenders Signal Booster for Home - Covers Up to 7800 Sq.ft & 35 Devices, 1-Tap Setup Internet Booster
3. `B0C6LCJ95V`: WiFi Extenders Signal Booster for Home, WiFi Extender 300Mbps, WiFi Booster, Wireless Signal Range Amplifier Covers Up to 3000 Sq.ft, WiFi Repeater, Internet Booster, Internet Extender
- **BM25 Top-3**:
  1. `B087D7C96F`: RANGEXTD WiFi Extender with Ethernet Port - WiFi Signal Amplifier Increases Home WiFi Coverage | Single-Band 2.4GHz WiFi Range Booster Fixes Dead Spots | Up to 300Mbps Speed, 10 Devices
2. `B0BSFLWVRL`: WiFi Extender WiFi Booster with Ethernet Port, 2022 Upgraded 1200mbps Wi-Fi Range Extender, WiFi Extenders Signal Booster for Home - Covers Up to 7800 Sq.ft & 35 Devices, 1-Tap Setup Internet Booster
3. `B093KHD7XZ`: TGVi's WiFi Extender 1200Mbps,WiFi Range Extender 2.4 & 5GHz Dual Band,WiFi Extender with Ethernet Port, WiFi Extenders Signal Booster for Home, 360 Degree Wireless Network Signal Coverage White


### D. Both Fail at Top 10 (Challenging / Constraint Queries)
#### Query `q_001`: *"noise cancelling bluetooth headphones for travel"* (`semantic_use_case`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B0BW4PFM58`: OontZ Angle 3 Bluetooth Speaker, Portable Wireless Bluetooth 5.0 Speaker, 10 Watts, Crystal Clear Stereo Sound, Rich Bass, IPX5 Water Resistant, Loud Portable Bluetooth Speaker (Black)
  - `B07S764D9V`: Panasonic ErgoFit Wired Earbuds, In-Ear Headphones with Microphone and Call Controller, Ergonomic Custom-Fit Earpieces (S/M/L), 3.5mm Jack for Phones and Laptops - RP-TCM125-A (Blue)
- **Dense Top-3**:
  1. `B0B9SC3MHY`: Noise Cancelling Headphones Bluetooth 5.0, 35Hrs Playtime Wireless Headphones Over-Ear w/Hi-Fi Stereo Deep Bass, CVC 8.0 Noise Cancelling, Bluetooth Headphones for Online Course Airplane Travel
2. `B0B3LXYHVH`: Active Noise Cancelling Headphones,Wireless Bluetooth Headphones Built-in Mic 40 Hours Playtime Wireless Noise Cancelling Headphone 3D Low Bass Tone Fast Charge for Cellphone/Work/Gym/TravelComputers
3. `B0BRZ2BFVK`: E7 Active Noise Cancelling Headphones Bluetooth Headphones Over Ear Wireless Headphones with Microphone Deep Bass, Comfortable Protein Earpads, 30HPlaytime for Travel/Work
- **BM25 Top-3**:
  1. `B0B3LXYHVH`: Active Noise Cancelling Headphones,Wireless Bluetooth Headphones Built-in Mic 40 Hours Playtime Wireless Noise Cancelling Headphone 3D Low Bass Tone Fast Charge for Cellphone/Work/Gym/TravelComputers
2. `B0B9SC3MHY`: Noise Cancelling Headphones Bluetooth 5.0, 35Hrs Playtime Wireless Headphones Over-Ear w/Hi-Fi Stereo Deep Bass, CVC 8.0 Noise Cancelling, Bluetooth Headphones for Online Course Airplane Travel
3. `B0BN8J8XD5`: Soundcore by Anker Life Q35 Multi Mode Active Noise Cancelling Headphones, Bluetooth Headphones with LDAC for Hi Res Wireless Audio, 40H Playtime, Comfortable Fit, Clear Calls, for Home, Work, Travel

#### Query `q_002`: *"mechanical keyboard for programming with quiet switches"* (`attribute_use_case`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B07KG7BLFD`: MOSISO Compatible with MacBook Air 13 inch Case 2022, 2021-2018 Release A2337 M1 A2179 A1932 Retina Display Touch ID, Plastic Hard Shell&Keyboard Cover&Screen Protector&Storage Bag, Lavender Gray
  - `B077TPFXZV`: MOSISO Compatible with MacBook Air 13 inch Case (Models: A1369 & A1466, Older Version 2010-2017 Release), Protective Plastic Hard Shell Case & Keyboard Cover & Screen Protector, Deep Teal
- **Dense Top-3**:
  1. `B000UC1W3C`: Ione Scorpius M10 USB Mechanical Keyswitch Keyboard
2. `B09W9LV8SR`: DANSHER Percent 60% Mechanical Gaming Keyboard,81Keys Wired Mechanical Keyboard LED Backlit Compact Office Keyboard with Blue Switch for Windows Laptop PC Mac Xbox
3. `B01MS8YTYX`: VELOCIFIRE TKL01 Wired Mechanical Keyboard 87-Key Tenkeyless with Brown Switches, ICY Blue LED Backlit for Copywriters, Gamers, and Programmers
- **BM25 Top-3**:
  1. `B07D5JM51Q`: CORSAIR K70 LUX Mechanical Gaming Keyboard - Backlit Red LED - USB Passthrough & Media Controls - Linear & Quiet - Cherry MX Red (Renewed)
2. `B01NAZ6RE3`: iKBC New Poker II Mechanical Keyboard with Cherry MX Black Switch, PBT Keycaps, Macro Programming, 6 DIP Switch, Win/Mac compatible, NKRO, Detachable USB Type-C Cable, Black Case, ANSI Layout
3. `B0BFKCS22P`: Logitech MX Mechanical Mini for Mac Wireless Illuminated Keyboard, Low-Profile Performance Switches, Tactile Quiet Keys, Backlit, Bluetooth, USB-C, Apple, iPad - Space Grey

#### Query `q_003`: *"wireless earbuds with charging case for running"* (`semantic_use_case`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.12`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B07S764D9V`: Panasonic ErgoFit Wired Earbuds, In-Ear Headphones with Microphone and Call Controller, Ergonomic Custom-Fit Earpieces (S/M/L), 3.5mm Jack for Phones and Laptops - RP-TCM125-A (Blue)
  - `B07V6CGGW2`: R-fun AirPods Case Cover, Soft Silicone Protective Cover with Keychain for Women Men Compatible with Apple AirPods 2nd 1st Generation Charging Case, Front LED Visible-Pink Sand
- **Dense Top-3**:
  1. `B0C6KGG6N7`: Wireless Earbuds Bluetooth Headphones 130Hrs Playtime with 2500mAh Wireless Charging Case LED Diaplay Hi-Fi Waterproof Over Ear Earphones for Sports Running Workout Gaming
2. `B07M5S4SNX`: SOAIY Wireless Earbuds,True Bluetooth 5.0 Sports Earbuds- Waterproof & Sweatproof, Build-in Microphone, Auto Paring and Hi-Fi 3D Stereo, Unique Charging Box with Digital Power Display.
3. `B08395BMY2`: Wireless Bluetooth Earbuds,Smart Touch Wireless Headphones Bluetooth 5.0 3D Stereo Bluetooth Headsets in-Ear with Microphone Charging Case Sweatproof Noise Cancelling
- **BM25 Top-3**:
  1. `B07DD2RSSL`: Rowkin Ascent Charge True Wireless Earbuds Headphones: 50+ Hours, Bluetooth 5 Smallest Earphones & Qi Charging Case. Deep Bass Headset, Mic & Noise Reduction for Android Samsung & iPhone (Slate Gray)
2. `B09W5DX5JH`: KOSETON E9 True Wireless Earbuds, Baby Blue – Wireless in-Ear Headphones for Running and Sport – Bluetooth Earbuds with a Comfortable, Secure Fit, 30 Hour Battery, Great Sound
3. `B0B55Z76XM`: comiso Wireless Earbuds, True Wireless in Ear Bluetooth 5.0 with Microphone, Deep Bass, IPX7 Waterproof Loud Voice Sport Earphones with Charging Case for Outdoor Running Gym Workout (All Black)

#### Query `q_004`: *"4K HDMI 2.1 cable high speed for gaming console"* (`attribute_compatibility`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B0BGNG1294`: Amazon Basics HDMI Cable, 18Gbps High-Speed, 4K@60Hz, 2160p, Ethernet Ready, 10 Foot, Black
  - `B0756HHHL2`: Amazon Basics Nylon Braided Lightning to USB A Cable, MFi Certified Apple iPhone Charger, Silver, 3-Foot - Pack of 10
- **Dense Top-3**:
  1. `B08Q3DRTXB`: EONZONE 3.3ft 8K HDMI Cable 2.1 Ultra HD High Speed 48Gpbs 8K60 4K120 144Hz eARC HDR10 HDCP 2.2&2.3 Compatible for Xbox One X PS4 PS5 Roku Fire TV Apple TV Nintendo Switch Sony LG Samsung
2. `B08MXPWJ1Y`: 8K HDMI 2.1 Cable 3Ft,ALLEASA Ultra High Speed 8K@60Hz,4K@120Hz@144Hz DSC,HDR UHD 7680×4320,eARC HDR10+,HDCP 2.2&2.3,Compatible with PS5/PS4/PS3(Black)
3. `B08HYMHBH7`: POLOK 4K HDMI Cable 15ft Prime,HDR HDMI Cable 4K 2.0b,HDMI Cord Braided,18Gbps High Speed Certified,Ethernet,4K Ultra HD,3D HDCP2.2 Audio Return(ARC) CEC for HDTV PC 4K Fire TV Gaming PS4 Monitor,etc
- **BM25 Top-3**:
  1. `B09X4R7WH7`: Fusion8K White HDR HDMI 2.1 Cable Supports 8K @60Hz and 4K @120Hz Compatible with Dolby Vision and All TVs, BluRay, Xbox Series X, PS5 (10 Feet)
2. `B0BH8BXTKK`: 8K HDMI Cable 15 FT, Capshi Ultra 48Gbps High Speed 15 FT HDMI Cable, Long HDMI Cable 15 FT -4K@120Hz 8K@150Hz, eARC, HDR10, DTS:X, HDCP 2.2 & 2.3, Compatible with PS4/5,Blu-ray, Monitor,PC and More
3. `B0BKZT9QR3`: huaham 8K Fiber Optic HDMI Cable 25ft, 48Gbps Ultra High Speed HDMI 2.1 Cable 8K@60Hz 4K@120Hz, Support eARC RTX 3090 HDCP 2.2&2.3 Dolby Compatible with PS5, Xbox Series X, Roku/Fire/Sony/LG CX TV

#### Query `q_005`: *"USB C multiport hub adapter for MacBook Pro"* (`compatibility_hardware`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B07PWCN4LC`: Syntech USB C to USB Adapter Pack of 2 USB C Male to USB3 Female Adapter Compatible with MacBook Pro 2021 iMac iPad Mini 6/Pro MacBook Air 2022 and Other Type C or Thunderbolt 4/3 Devices Midnight
  - `B0BRZPCQNJ`: SanDisk 64GB 2-Pack Ultra microSDXC UHS-I Memory Card (2x64GB) with Adapter - SDSQUAB-064G-GN6MT
- **Dense Top-3**:
  1. `B0B72PSF14`: Minisopuru USB C Hub,7-in-1 USB-C Hub for Laptop & Tablet, USB C to USB C Multiport Adapter with 3 USB 3.0, 4K HDMI,100W Charging, SD/TF, USB C Dongle for MacBook Pro, Surface & More(Gun Black)
2. `B07SX5TBV2`: USB C Hub Multiport Adapter - iDriveTech 10 in 1. 4x USB Ports, HDMI and VGA ports, SD/TF Card Reader, Gigabit Ethernet port, USB-C port. Compatible MacBook Pro and more Type C Devices (Thunderbolt 3)
3. `B08P39D53Y`: USB C Hub 7 in 1 Multiport Adapter with HDMI Port,3 USB 3.0 Ports,100W USB-C PD Charging Port and SD/TF Docking Station for MacBook Pro/Air and Other USB-C Laptops
- **BM25 Top-3**:
  1. `B0BYSDFT45`: USB C Hub 10Gbps – Minisopuru USB C Hub for Laptop, USB C Hub with 100W Power Delivery, USB C to USB C Hub (Not Support Video), USB C Splitter, USB C Multiport Adapter for MacBook Air/Pro, etc.
2. `B0BZ43P36B`: USB C Hub Docking Station - iDsonix 8 in 1 USB C Hub Aluminum Multiport Adapter with HDMI 4K@60Hz, PD 100W, 1Gigabit Ethernet, SD/TF Card Reader for MacBook Air/Pro iPad Dell/Hp Laptop and More
3. `B0BZP7PJ9G`: Minisopuru 4 Ports USB C Hub – 10Gbps USB Hub for Laptop, USB Hub Multiport Adapter, USB C Adapter for MacBook Pro, MacBook, MacBook Air, iPad, Surface Pro, Chromebook. (No Charging/Video Transfer)

#### Query `q_006`: *"portable bluetooth speaker waterproof with deep bass"* (`semantic_attribute`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.25`) vs Dense R@10 = `0.00` (R@100 = `0.25`)
- **Target Relevant**:
  -   - `B0BW4PFM58`: OontZ Angle 3 Bluetooth Speaker, Portable Wireless Bluetooth 5.0 Speaker, 10 Watts, Crystal Clear Stereo Sound, Rich Bass, IPX5 Water Resistant, Loud Portable Bluetooth Speaker (Black)
  - `B07PGL2N7J`: Echo Dot (3rd Gen) - Smart speaker with Alexa - Sandstone
- **Dense Top-3**:
  1. `B0899SX15V`: BassPal Bluetooth Speaker IPX7 Waterproof Shower Speakers Bluetooth Wireless Shower Radio with 10W 15 Hours Playtime, TWS, Loud Stereo Sound for Shower Hiking Camping (Blue)
2. `B08YKK29L5`: Ultimate Ears WONDERBOOM Portable Waterproof Bluetooth Speaker - Patches
3. `B088B9Q1QM`: INSMY Portable Bluetooth Speakers, IPX7 Waterproof Floating 20W Wireless Speaker Loud Sound Rich Bass, Stereo Pairing Max 40W, 24 Hours Bluetooth 5.0 Built-in Mic for Outdoors Camping Pool (Blue)
- **BM25 Top-3**:
  1. `B0BY8N7HHG`: W-KING Bluetooth Speaker, 50W Powerful Bluetooth Speaker Loud IPX6 Waterproof, Large Outdoor Portable Speaker Wireless for Deep Bass/Bluetooth 5.0/Power Bank/40H Playtime/TF-Card/AUX/NFC/EQ (Black)
2. `B0BNLMG9VT`: Speaqua – Bluetooth Speaker-Waterproof, Floatable, Portable Speaker Beach Accessory - Dual Portable Speakers Bluetooth Wireless Pairing - Removable Suction - Barnacle Vibe 2.0 (Manta Ray Black)
3. `B09ZKW7DZV`: SFABF Bluetooth Speaker,Speakers,Outdoor, Portable,Waterproof,Wireless Speaker,Dual Pairing, 5.0,Loud Stereo,Booming Bass,300 Mins Playtime for Home,Party, Compatible with iOS/Android/PC

#### Query `q_008`: *"ergonomic wireless vertical mouse for wrist pain"* (`semantic_use_case`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B0BW4PFM58`: OontZ Angle 3 Bluetooth Speaker, Portable Wireless Bluetooth 5.0 Speaker, 10 Watts, Crystal Clear Stereo Sound, Rich Bass, IPX5 Water Resistant, Loud Portable Bluetooth Speaker (Black)
  - `B07S764D9V`: Panasonic ErgoFit Wired Earbuds, In-Ear Headphones with Microphone and Call Controller, Ergonomic Custom-Fit Earpieces (S/M/L), 3.5mm Jack for Phones and Laptops - RP-TCM125-A (Blue)
- **Dense Top-3**:
  1. `B081DJHGX2`: Wireless Ergonomic Mouse, Jelly Comb Rechargeable 2.4G Bluetooth Vertical Mouse Switch to 3 Devices Optical Mice with 6 Buttons 3 Adjustable DPI Levels for Laptop, PC, MacBook, Notebook
2. `B07G4SQQH2`: DeLUX Ergonomic Wireless Vertical Silent Mouse - 2.4G USB Receiver, 3 DPI Levels (800/1200/1600), 6 Buttons, Removable Wrist Rest for Laptop PC (M618Plus Wireless-Black)
3. `B09PR9QJ2B`: Wireless Ergonomic Mouse 2.4G Rechargeable Optical Vertical Mice with USB Receiver, 800/1200 /1600 DPI, 5 Buttons for Laptop, Desktop, PC, MacBook - White
- **BM25 Top-3**:
  1. `B09PR9QJ2B`: Wireless Ergonomic Mouse 2.4G Rechargeable Optical Vertical Mice with USB Receiver, 800/1200 /1600 DPI, 5 Buttons for Laptop, Desktop, PC, MacBook - White
2. `B00GN0WQBW`: Adesso iMouseE1 - Vertical Ergonomic Illuminated Optical 6-Button USB Mouse - Right Hand Orientation, Black
3. `B07G4SQQH2`: DeLUX Ergonomic Wireless Vertical Silent Mouse - 2.4G USB Receiver, 3 DPI Levels (800/1200/1600), 6 Buttons, Removable Wrist Rest for Laptop PC (M618Plus Wireless-Black)

#### Query `q_009`: *"gaming headset with microphone for PC and console"* (`category_use_case`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B07S764D9V`: Panasonic ErgoFit Wired Earbuds, In-Ear Headphones with Microphone and Call Controller, Ergonomic Custom-Fit Earpieces (S/M/L), 3.5mm Jack for Phones and Laptops - RP-TCM125-A (Blue)
  - `B0C337TNGS`: Beats Powerbeats Pro Wireless Earbuds - Apple H1 Headphone Chip, Class 1 Bluetooth Headphones, 9 Hours of Listening Time, Sweat Resistant, Built-in Microphone - Ivory
- **Dense Top-3**:
  1. `B08938RDSV`: RGB LED Gaming Headset PS4 Headset Gamer Headset for Xbox one Headset Gaming Headphone with Surround Sound Noise Canceling Microphone
2. `B01ASTA6Q0`: Sades Over-Ear Stereo Gaming Headsets Headphones with Microphone for Plastation4/New Xbox One/PC Computers/Mac/Tablets/Phones (Black)
3. `B07GVM74BS`: Wired Gaming Headset Headphones with Microphone for PS4 PC Laptop Phone
- **BM25 Top-3**:
  1. `B01DNCFM72`: Ailihen K6 Headset with Microphone Over Ear Headphones for Computer PC Laptop with in-Line Volume Control(Black/Blue)
2. `B0922H76XN`: T2 Gaming Headset USB Over-Ear Headphone Wired with Cool RGB Light, 3.5MM Surround Sound Gaming Headphones for PS5, PS4, Switch, PC, PS2, Mac, Laptop, Xbox One, Xbox Series X & S, (Black)
3. `B08NWG321Q`: Devinal Headset Splitter, Headphone mic Splitter, 3.5mm trrs Audio Splitter, 1/8" inch 4 Pole Male to 2 Dual Female Adapter Cord Converter Connector 2 Pack

#### Query `q_010`: *"webcam with ring light and microphone for streaming"* (`attribute_use_case`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.12`)
- **Target Relevant**:
  -   - `B075X8471B`: Fire TV Stick with Alexa Voice Remote, streaming media player - Previous Generation
  - `B08F9ZCTCL`: Fire TV Stick Lite, free and live TV, Alexa Voice Remote Lite, smart home controls, HD streaming
- **Dense Top-3**:
  1. `B0841RV64Q`: Angetube Streaming 1080P HD Webcam Built in Adjustable Ring Light and Mic. Advanced autofocus AF Web Camera for Xbox Gamer Facebook YouTube Streamer (Renewed)
2. `B09GLHW26D`: Aluratek 2K HD Ring Light Webcam with Auto Focus w/Tripod
3. `B08RDQ5YG9`: Tenveo 3 in 1 Webcam with Ring Light and Microphone, T1 1080P USB HD PC Webcam for Streaming Gaming Conferencing Studying Tripod Included T1(Black)
- **BM25 Top-3**:
  1. `B08RDQ5YG9`: Tenveo 3 in 1 Webcam with Ring Light and Microphone, T1 1080P USB HD PC Webcam for Streaming Gaming Conferencing Studying Tripod Included T1(Black)
2. `B0B8D6L6D3`: MEE audio CL8A 1080p HD Webcam with Ring Light, Microphone, Autofocus, Low Light Correction, 360° Rotation; USB Streaming Web Camera for Video Calling via Zoom/Skype on Computer PC Mac Laptop Desktop
3. `B08R1BJCGC`: DANGAN 2021 Webcam with Ring Light,1080P HD Web Camera with Light and Microphone, Optional Privacy Cover, Plug and Play Stream Camera for Online Class, Zoom Meeting, Facetime Gathering, Skype Teams

#### Query `q_012`: *"ultra high speed micro SD card for 4K action camera"* (`attribute_spec`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.25`) vs Dense R@10 = `0.00` (R@100 = `0.50`)
- **Target Relevant**:
  -   - `B0BRZPCQNJ`: SanDisk 64GB 2-Pack Ultra microSDXC UHS-I Memory Card (2x64GB) with Adapter - SDSQUAB-064G-GN6MT
  - `B07WMB4XS4`: SanDisk Ultra 16GB Class 10 SDHC UHS-I Memory Card up to 80MB/s (SDSDUNC-016G-GN6IN)
- **Dense Top-3**:
  1. `B00MBFPT1W`: SanDisk Extreme 128GB U3/UHS-I SDXC with 4K Ultra HD, up to 80MB/s Read; 60MB/s Write- SDSDXN-128G-G46 [Older Version]
2. `B0153ARJ2S`: SanDisk 128GB Extreme Plus microSDXC UHS-I Memory Card with Adapter - 95MB/s, U3, V30, 4K UHD, Micro SD Card - SDSQXWG-128G-GN6MA
3. `B00PAXEGKC`: Zectron 32GB Micro SDHC-UHS-1 Memory Card for LG G Pad 7.0 LTE
- **BM25 Top-3**:
  1. `B09LM915XQ`: KEXIN Micro SD Card 128GB -Memory -Card + Adapter, 128GB microSDXC Full HD & 4K UHD, UHS-I, U3, 3Pack Mini SD Card Expanded Storage for Android Smartphones, Tablets
2. `B08KSDYG1B`: SanDisk 64GB Micro Extreme Memory Card for Samsung Phone Works with Galaxy S20, S20+, S20 Ultra, S20 Fan Edition (SDSQXA2-064G-GN6MN) Bundle with (1) Everything But Stromboli SD & MicroSD Card Reader
3. `B0153ARJ2S`: SanDisk 128GB Extreme Plus microSDXC UHS-I Memory Card with Adapter - 95MB/s, U3, V30, 4K UHD, Micro SD Card - SDSQXWG-128G-GN6MA

#### Query `q_013`: *"surge protector power strip with USB ports"* (`attribute_spec`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.25`) vs Dense R@10 = `0.00` (R@100 = `0.50`)
- **Target Relevant**:
  -   - `B0BXZ4D1C2`: GE Pro 3-Outlet Power Strip with Surge Protection, 8 Ft Designer Braided Extension Cord, Grounded, Flat Plug, 250 Joules, Warranty, UL Listed, Gray/White, 38433
  - `B09XYMSZLL`: GE 6-Outlet Surge Protector, 15 Ft Extension Cord, Power Strip, 800 Joules, Flat Plug, Twist-to-Close Safety Covers, Protected Indicator Light, UL Listed, White, 50768
- **Dense Top-3**:
  1. `B07Z5VFMR2`: Power Strip with USB, NTONPOWER Surge Protector with 8 Outlets(3 Extra Widely Spaced) and 2 USB Charging Ports, Flat Plug Extension Cord 5 Ft, 1875W/15A Overload Protection, UL Listed for Home Office
2. `B08JKV56LY`: Surge Protector Power Strip with 3 Outlets and 4 USB Ports, 6 Ft Extension Cord Overload Protection Power Strip for Home Office White
3. `B074M9FDKT`: Power Strip Tower ONEreach Surge Protector Electric Charging Station 2500W 10A of 2 USB 11 AC Outlets with 6.5ft Long Cord Wire Extension Universal Socket for Laptops Mobile Office Home Use
- **BM25 Top-3**:
  1. `B0B588ZVRK`: Surge Protector Power Strip with 18W Fast Charging Port, 10FT Extension Cord with 12 AC Outlets and 5 USB Ports, 1800J Protection, Overload Protection for Home Office Dorm Room
2. `B0854G9S2D`: Aduro Surge Protector 6 Outlets Power Strip Station with USB (4 Ports 4.8A) Wall Mount Multiple Outlet Splitter Extender Adapter with Phone Shelf Stand ETL Listed, Pink
3. `B0928LWD89`: Power Strip Surge Protector,6 AC Outlets Power Strip with 3 USB Charging Ports(3 x 2.4A ),Slide-to-Close Outlet Covers and 5 ft Extension Cord, 1250W/10A, 1200 Joules,for Home, Office and Hotel,White

#### Query `q_014`: *"smart home indoor wifi security camera night vision"* (`semantic_attribute`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B08F6GPRH6`: Blink Whole Home Bundle | Video Doorbell System, Outdoor camera, and Mini camera | HD video, motion detection, Works with Alexa
  - `B08QSB7SR3`: Ring Stick Up Cam Battery HD security camera with custom privacy controls, Simple setup, Works with Alexa - White
- **Dense Top-3**:
  1. `B09YNCMX7X`: Light Bulb Camera, 1080P Wireless Home Security Camera, 360 Degree 2.4GHz WiFi Smart Surveillance Cam with Motion Detection Alarm Night Vision
2. `B0C5WZXK25`: Security Cameras Wireless Outdoor - 1080p HD Night Vision WiFi Wireless Cameras for Home Security, Waterproof Surveillance Camera with Motion Detection, 2-Way Audio, Rechargeable Battery, SD Storage
3. `B07MR3YLML`: ESCAM Outdoor WiFi Security Light Camera, 1080P HD Night Vision Two-Way Audio IP66 Weatherproof Camera with IR LED Motion Detection
- **BM25 Top-3**:
  1. `B08PN3PZWM`: Lorex 1080p HD Smart Indoor/Outdoor Wi-Fi Security Camera with 32GB, Smart Deterrence, and Color Night Vision
2. `B0BTGQHYZB`: Geeni Vivid Indoor Smart WiFi Security Camera - 1080p HD Surveillance with 2-Way Speaker, Motion Sensor, and Night Vision - Compatible with Alexa and Google Assistant
3. `B0BSLH5GLJ`: CAMPPARKT 4PSC Light Bulb Security Cameras Wireless, 1080P WiFi Light Socket Camera, 360° Smart Porch Lightbulb Camera for Home, Auto Tracking, Color Night Vision, Two Way Audio, Work with Alexa

#### Query `q_015`: *"clip-on lapel lavalier microphone for smartphone video recording"* (`long_tail_intent`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B07S764D9V`: Panasonic ErgoFit Wired Earbuds, In-Ear Headphones with Microphone and Call Controller, Ergonomic Custom-Fit Earpieces (S/M/L), 3.5mm Jack for Phones and Laptops - RP-TCM125-A (Blue)
  - `B0BRZPCQNJ`: SanDisk 64GB 2-Pack Ultra microSDXC UHS-I Memory Card (2x64GB) with Adapter - SDSQUAB-064G-GN6MT
- **Dense Top-3**:
  1. `B08MVTG95S`: KINGONE Microphone Professional for Phone Lavalier Lapel Omnidirectional Condenser Mic Compatible with Phone 7/7 plus/8/8 plus/11/ Pro Max, Phone X/XS for Interview, Studio, Video, Vlogging,YouTube
2. `B078RFQ5P9`: Lavalier Microphone, Mouriv CM201 Hands Free Clip-on Lapel Mic with Omnidirectional Condenser for Podcast, Recording, DSLR,Camera, Smartphone, Sony,PC,Laptop (236 in)
3. `B075ZDKDY5`: Nicama LVM3 Lavalier Lapel Microphone for DSLR Camera Canon Nikon Sony Camcorder Smartphone PC Laptop (20FT)
- **BM25 Top-3**:
  1. `B08MVTG95S`: KINGONE Microphone Professional for Phone Lavalier Lapel Omnidirectional Condenser Mic Compatible with Phone 7/7 plus/8/8 plus/11/ Pro Max, Phone X/XS for Interview, Studio, Video, Vlogging,YouTube
2. `B089VPJ4Z2`: Saramonic LavMicro U3-OP Plug and Play Lavalier Microphone Digital Omnidirectional Clip-on Lapel Mic USB Type-C Plug Compatible with DJI OSMO Pocket Camera for Vlog Film Video Recording
3. `B01IQWRTSY`: BOYA by-M1 3.5mm Lavalier Condenser Microphone with Windscreen Windshield for Smartphones, DSLR, Recorder,Camcorders

#### Query `q_020`: *"laptop stand aluminum foldable portable for desk"* (`category_use_case`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B07XY2ZKYB`: ProCase iPad 10.2 Case iPad 9th Generation 2021/ iPad 8th Generation 2020/ iPad 7th Generation 2019 Case, Slim Stand Hard Back Shell Protective Smart Cover Case for iPad 10.2 Inch -Purple
  - `B07XCGYYVF`: MoKo iPad 10.2 Case for iPad 9th Generation 2021/ iPad 8th Generation 2020/ iPad 7th Generation 2019, Slim Stand Hard Back Shell Smart Cover Case for iPad 10.2 inch, Auto Wake/Sleep, Navy Blue
- **Dense Top-3**:
  1. `B0B4JYB4HG`: Rmour Portable Laptop Stand for Desk Aluminum Desk Accessories for Women | Unlimited Adjustable Angle & Height MacBook Laptop for Home Office Accessories (Black)
2. `B08S2XCPMV`: Laptop Stand, Foldable Laptop Riser Holder Computer Stand,laptop cooling Stand,Adjustable Aluminum Portable Notebooks Stand, Compatible with MacBook Air Pro, HP, Dell, All 6-18” Laptops and Tablets
3. `B08QS2JFK5`: Foldable Laptop Stand,Adjustable Aluminum Computer Stand Ergonomic Laptop Riser Portable Laptop Stand for Desk Cooling Holder MacBook pro Stand,Laptop Stands Compatible with All 6-18” Laptop Tablet
- **BM25 Top-3**:
  1. `B097MS1YHG`: Laptop Stand, Laptop Stand for Desk, Adjustable Stands - Portable Notebook Computer Holder Compatible with MacBook Pro,Air and More, Ergonomic Foldable Height Adjustable Aluminum Stand
2. `B08KKV3DXT`: Aloptis Laptop Stand, Laptop Riser, Computer Stand for Laptop, Aluminum Foldable Laptop Stand for Desk, Compatible with Dell, HP, MacBook, Lenovo and More Upto 10-15.6" Laptops Notebooks
3. `B08JCV9TCH`: Laptop Stand for Desk, Cooling Fan and Mouse Pad Adjustable Laptop Bed Table Laptop Desk Stand Portable Multifunctional Laptop Bed Desk with Mouse Pad Lap Desk for Home Office Aluminum Black

#### Query `q_021`: *"electronic digital luggage scale for travel suitcase"* (`long_tail_intent`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B07PMGBVV8`: LiNKFOR 1080P HDMI to Component Converter Scaler, HDMI Input to YPbPr Convert HDMI to Component, Only HDMI to Component Converter for HDTV Box PC PS3 Roku Blu-Ray DVD (NOT Component to HDMI)
  - `B08CTRW5GX`: OREI 4K@60Hz 1 in 2 Out HDMI Duplicator Splitter - with Scaler 1x2 2 Ports with Full Ultra HD, HDCP 2.2, 4K at 60Hz 4: 4: 4 1080p & 3D Supports EDID Control - UHD-PRO102
- **Dense Top-3**:
  1. `B00006IC30`: Pelouze ADPT2 AC Adapter for Digital Postal Scales
2. `B09CCTM3LR`: BAGSMART Electronics Organizer Travel Case, Small Travel Cable Organizer Bag for Travel Essentials, Travel Tech Organizer as Travel Accessories for Women, Cord Organizer for Phone, SD Card, Soft Pink
3. `B0C5J8Q9LY`: BAGSMART Electronics Organizer Travel Case, Small Travel Cord Organizer Bag for Travel Essentials, Travel Tech Organizer as Travel Accessories for Men Women, Cable Organizer for Phone, SD Card, Blue
- **BM25 Top-3**:
  1. `B0B2BYB14N`: Carry On Backpack, Large Travel Backpack, 50L Backpack, 18 Inch Convertible Lightweight Expandable Travel Back Pack for Men Carry on, TSA Black Luggage Suitcase Backpack Mens Travel Gifts 17.3
2. `B09WZ4B8P2`: MAQTOIZ Carry On Backpack, Large Luggage Backpack for Men/Women, 40L Weekender Overnight Trip Travel Backpack, 2023 Gift for Him
3. `B09NR2PJLL`: Large Travel Backpack, Carry on Backpack, 45L Expandable Flight Approved Big Extra Large Travel Backpack for Men Women, Waterproof Hiking Daypack Luggage Weekender Bag Fits 17 Inch Laptops, Black

#### Query `q_022`: *"wireless barcode scanner handheld for inventory"* (`professional_use_case`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.25`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B0BPMLRQHQ`: KODAK Step Printer Wireless Mobile Photo Printer with Zink Zero Ink Technology & Kodak App for iOS & Android (White) Gift Bundle
  - `B07B9XJ3CF`: CableCreation USB C Printer Cable 3.3FT USB C to Printer Cable USB C to B, Scanner Cable Printer Cable to USB C MIDI Cable for Yamaha Casio Digital Piano MIDI Controller DJ Controller, 1M Black
- **Dense Top-3**:
  1. `B086DGZZXR`: SOCKET DuraScan D700, 1D Barcode Scanner, Gray & Charging Dock
2. `B07T7WRRD9`: Uniden BCD536HP HomePatrol Series Digital Phase 2 Base/Mobile Scanner with HPDB and Wi-Fi & (BC23A) Bearcat 15-Watt Amplified External Communications Speaker
3. `B001NPISZA`: GRE PSR500 Digital APCO-25 Triple-Trunking Handheld Scanner
- **BM25 Top-3**:
  1. `B01KX1MERM`: BATIGE Barcode Scanner Cable Wire Replacement for Motorola Symbol RS409 RS419 Cable
2. `B086DGZZXR`: SOCKET DuraScan D700, 1D Barcode Scanner, Gray & Charging Dock
3. `B09RPGVJ22`: USB Cable for Motorola Symbol Scanner LS2208 DS6878 LI4278 LS3578 7FT 2M USB Type A barcode scanner Cable CBA-U01-S07ZAR, Gray

#### Query `q_024`: *"headphone stand with USB charger ports desktop"* (`attribute_use_case`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B07XY2ZKYB`: ProCase iPad 10.2 Case iPad 9th Generation 2021/ iPad 8th Generation 2020/ iPad 7th Generation 2019 Case, Slim Stand Hard Back Shell Protective Smart Cover Case for iPad 10.2 Inch -Purple
  - `B07XCGYYVF`: MoKo iPad 10.2 Case for iPad 9th Generation 2021/ iPad 8th Generation 2020/ iPad 7th Generation 2019, Slim Stand Hard Back Shell Smart Cover Case for iPad 10.2 inch, Auto Wake/Sleep, Navy Blue
- **Dense Top-3**:
  1. `B09BFNB5C3`: Rhinenet RGB Headphone Stand Dual Headset Holder Hanger with PD Fast Charging + 3 USB Charger Hub Port for Desktop Desk Gamers Accessories Boyfriend Gifts Anti-Slip Base Earphone Rack Station Dock
2. `B08FQPT29Q`: NEETTO HS906 Headphone Stand & Hanger 2 in 1, Above & Under Desk Gaming Headset Holder Mount Hook with Height Adjustable & Rotating Clamp, Earphone Rack with Cable Clip
3. `B08JYHPKRB`: Gaming Headphone Stand Holder Hook Mount,Foldable Headset Hanger Earphone Rack with Adjustable & Rotating Arm Clamp,Under Desk Design for All Headphone (Black)
- **BM25 Top-3**:
  1. `B00NR83V0I`: Samsung Universal Cradles Dock with Samsung Galaxy Note 3 & Galaxy S5 Note Pro Charger with 5FT 21-Pin Data Cable 3.0 for High Speed USB Data Transfer
2. `B09BFNB5C3`: Rhinenet RGB Headphone Stand Dual Headset Holder Hanger with PD Fast Charging + 3 USB Charger Hub Port for Desktop Desk Gamers Accessories Boyfriend Gifts Anti-Slip Base Earphone Rack Station Dock
3. `B093BH1N32`: Rolanstar Monitor Stand Riser with 4-Port USB Hub, 3 Shelf Desktop Stand with Phone Holder, Adjustable Desk Organizer Stand for Laptop/Computer/Tablet(Bamboo)

#### Query `q_026`: *"cable management box organizer for desk cords"* (`use_case`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.12`)
- **Target Relevant**:
  -   - `B07S8F43Y6`: D-Line Cord Cover 2-Pack, Cord Hiders, Cable Concealer, Cable Raceway, Wire Covers, Wall Mounted TV Wire Hider, Cable Management - 2X 1.18in W x 0.59in H x 39in Lengths (78in Total) - Wood-Effect
  - `B0B8CPMS4T`: SOULWIT Cable Holder Clips, 3-Pack Cable Management Cord Organizer Clips Silicone Self Adhesive for Desktop USB Charging Cable Nightstand Power Cord Mouse Cable Wire PC Office Home
- **Dense Top-3**:
  1. `B09TVSC2HM`: Cable Management Kit, PC Cord Management Organizer for Desk, 4 Cable Sleeve, 10 Self Adhesive Cable Clips, 10pcs and 2 Rolls Self Adhesive Tie and 100 Zip Ties for TV Computer Office Home
2. `B0BR9Y79ST`: Cable Management Box - Wooden Style Cord Organizer Box to Hide Wires & Power Strips | Desk Computer Cable Organizer Box | Safe ABS Material | 12.6" (L) x 5.3" (W) 4.9" (H) | for Home & Office - Black
3. `B0BPLR8HTS`: Under Desk Cable Management Tray Black, Cinati Cable Management Under Desk No Drill, Cable Tray with Clamp for Desk Wire Management,Desk Cable Management Box for Office, Home - No Damage to Desk
- **BM25 Top-3**:
  1. `B08N6VY2ZH`: Large Cable Management Box - Cable Organizer Box and Power Strip Box for Electrical Cord Management - Desk Cord Hider and Floor Cable Management - Wire Storage and Organization for Cords
2. `B0BPLR8HTS`: Under Desk Cable Management Tray Black, Cinati Cable Management Under Desk No Drill, Cable Tray with Clamp for Desk Wire Management,Desk Cable Management Box for Office, Home - No Damage to Desk
3. `B0BDZ5LX24`: HAOBAOBEI Cable Ties, 4 Inch Reusable Fastening Cord Ties Keeper, Hook and Loop Cable Wrappers Cord Organizer, Adjustable Wire Management Straps for Home Office Data Centers PC TV (50PCS - Black)

#### Query `q_027`: *"anti blue light screen protector for computer monitor"* (`attribute_use_case`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B07KG7BLFD`: MOSISO Compatible with MacBook Air 13 inch Case 2022, 2021-2018 Release A2337 M1 A2179 A1932 Retina Display Touch ID, Plastic Hard Shell&Keyboard Cover&Screen Protector&Storage Bag, Lavender Gray
  - `B077TPFXZV`: MOSISO Compatible with MacBook Air 13 inch Case (Models: A1369 & A1466, Older Version 2010-2017 Release), Protective Plastic Hard Shell Case & Keyboard Cover & Screen Protector, Deep Teal
- **Dense Top-3**:
  1. `B0C611NNJX`: Blue Light Screen Protector for Computer Screen Blue Light Blocker, Anti Glare Computer Screen Cover, Blue Light Filter for 23, 23.6, 23.8, 24 inch Diagonal Widescreen Monitor Frame Hanging Type
2. `B0BRT84GJF`: 22 inch Anti-Glare Blue Light Blocking Screen Protector Panel for 16:10 Widescreen Computer Monitor - LED PC Anti-UV Eye Protection Filter Film - Anti-Scratch Diagonal Frame Shield
3. `B09ZXL1KB1`: Blue Light Blocking Screen Protector Panel for 14 inch Diagonal LED PC Laptop Anti-UV Eye Protection Filter Film - Widescreen Laptop Frame Hanging Type (W 12.6" X H 8.1")
- **BM25 Top-3**:
  1. `B0C611NNJX`: Blue Light Screen Protector for Computer Screen Blue Light Blocker, Anti Glare Computer Screen Cover, Blue Light Filter for 23, 23.6, 23.8, 24 inch Diagonal Widescreen Monitor Frame Hanging Type
2. `B089B6WPRP`: WS Screen Protector Blue Light Blocking Screen Protector Panel for 23 Inches (Diagonally Measured Screen) Desktop Monitor, Block Hazardous HEV Light, Reduce PC Eye Strain
3. `B073T5J216`: Anti Blue Light Screen Filter for 19 Inches Widescreen Desktop Monitor, Blocks Excessive Harmful Blue Light, Reduce Eye Fatigue and Eye Strain

#### Query `q_029`: *"headphones under 50"* (`budget_constraint`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B07S764D9V`: Panasonic ErgoFit Wired Earbuds, In-Ear Headphones with Microphone and Call Controller, Ergonomic Custom-Fit Earpieces (S/M/L), 3.5mm Jack for Phones and Laptops - RP-TCM125-A (Blue)
  - `B07V6CGGW2`: R-fun AirPods Case Cover, Soft Silicone Protective Cover with Keychain for Women Men Compatible with Apple AirPods 2nd 1st Generation Charging Case, Front LED Visible-Pink Sand
- **Dense Top-3**:
  1. `B0BCVRB1TY`: iClever Smiley Kids Headphones Wired with Microphone, 85/94dB Volume Limited, Over-Ear Headphones for Kids with Share Port, Stereo Sound, Foldable Kids Headphones for School/Travel/iPad/Fire Tablet
2. `B0BP6SWDD7`: Bulk Headphone Earphones 32 Pack for School Headphones with 3.5 mm Headphone Plug for School Classroom Library Students Kids Children Teen and Adults (Multicolor)
3. `B07XFFZT4M`: LowCostEarbuds Bulk Wholesale Lot of 25 White/Gray Earbuds Headphones
- **BM25 Top-3**:
  1. `B073QX9RMV`: Keewonda Wholesale Bulk Earbuds Headphones - 50 Pack Cute Earphone Colored Earbuds for Kids Teens Students (Black/White)
2. `B07GLTGP54`: Keewonda Bulk Earbuds Headphones Wholesale Earphones, 100 Pack Disposable Ear Buds Bulk Individually Wrapped Headphones for School Classroom Students
3. `B09MDT9K22`: JBL Under Armour Sport Train Wireless On-Ear Bluetooth Headphones, IPX4 Sweatproof for Sports with Under Armour Carrying Case (Black)

#### Query `q_030`: *"usb c hub under 30"* (`budget_constraint`)
- **Metrics**: BM25 R@10 = `0.00` (R@100 = `0.00`) vs Dense R@10 = `0.00` (R@100 = `0.00`)
- **Target Relevant**:
  -   - `B0BGS23YKX`: JSAUX USB-C to USB A Cable 3.1A Fast Charging [2-Pack 6.6ft], USB Type C Charger Cord Compatible with Samsung Galaxy S10 S9 S8 S20 Plus A51 A12 A11, Note 10 9 8, PS5 Controller USB C Charger-Green
  - `B07PWCN4LC`: Syntech USB C to USB Adapter Pack of 2 USB C Male to USB3 Female Adapter Compatible with MacBook Pro 2021 iMac iPad Mini 6/Pro MacBook Air 2022 and Other Type C or Thunderbolt 4/3 Devices Midnight
- **Dense Top-3**:
  1. `B09KX28C4K`: StarTech.com 4-Port USB-C Hub - 10Gbps - 3x USB-A & 1x USB-C - 9.8” Host Cable (HB31C3A1CB)
2. `B08VDD6TQY`: USB HUB 3.0 Type C, USB-C to USB-C, USB-C Hub 4 Ports, High-Speed Data Transfer, USB C HUB Compatible with Laptops, MacBook Pro Air and PS5
3. `B0BZH26KV4`: USB C HUB 4K 30Hz, VENTION USB C Multiport Adapter 5-in-1 with 4K HDMI, 100W Power Delivery, 3 USB 3.0 Data Port for MacBook Pro/Air M1 2020, iPad Pro 2021, iPad Mini 6, Surface Pro and More
- **BM25 Top-3**:
  1. `B08KGCY9X7`: USB C HUB, BENFEI 5-in-1 USB Type-C Hub with 4K HDMI VGA, USB 3.0, 3.5mm Audio and 60W PD, Compatible with MacBook Pro 2021/2020/2019, Surface Book 2, Dell XPS 13/15, Pixelbook and More
2. `B08CV6P5BD`: USB C Hub HDMI Adapter, 8-in-1 USB C Adapter with 4K HDMI, 3 USB 3.0, 100W PD, VGA and Audio, USB C Multiport Adapter Hub for MacBook Pro 2019/2018/2017, MacBook Air, Dell XPS More USB Type C Devices
3. `B07DT88488`: USB Hub Powered - Unibody Aluminum Multi-Port USB Hub with 10 USB 3.0 Ports, 3 Charging Ports, Cords C and A, Powered USB Splitter HUB - by Latorice


---

## 3. Systematic Dense Retrieval Failure Modes Identified

1. **Numeric & Budget Constraints (e.g. *"under $50"* / *"under $1000"* )**:
   - Bi-encoders map queries into continuous conceptual spaces where exact numeric bounds (e.g. price limits) are not mathematically enforced.
   - *Mitigation in later phases*: Stage 3 structured hybrid scoring and metadata range filtering.

2. **Fine-Grained Hardware Spec Specificity (e.g. *"RTX 4060"* vs *"RTX 3060"*, *"HDMI 2.1"* vs *"HDMI 2.0"*)**:
   - Bi-encoders cluster similar hardware entities closely together in latent space, causing occasional confusion between adjacent generations or model numbers.
   - *Mitigation in later phases*: Cross-encoder reranking (Stage 2) with token-level cross-attention.

3. **Complex Compatibility Reasoning (e.g. *"Adapter for MacBook Pro M2"* )**:
   - Requires relational understanding between host device port standards and accessory capabilities.
