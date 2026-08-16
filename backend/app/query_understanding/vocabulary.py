"""Catalog-derived vocabulary for entity, category, brand, and attribute extraction."""

from typing import Dict, List, Set

# Controlled Category Canonical Mapping with Synonyms
# Derived directly from the Amazon Electronics dataset
CATEGORY_SYNONYMS: Dict[str, str] = {
    # Laptops & Computers
    "laptop": "laptop",
    "laptops": "laptop",
    "notebook": "laptop",
    "notebooks": "laptop",
    "notebook computer": "laptop",
    "ultrabook": "laptop",
    "ultrabooks": "laptop",
    "macbook": "laptop",
    "chromebook": "laptop",
    "gaming laptop": "laptop",
    
    # Headphones & Audio
    "headphone": "headphones",
    "headphones": "headphones",
    "headset": "headphones",
    "headsets": "headphones",
    "earphone": "headphones",
    "earphones": "headphones",
    "over ear headphones": "headphones",
    "over-ear headphones": "headphones",
    "on ear headphones": "headphones",
    "on-ear headphones": "headphones",
    "studio monitor headphones": "headphones",
    "monitor headphones": "headphones",
    "studio headphones": "headphones",
    
    # Earbuds
    "earbud": "earbuds",
    "earbuds": "earbuds",
    "ear buds": "earbuds",
    "in ear headphones": "earbuds",
    "in-ear headphones": "earbuds",
    "in-ear earbuds": "earbuds",
    "tws": "earbuds",
    "true wireless earbuds": "earbuds",
    "airpods": "earbuds",
    
    # Monitors
    "monitor": "monitor",
    "monitors": "monitor",
    "display": "monitor",
    "displays": "monitor",
    "computer monitor": "monitor",
    "gaming monitor": "monitor",
    "screen": "monitor",
    
    # Keyboards
    "keyboard": "keyboard",
    "keyboards": "keyboard",
    "mechanical keyboard": "keyboard",
    "gaming keyboard": "keyboard",
    "keypad": "keyboard",
    
    # Mice
    "mouse": "mouse",
    "mice": "mouse",
    "gaming mouse": "mouse",
    "wireless mouse": "mouse",
    "trackball": "mouse",
    
    # Networking / Routers
    "router": "router",
    "routers": "router",
    "wifi router": "router",
    "wi-fi router": "router",
    "mesh wifi": "router",
    "mesh router": "router",
    "modem": "router",
    "access point": "router",
    "range extender": "router",
    "wifi extender": "router",
    
    # Speakers & Soundbars
    "speaker": "speaker",
    "speakers": "speaker",
    "bluetooth speaker": "speaker",
    "soundbar": "speaker",
    "sound bar": "speaker",
    "subwoofer": "speaker",
    "smart speaker": "speaker",
    
    # Power & Surge Protection
    "power strip": "power strip",
    "power strips": "power strip",
    "surge protector": "power strip",
    "surge protectors": "power strip",
    "extension cord": "power strip",
    
    # Power Banks & Chargers
    "power bank": "power bank",
    "power banks": "power bank",
    "portable charger": "power bank",
    "portable power bank": "power bank",
    "battery pack": "power bank",
    "external battery": "power bank",
    "charger": "charger",
    "chargers": "charger",
    "wall charger": "charger",
    "fast charger": "charger",
    "power adapter": "charger",
    
    # Cables
    "cable": "cable",
    "cables": "cable",
    "usb cable": "cable",
    "hdmi cable": "cable",
    "charging cable": "cable",
    "aux cable": "cable",
    "power cord": "cable",
    
    # Webcams & Cameras
    "webcam": "webcam",
    "webcams": "webcam",
    "web camera": "webcam",
    "streaming camera": "webcam",
    "camera": "camera",
    "cameras": "camera",
    "dslr": "camera",
    "action camera": "camera",
    "mirrorless camera": "camera",
    
    # Tablets
    "tablet": "tablet",
    "tablets": "tablet",
    "ipad": "tablet",
    "android tablet": "tablet",
    "drawing tablet": "tablet",
    "graphics tablet": "tablet",
    
    # Smartwatches & Wearables
    "smartwatch": "smartwatch",
    "smart watch": "smartwatch",
    "smartwatches": "smartwatch",
    "fitness tracker": "smartwatch",
    "fitness band": "smartwatch",
    
    # Television
    "tv": "tv",
    "tvs": "tv",
    "television": "tv",
    "smart tv": "tv",
    "oled tv": "tv",
    
    # Mounts & Stands
    "mount": "mount",
    "monitor arm": "mount",
    "desk mount": "mount",
    "wall mount": "mount",
    "tv mount": "mount",
    "laptop stand": "mount",
    
    # Cooling Pads
    "cooling pad": "cooling pad",
    "laptop cooling pad": "cooling pad",
    "laptop cooler": "cooling pad",
    
    # Storage & SSDs
    "ssd": "storage",
    "solid state drive": "storage",
    "hard drive": "storage",
    "external hard drive": "storage",
    "flash drive": "storage",
    "thumb drive": "storage",
    "sd card": "storage",
    "microsd": "storage",
    
    # Stylus
    "stylus": "stylus",
    "stylus pen": "stylus",
    "apple pencil": "stylus",
    "touch pen": "stylus",
    
    # Microphones
    "microphone": "microphone",
    "microphones": "microphone",
    "mic": "microphone",
    "usb microphone": "microphone",
    "condenser mic": "microphone",
}

# Catalog Brands Vocabulary
# Derived from actual brand distribution in products.parquet
CATALOG_BRANDS: Set[str] = {
    "sony", "hp", "samsung", "asus", "lenovo", "dell", "canon", "acer",
    "garmin", "panasonic", "philips", "belkin", "msi", "apple", "lg",
    "toshiba", "fujifilm", "intel", "logitech", "anker", "bose", "jbl",
    "sennheiser", "sandisk", "tp-link", "netgear", "ugreen", "baseus",
    "crucial", "western digital", "seagate", "kingston", "corsair", "razer",
    "steelseries", "audio-technica", "shure", "gopro", "fitbit", "roku",
    "amazon basics", "monoprice", "startech", "fintie", "moko", "pyle",
    "neewer", "kastar", "upbright", "uxcell", "hqrp",
}

# Multi-word brands sorted by descending length to prevent partial matching
MULTI_WORD_BRANDS: List[str] = sorted(
    [b for b in CATALOG_BRANDS if " " in b or "-" in b],
    key=len,
    reverse=True,
)

# Product Attributes Vocabulary
ATTRIBUTE_PATTERNS: Dict[str, Dict[str, str]] = {
    "gpu": {
        "rtx 4090": "RTX 4090",
        "rtx 4080": "RTX 4080",
        "rtx 4070": "RTX 4070",
        "rtx 4060": "RTX 4060",
        "rtx 3080": "RTX 3080",
        "rtx 3070": "RTX 3070",
        "rtx 3060": "RTX 3060",
        "rtx 3050": "RTX 3050",
        "gtx 1650": "GTX 1650",
        "rtx": "RTX",
        "gtx": "GTX",
        "radeon": "Radeon",
        "apple m1": "Apple M1",
        "apple m2": "Apple M2",
        "apple m3": "Apple M3",
    },
    "ram": {
        "64gb ram": "64GB",
        "32gb ram": "32GB",
        "16gb ram": "16GB",
        "8gb ram": "8GB",
        "4gb ram": "4GB",
        "64gb": "64GB",
        "32gb": "32GB",
        "16gb": "16GB",
        "8gb": "8GB",
    },
    "storage": {
        "4tb": "4TB",
        "2tb": "2TB",
        "1tb": "1TB",
        "512gb": "512GB",
        "256gb": "256GB",
        "128gb": "128GB",
        "ssd": "SSD",
        "nvme": "NVMe",
        "hdd": "HDD",
    },
    "connectivity": {
        "usb-c": "USB-C",
        "usbc": "USB-C",
        "usb-a": "USB-A",
        "usba": "USB-A",
        "usb 3.0": "USB 3.0",
        "usb 2.0": "USB 2.0",
        "thunderbolt": "Thunderbolt",
        "bluetooth": "Bluetooth",
        "wireless": "Wireless",
        "wired": "Wired",
        "wifi 7": "WiFi 7",
        "wifi 6e": "WiFi 6E",
        "wifi 6": "WiFi 6",
        "wifi": "WiFi",
        "hdmi": "HDMI",
        "displayport": "DisplayPort",
        "ethernet": "Ethernet",
        "usb": "USB",
        "aux": "Aux",
        "3.5mm": "3.5mm",
    },
    "use_case": {
        "gaming": "gaming",
        "gamer": "gaming",
        "office": "office",
        "business": "office",
        "work": "office",
        "travel": "travel",
        "airplane": "travel",
        "flight": "travel",
        "home": "home",
        "outdoor": "outdoor",
        "outdoors": "outdoor",
        "studio": "studio",
        "programming": "programming",
        "coding": "programming",
        "editing": "editing",
        "video editing": "editing",
        "streaming": "streaming",
        "fitness": "fitness",
        "gym": "fitness",
        "running": "fitness",
    },
    "features": {
        "noise cancelling": "noise_cancelling",
        "noise-cancelling": "noise_cancelling",
        "anc": "noise_cancelling",
        "active noise cancellation": "noise_cancelling",
        "mechanical": "mechanical",
        "ergonomic": "ergonomic",
        "rgb": "rgb",
        "waterproof": "waterproof",
        "fast charger": "fast_charging",
        "fast charging": "fast_charging",
        "quick charge": "fast_charging",
        "magsafe": "magsafe",
        "foldable": "foldable",
        "portable": "portable",
        "compact": "compact",
        "quiet fans": "quiet_fans",
        "adjustable": "adjustable",
        "dual": "dual",
    },
}
