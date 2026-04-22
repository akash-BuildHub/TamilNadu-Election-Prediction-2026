# Tamil Nadu Legislative Assembly Election 2026 - Shared Constants

import os
from typing import Dict, List

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_FILES_DIR = os.path.join(BASE_DIR, "data_files")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(DATA_FILES_DIR, exist_ok=True)

# Alliance classes used as model labels. See README of this module.
#   DMK_ALLIANCE: DMK + Congress + CPI + CPI(M) + VCK + DMDK + MDMK + IUML + allies
#   AIADMK_NDA:   AIADMK + BJP + PMK + AMMK + NDA allies
#   TVK:          Tamilaga Vettri Kazhagam (Vijay) - standalone
#   NTK:          Naam Tamilar Katchi (Seeman) - standalone
#   OTHERS:       independents, rebels, minor parties not in the above 4
PARTIES: List[str] = ["DMK_ALLIANCE", "AIADMK_NDA", "TVK", "NTK", "OTHERS"]
NUM_CLASSES = len(PARTIES)
NUM_CONSTITUENCIES = 234

# Canonical Tamil Nadu district names (38). These match the spellings used
# in backend/dataset/tamilnadu_assembly_constituency_2026.csv -- the state
# voters list uses variant spellings that are normalised in build_data_files.py.
DISTRICTS: List[str] = [
    "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore",
    "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kancheepuram",
    "Kanniyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
    "Nagapattinam", "Namakkal", "Perambalur", "Pudukkottai", "Ramanathapuram",
    "Ranipet", "Salem", "Sivaganga", "Tenkasi", "Thanjavur",
    "The Nilgiris", "Theni", "Thiruvarur", "Thoothukudi", "Tiruchirappalli",
    "Tirunelveli", "Tirupattur", "Tiruppur", "Tiruvallur", "Tiruvannamalai",
    "Vellore", "Viluppuram", "Virudhunagar",
]

# 5-way region coding used in constituency master
REGIONS: List[str] = ["North", "Central", "South", "West", "Chennai_Metro"]

# Party -> alliance mapping as of the 2026 cycle. When a party has moved
# alliance across years, per-year mapping happens inside create_dataset.py
# (see PARTY_TO_ALLIANCE_2021 etc.); this map is the 2026-cycle assignment.
PARTY_TO_ALLIANCE_2026: Dict[str, str] = {
    # DMK-led
    "DMK": "DMK_ALLIANCE", "INC": "DMK_ALLIANCE", "Congress": "DMK_ALLIANCE",
    "CPI": "DMK_ALLIANCE", "CPI(M)": "DMK_ALLIANCE",
    "VCK": "DMK_ALLIANCE", "DMDK": "DMK_ALLIANCE",
    "MDMK": "DMK_ALLIANCE", "IUML": "DMK_ALLIANCE",
    # AIADMK-led NDA
    "AIADMK": "AIADMK_NDA", "BJP": "AIADMK_NDA",
    "PMK": "AIADMK_NDA", "AMMK": "AIADMK_NDA",
    # Standalone
    "TVK": "TVK",
    "NTK": "NTK",
    # Everything else
    "MNM": "OTHERS", "IND": "OTHERS", "Other": "OTHERS",
}

# Sentiment/keyword set for live_collectors.py (X, YouTube, NewsAPI queries).
# Keys are the 4 alliance classes plus a "general" bucket for non-partisan
# election-related keywords. Includes Tamil-script entries.
SENTIMENT_KEYWORDS: Dict[str, List[str]] = {
    "DMK_ALLIANCE": [
        "DMK", "Stalin", "M.K. Stalin", "Congress Tamil Nadu", "VCK",
        "Thirumavalavan", "CPI Tamil Nadu", "CPI(M) Tamil Nadu", "MDMK", "DMDK",
        "DMK alliance", "INDIA bloc Tamil Nadu",
        "திமுக", "ஸ்டாலின்", "திருமாவளவன்", "விசிக",
    ],
    "AIADMK_NDA": [
        "AIADMK", "EPS", "Edappadi Palaniswami", "BJP Tamil Nadu",
        "Annamalai", "PMK", "Anbumani Ramadoss", "AMMK", "TTV Dhinakaran",
        "AIADMK alliance", "NDA Tamil Nadu",
        "அதிமுக", "எடப்பாடி பழனிசாமி", "பாஜக", "பமக",
    ],
    "TVK": [
        "TVK", "Tamilaga Vettri Kazhagam", "Vijay", "Thalapathy Vijay",
        "TVK Vijay", "Vijay politics", "TVK rally",
        "தவெக", "விஜய்", "தமிழக வெற்றிக் கழகம்",
    ],
    "NTK": [
        "NTK", "Seeman", "Naam Tamilar Katchi", "Naam Tamilar",
        "Seeman NTK", "NTK rally",
        "நாம் தமிழர்", "நாம் தமிழர் கட்சி", "சீமான்",
    ],
    "OTHERS": [
        "MNM", "Kamal Haasan",
        "independent candidate Tamil Nadu", "Tamil Nadu rebel candidate",
        "மநீம", "கமல் ஹாசன்",
    ],
    "general": [
        "Tamil Nadu election", "Tamil Nadu assembly election 2026",
        "TN polls 2026", "Tamil Nadu niyamasabha", "anti-incumbency Tamil Nadu",
        "Tamil Nadu voter turnout", "Tamil Nadu seat sharing",
        "தமிழ்நாடு தேர்தல்", "சட்டமன்றத் தேர்தல்",
    ],
}
