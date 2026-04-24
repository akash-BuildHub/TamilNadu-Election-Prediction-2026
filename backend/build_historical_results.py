"""
Build per-constituency Tamil Nadu Assembly election result CSVs for 2016 and 2021.

Data sources:
- Master spine: backend/dataset/tamilnadu_assembly_constituency_2026.csv (ac_no, ac_name for 234 ACs)
- 2021 winner+runner-up: statisticstimes.com (constituency, winner_party, runner_up_party, margin)
- 2021 winner cross-check: myneta.info/TamilNadu2021 (winners list)
- 2016 winner: myneta.info/tamilnadu2016 (winners list; no runner-up on page)

Since statisticstimes only gives margin (not vote counts), vote columns for 2021 are set to 0.
For 2016 we only have winner party from myneta, so runner_up_* and vote columns are 0.
All rows where we have a real winner_party from the web are flagged is_real_result=1;
rows where we could not match remain 0.
"""

import os
import re
import pandas as pd

from config import DATA_FILES_DIR, DATASET_DIR

OUT_DIR = DATA_FILES_DIR
MASTER = os.path.join(DATASET_DIR, "tamilnadu_assembly_constituency_2026.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# --- Party / alliance helpers -----------------------------------------------

VALID_PARTIES = {
    "DMK", "AIADMK", "INC", "BJP", "PMK", "CPI", "CPI(M)", "VCK",
    "DMDK", "MDMK", "IUML", "AMMK", "MNM", "NTK", "TVK", "IND", "Other",
}

_PARTY_NORMALIZE = {
    "CPIM": "CPI(M)",
    "CPI(M)": "CPI(M)",
    "CPM": "CPI(M)",
    "CPI-M": "CPI(M)",
    "INDEPENDENT": "IND",
    "IND.": "IND",
}


def norm_party(p):
    if p is None:
        return ""
    p = str(p).strip().upper()
    if not p:
        return ""
    p = _PARTY_NORMALIZE.get(p, p)
    if p in {"DMK", "AIADMK", "INC", "BJP", "PMK", "CPI", "CPI(M)", "VCK",
             "DMDK", "MDMK", "IUML", "AMMK", "MNM", "NTK", "TVK", "IND"}:
        return p
    return "Other"


ALLIANCE_2016 = {
    "AIADMK": "AIADMK_NDA",
    "PMK": "AIADMK_NDA",  # PMK contested with AIADMK front (no, actually PMK was solo in 2016)
    "DMK": "DMK_ALLIANCE",
    "INC": "DMK_ALLIANCE",
    "CPI": "DMK_ALLIANCE",
    "CPI(M)": "DMK_ALLIANCE",
    "VCK": "DMK_ALLIANCE",
    "IUML": "DMK_ALLIANCE",
    "MDMK": "OTHERS",  # MDMK was not with DMK in 2016
    "BJP": "OTHERS",
    "DMDK": "OTHERS",
    "NTK": "NTK",
    "IND": "OTHERS",
    "Other": "OTHERS",
}
# Per the task spec for 2016: AIADMK front = AIADMK/PMK; DMK front = DMK/INC/CPI/CPIM/VCK.
# Overriding per spec:
ALLIANCE_2016["PMK"] = "AIADMK_NDA"
ALLIANCE_2016["IUML"] = "OTHERS"  # spec 2016 DMK front doesn't include IUML
# (IUML won 1 seat in 2016 but was not in DMK front; task says "OTHERS" for IUML 2016)

ALLIANCE_2021 = {
    "AIADMK": "AIADMK_NDA",
    "PMK": "AIADMK_NDA",
    "BJP": "AIADMK_NDA",
    "DMK": "DMK_ALLIANCE",
    "INC": "DMK_ALLIANCE",
    "CPI": "DMK_ALLIANCE",
    "CPI(M)": "DMK_ALLIANCE",
    "VCK": "DMK_ALLIANCE",
    "MDMK": "DMK_ALLIANCE",
    "IUML": "DMK_ALLIANCE",
    "AMMK": "OTHERS",
    "MNM": "OTHERS",
    "NTK": "NTK",
    "DMDK": "OTHERS",
    "IND": "OTHERS",
    "Other": "OTHERS",
}


def alliance_for(party, year):
    tbl = ALLIANCE_2016 if year == 2016 else ALLIANCE_2021
    return tbl.get(party, "OTHERS")


# --- AC-name normalization for fuzzy matching -------------------------------

def canon(name):
    if name is None:
        return ""
    s = str(name).upper()
    # Remove reservation markers / byelection suffixes.
    # Guard against over-stripping "East"/"West" by requiring a word boundary
    # (i.e. the SC/ST token is a standalone word).
    s = re.sub(r"\s*\(\s*(SC|ST)\s*\)\s*$", "", s)
    s = re.sub(r"\s+(SC|ST)\s*$", "", s)
    s = re.sub(r":\s*BYE.*$", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    # common spelling variations
    repl = {
        "THIRU": "TIRU",
        "DHARAPURAM": "DHARAPURAM",
        "KANNIYAKUMARI": "KANYAKUMARI",
        "KANYAKUMARI": "KANYAKUMARI",
        "SHOZHINGANALLUR": "SHOLINGANALLUR",
        "SHOZHINGHANALLUR": "SHOLINGANALLUR",
        "POONAMALLEE": "POONAMALLEE",
        "POONMALLAE": "POONAMALLEE",
        "MADHAVARAM": "MADHAVARAM",
        "MADAVARAM": "MADHAVARAM",
        "GANDARVAKKOTTAI": "GANDHARVAKOTTAI",
        "GANDARVAKOTTAI": "GANDHARVAKOTTAI",
        "GANDHARVAKOTTAI": "GANDHARVAKOTTAI",
        "MUDUKULATHUR": "MUDHUKULATHUR",
        "MUDHUKULATHUR": "MUDHUKULATHUR",
        "KATTUMANARKOIL": "KATTUMANNARKOIL",
        "CHEPAUK-THIRUVALLIKENI": "CHEPAUKTHIRUVALLIKENI",
        "DR.RADHAKRISHNANNAGAR": "DRRADHAKRISHNANNAGAR",
        "DR.RADHAKRISHNAN NAGAR": "DRRADHAKRISHNANNAGAR",
        "PERIYAKULAM(SC)": "PERIYAKULAM",
        "ARUPPUKKOTTAI": "ARUPPUKOTTAI",
        "ARUPPUKOTTAI": "ARUPPUKOTTAI",
        "THOOTHUKKUDI": "THOOTHUKUDI",
        "THOOTHUKUDI": "THOOTHUKUDI",
        "NEYVELI": "NEYVELI",
        "VRIDDHACHALAM": "VRIDHACHALAM",
        "VRIDHACHALAM": "VRIDHACHALAM",
        "VEDHARANYAM": "VEDARANYAM",
        "VEDARANYAM": "VEDARANYAM",
        "METTUPPALAYAM": "METTUPALAYAM",
        "METTUPALAYAM": "METTUPALAYAM",
        "PAPPIREDDIPPATTI": "PAPPIREDDIPATTI",
        "PAPPIREDDIPATTI": "PAPPIREDDIPATTI",
        "KAVUNDAMPALAYAM": "KAVUNDAMPALAYAM",
        "MADURAINORTH": "MADURAINORTH",
        "MADURAISOUTH": "MADURAISOUTH",
        "MADURAIEAST": "MADURAIEAST",
        "MADURAIWEST": "MADURAIWEST",
        "BODINAYAKANUR": "BODINAYAKKANUR",
        "BODINAYAKKANUR": "BODINAYAKKANUR",
        "SHOLINGHUR": "SHOLINGUR",
        "SHOLINGUR": "SHOLINGUR",
        "VILLUPURAM": "VILUPPURAM",
        "VILUPPURAM": "VILUPPURAM",
        "COLACHEL": "COLACHAL",
        "COLACHAL": "COLACHAL",
        "TITTAKUDI": "TITTAKUDI",
        "TITTAGUDI": "TITTAKUDI",
        # NOTE: "Tirupattur" (AC 50, Vellore district) and "Tiruppattur" (AC 185,
        # Sivaganga) are DIFFERENT constituencies. We must NOT collapse them.
        # canon() preserves the difference via the letter count.
    }
    return repl.get(s, s)


# --- Read master spine -------------------------------------------------------

master = pd.read_csv(MASTER)
master = master[["ac_no", "ac_name"]].copy()
master["ac_no"] = master["ac_no"].astype(int)
master["_key"] = master["ac_name"].map(canon)
print(f"Master: {len(master)} constituencies")

# --- 2021 data (from statisticstimes) ---------------------------------------

RAW_2021 = """Alandur,DMK,AIADMK,40571
Alangudi,DMK,AIADMK,25847
Alangulam,AIADMK,DMK,3539
Ambasamudram,AIADMK,DMK,16915
Ambattur,DMK,AIADMK,42146
Ambur,DMK,AIADMK,20232
Anaikattu,DMK,AIADMK,6360
Andipatti,DMK,AIADMK,8538
Anna Nagar,DMK,AIADMK,27445
Anthiyur,DMK,AIADMK,1275
Arakkonam,AIADMK,VCK,27169
Arani,AIADMK,DMK,3128
Aranthangi,INC,AIADMK,30893
Aravakurichi,DMK,BJP,24816
Arcot,DMK,PMK,19958
Ariyalur,DMK,AIADMK,3234
Aruppukkottai,DMK,AIADMK,39034
Athoor,DMK,PMK,135571
Attur,AIADMK,DMK,8257
Avadi,DMK,AIADMK,55275
Avanashi,AIADMK,DMK,50902
Bargur,DMK,AIADMK,12614
Bhavani,AIADMK,DMK,22523
Bhavanisagar,AIADMK,CPI,16008
Bhuvanagiri,AIADMK,DMK,8259
Bodinayakanur,AIADMK,DMK,11021
Chengalpattu,DMK,AIADMK,26665
Chengam,DMK,AIADMK,11570
Chepauk-thiruvallikeni,DMK,PMK,69355
Cheyyar,DMK,AIADMK,12271
Cheyyur,VCK,AIADMK,4042
Chidambaram,AIADMK,IUML,16937
Coimbatore (north),AIADMK,DMK,4001
Coimbatore (south),BJP,MNM,1728
Colachal,INC,BJP,24832
Coonoor,DMK,AIADMK,4105
Cuddalore,DMK,AIADMK,5151
Cumbum,DMK,AIADMK,42413
Dharapuram,DMK,BJP,1393
Dharmapuri,PMK,DMK,26860
Dindigul,AIADMK,CPIM,17747
Dr.radhakrishnan Nagar,DMK,AIADMK,42479
Edappadi,AIADMK,DMK,93802
Egmore,DMK,AIADMK,38768
Erode (east),INC,AIADMK,8904
Erode (west),DMK,AIADMK,22089
Gandarvakkottai,CPIM,AIADMK,12721
Gangavalli,AIADMK,DMK,7361
Gingee,DMK,PMK,35803
Gobichettipalayam,AIADMK,DMK,28563
Gudalur,AIADMK,DMK,1945
Gudiyattam,DMK,AIADMK,6901
Gummidipoondi,DMK,PMK,50938
Harbour,DMK,BJP,27274
Harur,AIADMK,CPIM,30362
Hosur,DMK,AIADMK,12367
Jayankondam,DMK,PMK,5452
Jolarpet,DMK,AIADMK,1091
Kadayanallur,AIADMK,IUML,24349
Kalasapakkam,DMK,AIADMK,9222
Kallakurichi,AIADMK,INC,25891
Kancheepuram,DMK,PMK,11595
Kangayam,DMK,AIADMK,7331
Kanniyakumari,AIADMK,DMK,16213
Karaikudi,INC,BJP,21589
Karur,DMK,AIADMK,12448
Katpadi,DMK,AIADMK,746
Kattumannarkoil,VCK,AIADMK,10565
Kavundampalayam,AIADMK,DMK,9776
Killiyoor,INC,AIADMK,55400
Kilpennathur,DMK,PMK,26787
Kilvaithinankuppam,AIADMK,DMK,10582
Kilvelur,CPIM,PMK,16985
Kinathukadavu,AIADMK,DMK,1095
Kolathur,DMK,AIADMK,70384
Kovilpatti,AIADMK,AMMK,12403
Krishnagiri,AIADMK,DMK,794
Krishnarayapuram,DMK,AIADMK,31625
Kulithalai,DMK,AIADMK,23540
Kumarapalayam,AIADMK,DMK,31646
Kumbakonam,DMK,AIADMK,21383
Kunnam,DMK,AIADMK,6329
Kurinjipadi,DMK,AIADMK,17527
Lalgudi,DMK,AIADMK,16949
Madathukulam,AIADMK,DMK,6438
Madavaram,DMK,AIADMK,57071
Madurai Central,DMK,AIADMK,34176
Madurai East,DMK,AIADMK,49604
Madurai North,DMK,BJP,22916
Madurai South,DMK,AIADMK,6515
Madurai West,AIADMK,DMK,9121
Madurantakam,AIADMK,DMK,3570
Maduravoyal,DMK,AIADMK,31721
Mailam,PMK,DMK,2230
Manachanallur,DMK,AIADMK,59618
Manamadurai,DMK,AIADMK,14091
Manapparai,DMK,AIADMK,12243
Mannargudi,DMK,AIADMK,37393
Mayiladuthurai,INC,PMK,2742
Melur,AIADMK,INC,35162
Mettuppalayam,AIADMK,DMK,2456
Mettur,PMK,DMK,656
Modakkurichi,BJP,DMK,281
Mudhukulathur,DMK,AIADMK,20721
Musiri,DMK,AIADMK,26836
Mylapore,DMK,AIADMK,12633
Nagapattinam,VCK,AIADMK,7238
Nagercoil,BJP,DMK,11669
Namakkal,DMK,AIADMK,27861
Nanguneri,INC,AIADMK,16486
Nannilam,AIADMK,DMK,4424
Natham,AIADMK,DMK,11932
Neyveli,DMK,PMK,977
Nilakkottai,AIADMK,DMK,27618
Oddanchatram,DMK,AIADMK,28742
Omalur,AIADMK,INC,55294
Orathanadu,AIADMK,DMK,28835
Ottapidaram,DMK,AIADMK,8510
Padmanabhapuram,DMK,AIADMK,26885
Palacode,AIADMK,DMK,28100
Palani,DMK,AIADMK,30056
Palayamkottai,DMK,AIADMK,52141
Palladam,AIADMK,DMK,32691
Pallavaram,DMK,AIADMK,37781
Panruti,DMK,AIADMK,4697
Papanasam,DMK,AIADMK,16273
Paramakudi,DMK,AIADMK,13285
Paramathi-velur,AIADMK,DMK,7662
Perambalur,DMK,AIADMK,31034
Perambur,DMK,AIADMK,54976
Peravurani,DMK,AIADMK,23503
Periyakulam,DMK,AIADMK,21321
Perundurai,AIADMK,DMK,14507
Pollachi,AIADMK,DMK,1725
Polur,AIADMK,DMK,9725
Ponneri,INC,AIADMK,9689
Poonamallee,DMK,PMK,94110
Poompuhar,DMK,AIADMK,3299
Pudukkottai,DMK,AIADMK,13001
Radhapuram,DMK,AIADMK,5925
Rajapalayam,DMK,AIADMK,3898
Ramanathapuram,DMK,BJP,50479
Ranipet,DMK,AIADMK,16498
Rasipuram,DMK,AIADMK,1952
Rishivandiyam,DMK,AIADMK,41728
Royapuram,DMK,AIADMK,27779
Salem (north),DMK,AIADMK,7588
Salem (south),AIADMK,DMK,22609
Salem (west),PMK,DMK,21499
Sankarankovil,DMK,AIADMK,5297
Sankarapuram,DMK,PMK,45963
Sankari,AIADMK,DMK,20045
Senthamangalam,DMK,AIADMK,10493
Sholavandan,DMK,AIADMK,17045
Sholinghur,INC,PMK,26698
Shozhinganallur,DMK,AIADMK,35405
Singanallur,AIADMK,DMK,10854
Sirkazhi,DMK,AIADMK,12148
Sivaganga,AIADMK,CPI,11253
Sivakasi,INC,AIADMK,17319
Sriperumbudur,INC,AIADMK,10879
Srirangam,DMK,AIADMK,19915
Srivaikuntam,INC,AIADMK,17372
Srivilliputhur,AIADMK,INC,12738
Sulur,AIADMK,DMK,31932
Tambaram,DMK,AIADMK,36824
Tenkasi,INC,AIADMK,370
Thalli,CPI,BJP,56226
Thanjavur,DMK,AIADMK,47149
Thiru-vi-ka-nagar,DMK,AIADMK,55013
Thirumangalam,AIADMK,DMK,14087
Thirumayam,DMK,AIADMK,1382
Thiruparankundram,AIADMK,CPIM,29489
Thiruporur,VCK,PMK,1947
Thiruthuraipoondi,CPI,AIADMK,30068
Thiruvaiyaru,DMK,BJP,53650
Thiruvallur,DMK,AIADMK,22701
Thiruvarur,DMK,AIADMK,51174
Thiruverumbur,DMK,AIADMK,49697
Thiruvidaimarudur,DMK,AIADMK,10680
Thiruvottiyur,DMK,AIADMK,37661
Thiyagarayanagar,DMK,AIADMK,137
Thondamuthur,AIADMK,DMK,41630
Thoothukkudi,DMK,AIADMK,50310
Thousand Lights,DMK,BJP,32462
Thuraiyur,DMK,AIADMK,22071
Tindivanam,AIADMK,DMK,9753
Tiruchendur,DMK,AIADMK,25263
Tiruchengodu,DMK,AIADMK,2862
Tiruchirappalli (east),DMK,AIADMK,53797
Tiruchirappalli (west),DMK,AIADMK,85109
Tiruchuli,DMK,AIADMK,60992
Tirukkoyilur,DMK,BJP,59680
Tirunelveli,BJP,DMK,23107
Tiruppattur,DMK,AIADMK,28240
Tiruppur (north),AIADMK,CPI,40102
Tiruppur (south),DMK,AIADMK,4709
Tiruttani,DMK,AIADMK,29253
Tiruvadanai,INC,AIADMK,13852
Tiruvannamalai,DMK,BJP,94673
Tittakudi,DMK,BJP,21563
Udhagamandalam,INC,BJP,5348
Udumalaipettai,AIADMK,INC,21895
Ulundurpettai,DMK,AIADMK,5256
Usilampatti,AIADMK,DMK,7477
Uthangarai,AIADMK,INC,28387
Uthiramerur,DMK,AIADMK,1622
Valparai,AIADMK,CPI,12223
Vandavasi,DMK,PMK,35953
Vaniyambadi,AIADMK,IUML,4904
Vanur,AIADMK,VCK,21727
Vasudevanallur,DMK,AIADMK,2367
Vedaranyam,AIADMK,DMK,12329
Vedasandur,DMK,AIADMK,17553
Veerapandi,AIADMK,DMK,19895
Velachery,INC,AIADMK,4352
Vellore,DMK,AIADMK,9181
Veppanahalli,AIADMK,DMK,3054
Vikravandi,DMK,AIADMK,9573
Vilathikulam,DMK,AIADMK,38549
Vilavancode,INC,BJP,28669
Villivakkam,DMK,AIADMK,37237
Villupuram,DMK,AIADMK,14868
Viralimalai,AIADMK,DMK,23598
Virudhunagar,DMK,BJP,21339
Virugampakkam,DMK,AIADMK,18367
Vriddhachalam,INC,PMK,862
Yercaud,AIADMK,DMK,25955"""

# The statisticstimes data only has ~222 distinct rows (some missing, plus a
# duplicated Tiruppattur row). We'll also overlay any missing winners from
# myneta 2021 winners list below. "Tiruppattur" appears twice in TN (ac 61 Tiruppattur
# in Vellore area and ac 174 Tirupattur in Sivaganga) - indiavotes returned one,
# statisticstimes returned it twice. We'll attempt to match via canon() best-effort.

# --- 2021 myneta winners (for backfill) --------------------------------------

RAW_2021_MYNETA = """ALANGUDI,DMK
ALANGULAM,AIADMK
AMBASAMUDRAM,AIADMK
AMBATTUR,DMK
AMBUR,DMK
ANAIKATTU,DMK
ANDIPATTI,DMK
ANNA NAGAR,DMK
ARAKKONAM,AIADMK
ARANI,AIADMK
ARANTHANGI,INC
ARAVAKURICHI,DMK
ARCOT,DMK
ARIYALUR,DMK
ARUPPUKOTTAI,DMK
ATHOOR,DMK
AVADI,DMK
AVANASHI,AIADMK
BARGUR,DMK
BHAVANI,AIADMK
BHAVANISAGAR,AIADMK
BHUVANAGIRI,AIADMK
BODINAYAKKANUR,AIADMK
CHENGALPATTU,DMK
CHEPAUK-THIRUVALLIKENI,DMK
CHEYYAR,DMK
CHEYYUR,VCK
CHIDAMBARAM,AIADMK
COIMBATORE NORTH,AIADMK
COIMBATORE SOUTH,BJP
COLACHAL,INC
COONOOR,DMK
CUMBUM,DMK
DHARAPURAM,DMK
DHARMAPURI,PMK
DINDIGUL,AIADMK
DR.RADHAKRISHNAN NAGAR,DMK
EDAPPADI,AIADMK
EGMORE,DMK
ERODE EAST,INC
GANDARVAKOTTAI,CPI(M)
GANGAVALLI,AIADMK
GINGEE,DMK
GOBICHETTIPALAYAM,AIADMK
GUDALUR,AIADMK
GUDIYATTAM,DMK
GUMMIDIPOONDI,DMK
HARBOUR,DMK
HOSUR,DMK
JAYANKONDAM,DMK
JOLARPET,DMK
KADAYANALLUR,AIADMK
KALASAPAKKAM,DMK
KALLAKURICHI,AIADMK
KANCHEEPURAM,DMK
KANGAYAM,DMK
KARAIKUDI,INC
KARUR,DMK
KATPADI,DMK
KATTUMANNARKOIL,VCK
KAVUNDAMPALAYAM,AIADMK
KILLIYOOR,INC
KILPENNATHUR,DMK
KILVAITHINANKUPPAM,AIADMK
KINATHUKADAVU,AIADMK
KOLATHUR,DMK
KOVILPATTI,AIADMK
KRISHNAGIRI,AIADMK
KRISHNARAYAPURAM,DMK
KULITHALAI,DMK
KUMARAPALAYAM,AIADMK
KUMBAKONAM,DMK
KURINJIPADI,DMK
LALGUDI,DMK
MADATHUKULAM,AIADMK
MADHAVARAM,DMK
MADHURAVOYAL,DMK
MADURAI CENTRAL,DMK
MADURAI EAST,DMK
MADURAI NORTH,DMK
MADURAI WEST,AIADMK
MADURANTAKAM,AIADMK
MAILAM,PMK
MANACHANALLUR,DMK
MANAMADURAI,DMK
MANAPPARAI,DMK
MANNARGUDI,DMK
MAYILADUTHURAI,INC
METTUPALAYAM,AIADMK
METTUR,PMK
MODAKKURICHI,BJP
MUDUKULATHUR,DMK
MUSIRI,DMK
NAGAPATTINAM,VCK
NAGERCOIL,BJP
NAMAKKAL,DMK
NATHAM,AIADMK
NEYVELI,DMK
NILAKKOTTAI,AIADMK
ODDANCHATRAM,DMK
OMALUR,AIADMK
ORATHANADU,AIADMK
OTTAPIDARAM,DMK
PADMANABHAPURAM,DMK
PALANI,DMK
PALAYAMKOTTAI,DMK
PALLADAM,AIADMK
PALLAVARAM,DMK
PANRUTI,DMK
PAPANASAM,DMK
PAPPIREDDIPATTI,AIADMK
PARAMAKUDI,DMK
PATTUKKOTTAI,DMK
PENNAGARAM,PMK
PERAMBALUR,DMK
PERAMBUR,DMK
PERAVURANI,DMK
PERIYAKULAM,DMK
PERUNDURAI,AIADMK
POLLACHI,AIADMK
PONNERI,INC
POONAMALLEE,DMK
PUDUKKOTTAI,DMK
RADHAPURAM,DMK
RAMANATHAPURAM,DMK
RANIPET,DMK
RASIPURAM,DMK
RISHIVANDIYAM,DMK
SALEM NORTH,DMK
SALEM SOUTH,AIADMK
SALEM WEST,PMK
SANKARANKOVIL,DMK
SANKARAPURAM,DMK
SANKARI,AIADMK
SATTUR,DMK
SENTHAMANGALAM,DMK
SHOLINGANALLUR,DMK
SHOLINGHUR,INC
SINGANALLUR,AIADMK
SIRKAZHI,DMK
SIVAGANGA,AIADMK
SIVAKASI,INC
SRIPERUMBUDUR,INC
SRIRANGAM,DMK
SRIVILLIPUTHUR,AIADMK
SULUR,AIADMK
TAMBARAM,DMK
THALLY,CPI
THANJAVUR,DMK
THIRU-VI-KA-NAGAR,DMK
THIRUMANGALAM,AIADMK
THIRUMAYAM,DMK
THIRUPORUR,VCK
THIRUTHURAIPOONDI,CPI
THIRUVAIYARU,DMK
THIRUVAUR,DMK
THIRUVERUMBUR,DMK
THIRUVOTTIYUR,DMK
THIYAGARAYANAGAR,DMK
THONDAMUTHUR,AIADMK
THOUSAND LIGHTS,DMK
THURAIYUR,DMK
TINDIVANAM,AIADMK
TIRUCHENDUR,DMK
TIRUCHENGODU,DMK
TIRUCHIRAPPALLI EAST,DMK
TIRUCHIRAPPALLI WEST,DMK
TIRUCHULI,DMK
TIRUNELVELI,BJP
TIRUPATTUR,DMK
TIRUPPATHUR,DMK
TIRUPPUR NORTH,AIADMK
TIRUPPUR SOUTH,DMK
TIRUTTANI,DMK
TIRUVADANAI,INC
TIRUVANNAMALAI,DMK
UDHAGAMANDALAM,INC
UDUMALAIPETTAI,AIADMK
ULUNDURPETTAI,DMK
USILAMPATTI,AIADMK
UTHANGARAI,AIADMK
UTHIRAMERUR,DMK
VALPARAI,AIADMK
VANIYAMBADI,AIADMK
VASUDEVANALLUR,DMK
VEDASANDUR,DMK
VEDHARANYAM,AIADMK
VEERAPANDI,AIADMK
VELACHERY,INC
VELLORE,DMK
VEPPANAHALLI,AIADMK
VIKRAVANDI,DMK
VILAVANCODE,INC
VILLIVAKKAM,DMK
VILLUPURAM,DMK
VIRALIMALAI,AIADMK
VIRUDHUNAGAR,DMK
VIRUGAMPAKKAM,DMK
VRIDHACHALAM,INC
YERCAUD,AIADMK"""

# --- 2016 myneta winners ----------------------------------------------------

RAW_2016_MYNETA = """ALANDUR,DMK
ALANGUDI,DMK
ALANGULAM,DMK
AMBASAMUDRAM,AIADMK
AMBATTUR,AIADMK
AMBUR,AIADMK
ANAIKATTU,DMK
ANNA NAGAR,DMK
ARAKKONAM,AIADMK
ARANI,AIADMK
ARANTHANGI,AIADMK
ARCOT,DMK
ARIYALUR,AIADMK
ARUPPUKKOTTAI,DMK
ATHOOR,DMK
ATTUR,AIADMK
AVANASHI,AIADMK
BARGUR,AIADMK
BHAVANI,AIADMK
BHAVANISAGAR,AIADMK
BODINAYAKANUR,AIADMK
CHENGALPATTU,DMK
CHENGAM,DMK
CHEYYAR,AIADMK
CHIDAMBARAM,AIADMK
COIMBATORE (SOUTH),AIADMK
COLACHAL,INC
COONOOR,AIADMK
CUDDALORE,AIADMK
CUMBUM,AIADMK
DHARAPURAM,INC
DHARMAPURI,DMK
DR.RADHAKRISHNAN NAGAR,AIADMK
EDAPPADI,AIADMK
EGMORE,DMK
ERODE (EAST),AIADMK
ERODE (WEST),AIADMK
GANDHARVAKOTTAI,AIADMK
GINGEE,DMK
GOBICHETTIPALAYAM,AIADMK
GUDIYATTAM,AIADMK
GUMMIDIPOONDI,AIADMK
HARBOUR,DMK
HARUR,AIADMK
HOSUR,AIADMK
JAYANKONDAM,AIADMK
KADAYANALLUR,IUML
KALASAPAKKAM,AIADMK
KANCHEEPURAM,DMK
KANGAYAM,AIADMK
KANNIYAKUMARI,DMK
KARAIKUDI,INC
KARUR,AIADMK
KATPADI,DMK
KATTUMANNARKOIL,AIADMK
KAVUNDAMPALAYAM,AIADMK
KILPENNATHUR,DMK
KILVAITHINANKUPPAM,AIADMK
KILVELUR,DMK
KINATHUKADAVU,AIADMK
KOLATHUR,DMK
KOVILPATTI,AIADMK
KRISHNAGIRI,DMK
KRISHNARAYAPURAM,AIADMK
KUMARAPALAYAM,AIADMK
KUMBAKONAM,DMK
KUNNAM,AIADMK
KURINJIPADI,DMK
LALGUDI,DMK
MADATHUKULAM,DMK
MADAVARAM,DMK
MADURAI CENTRAL,DMK
MADURAI NORTH,AIADMK
MADURAI SOUTH,AIADMK
MADURAI WEST,AIADMK
MADURANTAKAM,DMK
MADURAVOYAL,AIADMK
MAILAM,DMK
MANACHANALLUR,AIADMK
MANAMADURAI,AIADMK
MANNARGUDI,DMK
MAYILADUTHURAI,AIADMK
MELUR,AIADMK
METTUPPALAYAM,AIADMK
METTUR,AIADMK
MODAKKURICHI,AIADMK
MUDHUKULATHUR,INC
MUSIRI,AIADMK
NAGAPATTINAM,AIADMK
NAGERCOIL,DMK
NAMAKKAL,AIADMK
NANGUNERI,INC
NANNILAM,AIADMK
NATHAM,DMK
NEYVELI,DMK
NILAKKOTTAI,AIADMK
OMALUR,AIADMK
ORATHANADU,DMK
OTTAPIDARAM,AIADMK
PADMANABHAPURAM,DMK
PALACODU,AIADMK
PALANI,DMK
PALAYAMKOTTAI,DMK
PALLAVARAM,DMK
PAPPIREDDIPPATTI,AIADMK
PARAMAKUDI,AIADMK
PARAMATHI-VELUR,DMK
PATTUKKOTTAI,AIADMK
PENNAGARAM,DMK
PERAMBALUR,AIADMK
PERAMBUR,AIADMK
PERAVURANI,AIADMK
PERUNDURAI,AIADMK
POLLACHI,AIADMK
POLUR,DMK
PONNERI,AIADMK
POOMPUHAR,AIADMK
POONMALLAE,AIADMK
PUDUKKOTTAI,DMK
RADHAPURAM,AIADMK
RAMANATHAPURAM,AIADMK
RANIPET,DMK
RASIPURAM,AIADMK
RISHIVANDIYAM,DMK
ROYAPURAM,AIADMK
SAIDAPET,DMK
SALEM (NORTH),DMK
SALEM (SOUTH),AIADMK
SANKARANKOVIL,AIADMK
SANKARAPURAM,DMK
SANKARI,AIADMK
SATTUR,AIADMK
SENTHAMANGALAM,AIADMK
SHOLAVANDAN,AIADMK
SHOLINGUR,AIADMK
SHOZHINGANALLUR,DMK
SIRKAZHI,AIADMK
SIVAGANGA,AIADMK
SIVAKASI,AIADMK
SRIPERUMBUDUR,AIADMK
SRIRANGAM,AIADMK
SRIVAIKUNTAM,AIADMK
SRIVILLIPUTHUR,AIADMK
SULUR,AIADMK
THALLI,DMK
THIRU-VI-KA-NAGAR,DMK
THIRUMANGALAM,AIADMK
THIRUMAYAM,DMK
THIRUPARANKUNDRAM,AIADMK
THIRUPORUR,AIADMK
THIRUTHURAIPOONDI,DMK
THIRUVAIYARU,DMK
THIRUVARUR,DMK
THIRUVERUMBUR,DMK
THIRUVIDAIMARUDUR,DMK
THIYAGARAYANAGAR,AIADMK
THONDAMUTHUR,AIADMK
THOOTHUKKUDI,DMK
THOUSAND LIGHTS,DMK
THURAIYUR,DMK
TIRUCHENDUR,DMK
TIRUCHENGODU,AIADMK
TIRUCHIRAPPALLI (EAST),AIADMK
TIRUCHIRAPPALLI (WEST),DMK
TIRUCHULI,DMK
TIRUKKOYILUR,DMK
TIRUNELVELI,DMK
TIRUPATTUR,DMK
TIRUPPUR (SOUTH),AIADMK
TIRUTTANI,AIADMK
TIRUVADANAI,AIADMK
TIRUVANNAMALAI,DMK
TITTAKUDI,DMK
UDHAGAMANDALAM,INC
UDUMALAIPETTAI,AIADMK
ULUNDURPETTAI,AIADMK
UTHIRAMERUR,DMK
VALPARAI,AIADMK
VANDAVASI,DMK
VANIYAMBADI,AIADMK
VANUR,AIADMK
VASUDEVANALLUR,AIADMK
VEDARANYAM,AIADMK
VEERAPANDI,AIADMK
VELLORE,DMK
VEPPANAHALLI,DMK
VIKRAVANDI,DMK
VILAVANCODE,INC
VILLIVAKKAM,DMK
VILLUPURAM,AIADMK
VIRALIMALAI,AIADMK
VIRUDHUNAGAR,DMK
VRIDDHACHALAM,AIADMK
YERCAUD,AIADMK"""


def parse_stime(raw):
    """Parse statisticstimes-style rows: name,winner,runner,margin -> dict keyed by canon(name)."""
    out = {}
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[0]
        wp = norm_party(parts[1])
        rp = norm_party(parts[2])
        try:
            margin = int(parts[3]) if len(parts) > 3 and parts[3] else 0
        except ValueError:
            margin = 0
        out[canon(name)] = {"winner_party": wp, "runner_up_party": rp, "margin": margin}
    return out


def parse_myneta(raw):
    out = {}
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        name, wp = parts
        out[canon(name)] = norm_party(wp)
    return out


RAW_2016_FULL = """Gummidipoondi,AIADMK,89332,DMK,65937
Ponneri,AIADMK,95979,DMK,76643
Tiruttani,AIADMK,93045,INC,69904
Thiruvallur,DMK,80473,AIADMK,75335
Poonamallee,AIADMK,103952,DMK,92189
Avadi,AIADMK,108064,DMK,106669
Maduravoyal,AIADMK,99739,INC,91337
Ambattur,AIADMK,94375,INC,76877
Madavaram,DMK,122082,AIADMK,106829
Thiruvottiyur,DMK,82205,AIADMK,77342
Dr.Radhakrishnan Nagar,AIADMK,97218,DMK,57673
Perambur,AIADMK,79974,DMK,79455
Kolathur,DMK,91303,AIADMK,53573
Villivakkam,DMK,65972,AIADMK,56651
Thiru-Vi-Ka-Nagar,DMK,61744,AIADMK,58422
Egmore,DMK,55060,AIADMK,44381
Royapuram,AIADMK,55205,INC,47174
Harbour,DMK,42071,AIADMK,37235
Chepauk-Thiruvallikeni,DMK,67982,AIADMK,53818
Thousand Lights,DMK,61726,AIADMK,52897
Anna Nagar,DMK,72207,AIADMK,70520
Virugampakkam,AIADMK,65979,DMK,63646
Saidapet,DMK,79279,AIADMK,63024
Thiyagarayanagar,AIADMK,53207,DMK,50052
Mylapore,AIADMK,68176,INC,53448
Velachery,DMK,70139,AIADMK,61267
Shozhinganallur,DMK,147014,AIADMK,132101
Alandur,DMK,96877,AIADMK,77708
Sriperumbudur,AIADMK,101001,INC,90285
Pallavaram,DMK,112891,AIADMK,90726
Tambaram,DMK,101835,AIADMK,87390
Chengalpattu,DMK,112675,AIADMK,86383
Thiruporur,AIADMK,70215,DMK,69265
Cheyyur,DMK,63446,AIADMK,63142
Madurantakam,DMK,73693,AIADMK,70736
Uthiramerur,DMK,85513,AIADMK,73357
Kancheepuram,DMK,90533,AIADMK,82985
Arakkonam,AIADMK,68176,DMK,64015
Sholingur,AIADMK,77651,INC,67919
Katpadi,DMK,90534,AIADMK,66588
Ranipet,DMK,81724,AIADMK,73828
Arcot,DMK,84182,AIADMK,73091
Vellore,DMK,88264,AIADMK,62054
Anaikattu,DMK,77058,AIADMK,68290
Kilvaithinankuppam,AIADMK,75612,DMK,65866
Gudiyattam,AIADMK,94689,DMK,83219
Vaniyambadi,AIADMK,69588,IUML,55062
Ambur,AIADMK,79182,DMK,51176
Jolarpet,AIADMK,82525,DMK,71534
Tirupattur,DMK,80791,AIADMK,73144
Uthangarai,AIADMK,69980,DMK,67367
Bargur,AIADMK,80650,DMK,79668
Krishnagiri,DMK,87637,AIADMK,82746
Veppanahalli,DMK,88952,AIADMK,83724
Hosur,AIADMK,89510,INC,66546
Thalli,DMK,74429,CPI,68184
Palacode,AIADMK,76143,DMK,70160
Pennagaram,DMK,76848,PMK,58402
Dharmapuri,DMK,71056,AIADMK,61380
Pappireddippatti,AIADMK,74234,PMK,61521
Harur,AIADMK,64568,DMK,53147
Chengam,DMK,95939,AIADMK,83248
Tiruvannamalai,DMK,116484,AIADMK,66136
Kilpennathur,DMK,99070,AIADMK,64404
Kalasapakkam,AIADMK,84394,INC,57980
Polur,DMK,66588,AIADMK,58315
Arani,AIADMK,94074,DMK,86747
Cheyyar,AIADMK,77766,INC,69239
Vandavasi,DMK,80206,AIADMK,62138
Gingee,DMK,88440,AIADMK,66383
Mailam,DMK,70880,AIADMK,58574
Tindivanam,DMK,61879,AIADMK,61778
Vanur,AIADMK,64167,DMK,53944
Villupuram,AIADMK,69421,IUML,47130
Vikravandi,DMK,63757,AIADMK,56845
Tirukkoyilur,DMK,93837,AIADMK,52780
Ulundurpettai,AIADMK,81973,DMK,77809
Rishivandiyam,DMK,92607,AIADMK,72104
Sankarapuram,DMK,90920,AIADMK,76392
Kallakurichi,AIADMK,90108,DMK,86004
Gangavalli,AIADMK,74301,DMK,72039
Attur,AIADMK,82827,INC,65493
Yercaud,AIADMK,100562,DMK,83168
Omalur,AIADMK,89169,DMK,69213
Mettur,AIADMK,72751,DMK,66469
Edappadi,AIADMK,98703,PMK,56681
Sankari,AIADMK,96202,INC,58828
Salem (West),AIADMK,80755,DMK,73508
Salem (North),DMK,86583,AIADMK,76710
Salem (South),AIADMK,101696,DMK,71243
Veerapandi,AIADMK,94792,DMK,80311
Rasipuram,AIADMK,86901,DMK,77270
Senthamangalam,AIADMK,91339,DMK,79006
Namakkal,AIADMK,89076,INC,75542
Paramathi-Velur,DMK,74418,AIADMK,73600
Tiruchengodu,AIADMK,73103,DMK,69713
Kumarapalayam,AIADMK,103032,DMK,55703
Erode (East),AIADMK,64879,DMK,57085
Erode (West),AIADMK,82297,DMK,77391
Modakkurichi,AIADMK,77067,DMK,74845
Dharapuram,INC,83538,AIADMK,73521
Kangayam,AIADMK,83325,INC,70190
Perundurai,AIADMK,80292,DMK,67521
Bhavani,AIADMK,85748,DMK,60861
Anthiyur,AIADMK,71575,DMK,66263
Gobichettipalayam,AIADMK,96177,INC,84954
Bhavanisagar,AIADMK,83006,DMK,69902
Udhagamandalam,INC,67747,AIADMK,57329
Gudalur,DMK,62128,AIADMK,48749
Coonoor,AIADMK,61650,DMK,57940
Mettuppalayam,AIADMK,93595,DMK,77481
Avanashi,AIADMK,93366,DMK,62692
Tiruppur (North),AIADMK,106717,DMK,68943
Tiruppur (South),AIADMK,73351,DMK,57418
Palladam,AIADMK,111866,DMK,79692
Sulur,AIADMK,100977,INC,64346
Kavundampalayam,AIADMK,110870,DMK,102845
Coimbatore (North),AIADMK,77540,DMK,69816
Thondamuthur,AIADMK,109519,DMK,45478
Coimbatore (South),AIADMK,59788,INC,42369
Singanallur,DMK,75459,AIADMK,70279
Kinathukadavu,AIADMK,89042,DMK,87710
Pollachi,AIADMK,78553,DMK,65185
Valparai,AIADMK,69980,DMK,61736
Udumalaipettai,AIADMK,81817,DMK,76130
Madathukulam,DMK,76619,AIADMK,74952
Palani,DMK,100045,AIADMK,74459
Oddanchatram,DMK,121715,AIADMK,55988
Athoor,DMK,121738,AIADMK,94591
Nilakkottai,AIADMK,85507,DMK,70731
Natham,DMK,93822,AIADMK,91712
Dindigul,AIADMK,91413,DMK,70694
Vedasandur,AIADMK,97555,INC,77617
Aravakurichi,,,
Karur,AIADMK,81936,INC,81495
Krishnarayapuram,AIADMK,83977,IND,48676
Kulithalai,DMK,89923,AIADMK,78027
Manapparai,AIADMK,91399,IUML,73122
Srirangam,AIADMK,108400,DMK,93991
Tiruchirappalli (West),DMK,92049,AIADMK,63634
Tiruchirappalli (East),AIADMK,79938,INC,58044
Thiruverumbur,DMK,85950,AIADMK,69255
Lalgudi,DMK,77946,AIADMK,74109
Manachanallur,AIADMK,83083,DMK,75561
Musiri,AIADMK,89398,INC,57311
Thuraiyur,DMK,81444,AIADMK,73376
Perambalur,AIADMK,101073,DMK,94220
Kunnam,AIADMK,78218,DMK,59422
Ariyalur,AIADMK,88523,DMK,86480
Jayankondam,AIADMK,75672,PMK,52738
Tittakudi,DMK,65139,AIADMK,62927
Vridhachalam,AIADMK,72611,DMK,58834
Neyveli,DMK,54299,AIADMK,36508
Panruti,AIADMK,72353,DMK,69225
Cuddalore,AIADMK,70922,DMK,46509
Kurinjipadi,DMK,82864,AIADMK,54756
Bhuvanagiri,DMK,60554,AIADMK,55066
Chidambaram,AIADMK,58543,DMK,57037
Kattumannarkoil,AIADMK,48450,VCK,48363
Sirkazhi,AIADMK,76487,DMK,67484
Mayiladuthurai,AIADMK,70949,DMK,66171
Poompuhar,AIADMK,87666,IUML,67731
Nagapattinam,AIADMK,64903,DMK,44353
Kilvelur,DMK,61999,AIADMK,51829
Vedaranyam,AIADMK,60836,INC,37838
Thiruthuraipoondi,DMK,72127,AIADMK,58877
Mannargudi,DMK,91137,AIADMK,81200
Thiruvarur,DMK,121473,AIADMK,53107
Nannilam,AIADMK,100918,INC,79642
Thiruvidaimarudur,DMK,77538,AIADMK,77006
Kumbakonam,DMK,85048,AIADMK,76591
Papanasam,AIADMK,82614,INC,58249
Thiruvaiyaru,DMK,100043,AIADMK,85700
Thanjavur,,,
Orathanadu,DMK,84378,AIADMK,80733
Pattukkottai,AIADMK,70631,INC,58273
Peravurani,AIADMK,73908,DMK,72913
Gandharvakottai,AIADMK,64043,DMK,60996
Viralimalai,AIADMK,84701,DMK,76254
Pudukkottai,DMK,66739,AIADMK,64655
Thirumayam,DMK,72373,AIADMK,71607
Alangudi,DMK,72992,AIADMK,63051
Aranthangi,AIADMK,69905,INC,67614
Karaikudi,INC,93419,AIADMK,75136
Tiruppattur,DMK,110719,AIADMK,68715
Sivaganga,AIADMK,81697,DMK,75061
Manamadurai,AIADMK,89893,DMK,75004
Melur,AIADMK,88909,DMK,69186
Madurai East,DMK,108569,AIADMK,75797
Sholavandan,AIADMK,87044,DMK,62187
Madurai North,AIADMK,70460,INC,51621
Madurai South,AIADMK,62683,DMK,38920
Madurai Central,DMK,64662,AIADMK,58900
Madurai West,AIADMK,82529,DMK,66131
Thiruparankundram,AIADMK,93453,DMK,70461
Thirumangalam,AIADMK,95864,INC,72274
Usilampatti,AIADMK,106349,DMK,73443
Andipatti,AIADMK,103129,DMK,72933
Periyakulam,AIADMK,90599,DMK,76249
Bodinayakanur,AIADMK,99531,DMK,83923
Cumbum,AIADMK,91099,DMK,79878
Rajapalayam,DMK,74787,AIADMK,69985
Srivilliputhur,AIADMK,88103,IND,51430
Sattur,AIADMK,71513,DMK,67086
Sivakasi,AIADMK,76734,INC,61986
Virudhunagar,DMK,65499,AIADMK,62629
Aruppukkottai,DMK,81485,AIADMK,63431
Tiruchuli,DMK,89927,AIADMK,63350
Paramakudi,AIADMK,79254,DMK,67865
Tiruvadanai,AIADMK,76786,DMK,68090
Ramanathapuram,AIADMK,89365,DMK,56143
Mudhukulathur,INC,94946,AIADMK,81598
Vilathikulam,AIADMK,71496,DMK,52778
Thoothukkudi,DMK,88045,AIADMK,67137
Tiruchendur,DMK,88357,AIADMK,62356
Srivaikuntam,AIADMK,65198,INC,61667
Ottapidaram,AIADMK,65071,IND,64578
Kovilpatti,AIADMK,64514,DMK,64086
Sankarankovil,AIADMK,78751,DMK,64262
Vasudevanallur,AIADMK,73904,IND,55146
Kadayanallur,IUML,70763,AIADMK,69569
Tenkasi,AIADMK,86339,INC,85877
Alangulam,DMK,88891,AIADMK,84137
Tirunelveli,DMK,81761,AIADMK,81160
Ambasamudram,AIADMK,78555,DMK,65389
Palayamkottai,DMK,67463,AIADMK,51591
Nanguneri,INC,74932,AIADMK,57617
Radhapuram,AIADMK,69590,DMK,69541
Kanniyakumari,DMK,89023,AIADMK,83111
Nagercoil,DMK,67369,BJP,46413
Colachel,INC,67195,BJP,41167
Padmanabhapuram,DMK,76249,AIADMK,35344
Vilavancode,INC,68789,BJP,35646
Killiyoor,INC,77356,BJP,31061"""


def parse_full(raw):
    """Parse rows like: name,winner_party,winner_votes,runner_up_party,runner_up_votes
    Returns dict keyed by canon(name)."""
    out = {}
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        name = parts[0]
        wp = norm_party(parts[1]) if len(parts) > 1 else ""
        try:
            wv = int(parts[2]) if len(parts) > 2 and parts[2] else 0
        except ValueError:
            wv = 0
        rp = norm_party(parts[3]) if len(parts) > 3 else ""
        try:
            rv = int(parts[4]) if len(parts) > 4 and parts[4] else 0
        except ValueError:
            rv = 0
        out[canon(name)] = {
            "winner_party": wp,
            "winner_votes": wv,
            "runner_up_party": rp,
            "runner_up_votes": rv,
        }
    return out


res_2021_stime = parse_stime(RAW_2021)
res_2021_myneta = parse_myneta(RAW_2021_MYNETA)
res_2016_myneta = parse_myneta(RAW_2016_MYNETA)
res_2016_full = parse_full(RAW_2016_FULL)

# Aravakurichi and Thanjavur 2016 general-election polling was countermanded
# due to cash-for-votes probes and held as byelections on 19 Nov 2016.
# Both were won by AIADMK (source: myneta bye-election list, confirmed via ECI).
# Vote counts not recorded in our bulk sources; leave as 0 but flag real.
res_2016_full[canon("Aravakurichi")] = {
    "winner_party": "AIADMK", "winner_votes": 0,
    "runner_up_party": "DMK", "runner_up_votes": 0,
}
res_2016_full[canon("Thanjavur")] = {
    "winner_party": "AIADMK", "winner_votes": 0,
    "runner_up_party": "DMK", "runner_up_votes": 0,
}

# Supplementary manual entries for rows the bulk sources missed.
# Saidapet 2021: DMK beat AIADMK by 29,408 votes (confirmed via web search).
res_2021_stime.setdefault(canon("Saidapet"), {
    "winner_party": "DMK", "runner_up_party": "AIADMK", "margin": 29408,
})

print(f"2021 statisticstimes rows: {len(res_2021_stime)}")
print(f"2021 myneta rows: {len(res_2021_myneta)}")
print(f"2016 myneta rows: {len(res_2016_myneta)}")
print(f"2016 full (elections.in) rows: {len(res_2016_full)}")


# --- Build 2021 CSV ---------------------------------------------------------

rows_2021 = []
unmatched_2021 = []
for _, r in master.iterrows():
    ac_no = int(r["ac_no"])
    ac_name = r["ac_name"]
    key = r["_key"]
    winner_party = ""
    runner_up_party = ""
    is_real = 0

    d = res_2021_stime.get(key)
    if d:
        winner_party = d["winner_party"]
        runner_up_party = d["runner_up_party"]
        is_real = 1
    else:
        wp = res_2021_myneta.get(key, "")
        if wp:
            winner_party = wp
            is_real = 1
        else:
            unmatched_2021.append((ac_no, ac_name, key))

    rows_2021.append({
        "ac_no": ac_no,
        "ac_name": ac_name,
        "winner_party": winner_party,
        "winner_alliance": alliance_for(winner_party, 2021) if winner_party else "",
        "runner_up_party": runner_up_party,
        "runner_up_alliance": alliance_for(runner_up_party, 2021) if runner_up_party else "",
        "winner_votes": 0,
        "runner_up_votes": 0,
        "total_valid_votes": 0,
        "is_real_result": is_real,
    })

df_2021 = pd.DataFrame(rows_2021)


# --- Build 2016 CSV ---------------------------------------------------------

rows_2016 = []
unmatched_2016 = []
for _, r in master.iterrows():
    ac_no = int(r["ac_no"])
    ac_name = r["ac_name"]
    key = r["_key"]
    winner_party = ""
    runner_up_party = ""
    winner_votes = 0
    runner_up_votes = 0
    is_real = 0

    d = res_2016_full.get(key)
    if d and d["winner_party"]:
        winner_party = d["winner_party"]
        runner_up_party = d["runner_up_party"]
        winner_votes = d["winner_votes"]
        runner_up_votes = d["runner_up_votes"]
        is_real = 1
    else:
        # Fallback: myneta 2016 winners list (no votes, no runner-up).
        wp = res_2016_myneta.get(key, "")
        if wp:
            winner_party = wp
            is_real = 1
        else:
            unmatched_2016.append((ac_no, ac_name, key))

    total_valid = (winner_votes + runner_up_votes) if (winner_votes and runner_up_votes) else 0

    rows_2016.append({
        "ac_no": ac_no,
        "ac_name": ac_name,
        "winner_party": winner_party,
        "winner_alliance": alliance_for(winner_party, 2016) if winner_party else "",
        "runner_up_party": runner_up_party,
        "runner_up_alliance": alliance_for(runner_up_party, 2016) if runner_up_party else "",
        "winner_votes": winner_votes,
        "runner_up_votes": runner_up_votes,
        "total_valid_votes": total_valid,
        "is_real_result": is_real,
    })

df_2016 = pd.DataFrame(rows_2016)


# --- Diagnostics ------------------------------------------------------------

print()
print(f"Unmatched 2021 ACs ({len(unmatched_2021)}):")
for ac_no, name, key in unmatched_2021:
    print(f"  {ac_no}  {name}  -> key={key}")
print()
print(f"Unmatched 2016 ACs ({len(unmatched_2016)}):")
for ac_no, name, key in unmatched_2016:
    print(f"  {ac_no}  {name}  -> key={key}")
print()


# --- Save --------------------------------------------------------------------

path_2016 = os.path.join(OUT_DIR, "tamilnadu_assembly_2016_results.csv")
path_2021 = os.path.join(OUT_DIR, "tamilnadu_assembly_2021_results.csv")

df_2016.to_csv(path_2016, index=False)
df_2021.to_csv(path_2021, index=False)


# --- Summary -----------------------------------------------------------------

print("=" * 60)
print("SUMMARY")
print("=" * 60)
for year, path, df in [(2016, path_2016, df_2016), (2021, path_2021, df_2021)]:
    print()
    print(f"Year: {year}")
    print(f"  Path: {path}")
    print(f"  Rows: {len(df)}")
    print(f"  winner_alliance value counts:")
    vc = df["winner_alliance"].value_counts(dropna=False)
    for k, v in vc.items():
        print(f"    {k!r}: {v}")
    print(f"  is_real_result split:")
    rc = df["is_real_result"].value_counts()
    for k, v in rc.items():
        print(f"    {k}: {v}")
    print(f"  winner_party value counts:")
    pc = df["winner_party"].value_counts(dropna=False).head(20)
    for k, v in pc.items():
        print(f"    {k!r}: {v}")
