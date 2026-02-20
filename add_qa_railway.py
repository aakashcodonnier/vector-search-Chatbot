#!/usr/bin/env python3
"""Add Q&A pairs to Railway MySQL database."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mysql.connector
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Railway MySQL connection
conn = mysql.connector.connect(
    host="shinkansen.proxy.rlwy.net",
    user="root",
    password="RiUvcRBDuhxoDTtuDdnLCIFxwGDIlKvr",
    database="railway",
    port=29730
)
cur = conn.cursor()

QA_PAIRS = [
    {
        "title": "What makes MasterPeace Zeolite formula different from other detox products?",
        "url": "https://masterpeacebyhcs.com/product/masterpeace",
        "content": "Question: What makes the MasterPeace Zeolite formula different from other detox or zeolite products already on the market?\n\nAnswer: MasterPeace is by far the strongest cleansing formula of its kind, while still being gentle to take - thanks to our original breakthrough combination!\n#1: Microscopic organic zeolite - the master toxin binder.\n#2: Sea mineral plasma - essential, world-class nutrition.\n#3: Multiple water structuring methods.\nCleansing and nourishing, at the same time.\n\nLearn even more about the MasterPeace formula here:\n1. https://humanconsciousnesssupport.substack.com/p/what-is-masterpeace-zeolite-z-exactly\n2. https://masterpeacebyhcs.com/mp\nProduct comparison chart: https://masterpeacebyhcs.com/wp-content/uploads/HCSMPProdComparison.pdf"
    },
    {
        "title": "How to maintain low toxin levels long-term with MasterPeace",
        "url": "https://masterpeacebyhcs.com/long-term-detox",
        "content": "Question: If people are constantly exposed to environmental and nano-scale toxins, how can they maintain low levels of these compounds long-term?\n\nAnswer: Since the influx of novel nano poisons is provably constant and increasing, we recommend staying on MasterPeace long term. One can stop taking it for a while, cycling on and off, yet those who do so generally begin missing it.\n\nSee here for some perspectives on how long it takes to detox with MasterPeace and more info:\nhttps://rumble.com/v5e9pzp-how-long-does-it-take-to-detox-using-masterpeace.html\nhttps://rumble.com/v5fopbt-how-long-does-it-take-to-detox.html\nhttps://rumble.com/v5r6qd8-everybody-is-toxic.html"
    },
    {
        "title": "What is fog water contamination and why should you care?",
        "url": "https://drrobertyoung.com/insights-into-fog-water-composition-aerial-spraying-and-detoxification-solutions/",
        "content": "Question: What exactly is fog water contamination and why should I care about it?\n\nAnswer: Fog water contamination is a serious environmental and health concern. Learn more about the composition of this odd phenomenon from test results here: https://drrobertyoung.com/insights-into-fog-water-composition-aerial-spraying-and-detoxification-solutions/"
    },
    {
        "title": "How MasterPeace Zeolite protects from fog-borne and aerial spraying substances",
        "url": "https://masterpeacebyhcs.com/product/masterpeace-protection",
        "content": "Question: How does MasterPeace Zeolite work to protect me from these fog-borne or aerial spraying substances?\n\nAnswer: Harmful positively charged extra-small toxins, like heavy metals and microplastics, take up residence in body tissues. The health of all humans, animals, and our entire worldwide environment is inundated with positive-charged nano scale pollution. Unfortunately, this is a modern-day fact. However!\n\nMasterPeace with its safe and powerful NEGATIVE charge is proven to attract and bind incredible amounts of those extra-small toxins like a super magnet. The body then expels the binder with the poisons attached! At the same time, MasterPeace replaces the heavy metals and toxins with premium nutrient-dense sea plasma.\n\nThe MP formula is water structured multiple times, making it easier for the body to absorb. Water structuring, optimal pH (alkaline pH) and ORP (strong negative charge) all make the formula even more friendly to the body.\nBecome the best version of you!"
    },
    {
        "title": "MasterPeace Zeolite and long-term protection from aerial spraying and pollution",
        "url": "https://masterpeacebyhcs.com/long-term-protection",
        "content": "Question: If I use MasterPeace Zeolite, does that mean I do not need to worry about aerial spraying or fog-water pollution?\n\nAnswer: Since the influx of novel nano poisons is provably constant and increasing, we recommend staying on MasterPeace long term. One can stop taking it for a while, cycling on and off, yet those who do so generally begin missing it.\n\nSee here for some perspectives on how long it takes to detox with MasterPeace and more info:\nhttps://rumble.com/v5e9pzp-how-long-does-it-take-to-detox-using-masterpeace.html\nhttps://rumble.com/v5fopbt-how-long-does-it-take-to-detox.html\nhttps://rumble.com/v5r6qd8-everybody-is-toxic.html"
    },
    {
        "title": "5G 6G wireless technology health effects on human body",
        "url": "https://drrobertyoung.com/wireless-technology-health-effects",
        "content": "Question: Why should I be concerned about wireless technologies like 5G or 6G and how they affect my health?\n\nAnswer: According to Dr. Robert O. Young research on the negative effects of EMF on the human biofield, wireless technologies including 5G and 6G emit radio frequency electromagnetic fields (RF-EMFs) that can disrupt the delicate electrical balance of the human body at the cellular level. These frequencies can alter the body pH, damage cellular membranes, and create an acidic internal environment that promotes disease. Maintaining an alkaline pH and using detoxification methods like MasterPeace Zeolite Z can help mitigate the harmful effects of these environmental stressors on the human biofield."
    },
    {
        "title": "Practical steps to protect from RF-EMF electromagnetic radiation",
        "url": "https://drrobertyoung.com/protection-from-emf",
        "content": "Question: What practical steps can I take to protect myself from the possible harmful effects of RF-EMFs?\n\nAnswer: Practical steps to protect yourself from RF-EMF exposure include: maintaining distance from wireless devices when possible, turning off WiFi routers at night, using wired connections instead of wireless, grounding/earthing practices to neutralize positive charges, maintaining an alkaline diet rich in green vegetables and structured water to support the body natural electrical balance, and using MasterPeace Zeolite Z for detoxification of nano-scale pollutants that may be activated by electromagnetic frequencies. A strong alkaline internal environment helps the body resist the acidifying effects of EMF exposure."
    },
    {
        "title": "Detoxification for wireless radiation and nano-material exposures",
        "url": "https://drrobertyoung.com/detox-wireless-radiation",
        "content": "Question: Does detoxification really help in dealing with wireless radiation and nano-material exposures, and if so, how?\n\nAnswer: Detoxification is essential for dealing with wireless radiation and nano-material exposures. The body accumulates positively charged nano-scale toxins including graphene oxide, ferric oxide, and silicon-based compounds that can interact with and be activated by electromagnetic frequencies. MasterPeace Zeolite Z, with its powerful negative charge, acts as a broad-spectrum antidote by attracting and binding these nano-scale toxins like a super magnet, allowing the body to expel them naturally. At the same time, it replaces toxic materials with nutrient-dense sea plasma minerals. Combined with an alkaline lifestyle, structured water, and proper nutrition, detoxification helps restore the body natural biofield and cellular health."
    },
    {
        "title": "MasterPeace Zeolite Z Pilot Study - Safety and Efficacy for Removing Toxins",
        "url": "https://drrobertyoung.com/masterpeace-zeolite-pilot-study",
        "content": "Question: How can a Pilot Study of three-persons prove that MasterPeace Zeolite acts as a broad spectrum antidote for removing heavy metals like lead and mercury, micro plastics, like polyethylene glycol and polypropylene, forever chemicals like perfluoroalkyl acids, herbicides like glyphosate, transgender chemicals like atrazine, nano and micro biochipping the human body with graphene oxide, ferric oxide, and silicone based products that emit radio and microwave frequencies, from the human body?\n\nAnswer: A three person study was conducted on MasterPeace Zeolite Z using a nonaffiliated CLIA licensed lab to validate the safety and efficacy and then having the results of the study double-blinded peer-reviewed by an international scientific journal, ACTA Medical Science, is the unquestionable gold standard in validating the safety and efficacy of MasterPeace Zeolite Z for removal of over 28 known toxins found in the human lymphocytes of all three humans tested and the successful removal of over 70 percent of all toxins within the 60 day testing period, proving that taking MasterPeace Zeolite Z under the tongue, 3 times a day, is safe and effective with zero negative side effects.\n\nReferences:\nRobert O. Young and Caroline A. Mansfield, MasterPeace Zeolite Z Pilot Study Found to be Safe and Effective in Removing Nano and Micro Toxic Forever Chemicals, Heavy Metals, Micro Plastics, and Graphene, and Aluminum Found in the Human Body."
    },
]

added = 0
skipped = 0

for qa in QA_PAIRS:
    cur.execute("SELECT id FROM dr_young_all_articles WHERE title = %s", (qa["title"],))
    if cur.fetchone():
        print(f"  [SKIP] {qa['title'][:60]}...")
        skipped += 1
        continue

    print(f"  [EMBED] {qa['title'][:60]}...")
    embedding = embed_model.encode(qa["content"], convert_to_numpy=True)
    embedding_str = str(embedding.tolist())

    cur.execute(
        "INSERT INTO dr_young_all_articles (title, url, content, embedding) VALUES (%s, %s, %s, %s)",
        (qa["title"], qa["url"], qa["content"], embedding_str)
    )
    added += 1
    print(f"  [ADDED] {qa['title'][:60]}...")

conn.commit()

cur.execute("SELECT COUNT(*) FROM dr_young_all_articles")
total = cur.fetchone()[0]

cur.close()
conn.close()

print(f"\nResults: {added} added, {skipped} skipped")
print(f"Total Railway articles: {total}")
