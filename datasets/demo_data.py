import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from configs.logging_config import logger

# --- Hand-authored seed articles (kept from the original project) ---
_SEED_ARTICLES = [
    {
        "id": "news_001",
        "title": "Government Approves New National Clean Energy & Solar Grid Expansion Plan",
        "text": "The Ministry of Energy today announced a landmark initiative to expand renewable solar grids across major metropolitan regions. The project aims to reduce overall carbon emissions by 40% over the next decade. Industry experts have praised the move as a major step toward sustainable infrastructure.",
        "label": 0, "source": "PolitiFact", "author": "National Desk", "published_date": "2026-03-15",
    },
    {
        "id": "news_002",
        "title": "Secret NASA Satellite Confirms Aliens Installed Pyramid-Shaped Beacon on Moon",
        "text": "Shocking leaks from unverified classified military whistleblowers claim NASA has been hiding a mysterious electromagnetic pyramid discovered on the far side of the moon. Insiders state alien technology is actively transmitting signals directly to subterranean bunkers.",
        "label": 1, "source": "GossipCop", "author": "TruthSeeker99", "published_date": "2026-03-18",
    },
    {
        "id": "news_003",
        "title": "Global Health Organization Releases Updated Guidelines on Hydration and Fitness",
        "text": "Health officials have published a comprehensive peer-reviewed review detailing daily hydration recommendations. Experts emphasize drinking adequate water during high-intensity exercise to maintain optimal cardiovascular health.",
        "label": 0, "source": "PolitiFact", "author": "Health Today", "published_date": "2026-04-01",
    },
    {
        "id": "news_004",
        "title": "Miracle Herbal Extract Instantly Cures All Known Viruses in 24 Hours",
        "text": "Pharmaceutical companies don't want you to know about this secret Amazonian root. Drinking a single spoonful of this exotic extract guarantees 100% immunity against all bacterial and viral infections without any side effects.",
        "label": 1, "source": "Twitter", "author": "ViralAlerts", "published_date": "2026-04-10",
    },
    {
        "id": "news_005",
        "title": "Central Bank Adjusts Interest Rates to Counteract Inflationary Pressures",
        "text": "The Reserve Board voted to raise key interest rates by 25 basis points in response to persistent consumer price increases. Economists suggest the moderate adjustment will help stabilize long-term market growth.",
        "label": 0, "source": "PolitiFact", "author": "Economic Times", "published_date": "2026-05-02",
    },
    {
        "id": "news_006",
        "title": "Billionaire Tech CEO Replaces Entire Board of Directors with Autonomous AI Bots",
        "text": "In an unprecedented corporate move, tech mogul Alex Sterling fired his entire executive board overnight and granted voting rights to a proprietary neural network. Sources say the AI voted to double executive salaries.",
        "label": 1, "source": "GossipCop", "author": "TechRumors", "published_date": "2026-05-12",
    },
]

# --- Template-generated additional articles ---
# Provides a much larger, more varied training corpus than the 6 seed
# articles alone, so TF-IDF/BERT-probe attribution reflects the vocabulary
# of *whatever article the user submits* rather than always gravitating
# toward the handful of words that happened to appear in a tiny hand-written
# training set.
_REAL_TEMPLATES = [
    "Officials from the {org} confirmed today that the {topic} initiative will proceed following a unanimous council vote.",
    "According to a peer-reviewed study published this week, {topic} outcomes have shown measurable improvement over the past year.",
    "The {org} released an official statement clarifying its updated {topic} policy, citing input from independent researchers.",
    "Local authorities reported steady, well-documented progress on the {topic} programme, with quarterly figures published for public review.",
    "A spokesperson for the {org} confirmed the {topic} results after they were independently verified by external auditors.",
    "Economists at the {org} noted that {topic} trends remained stable, attributing the change to routine seasonal adjustment.",
]
_FAKE_TEMPLATES = [
    "SHOCKING: leaked documents reveal {org} has been secretly hiding the truth about {topic} for years!",
    "You won't believe what whistleblowers say the {org} doesn't want you to know about {topic} — 100% proof inside.",
    "BREAKING: anonymous insiders expose a hidden {topic} conspiracy allegedly orchestrated by {org}.",
    "Miracle {topic} secret banned by {org} finally exposed — doctors hate this one simple trick.",
    "Unverified sources claim {org} is covering up a massive {topic} scandal that mainstream media refuses to report.",
    "Viral post alleges {org} secretly controls {topic} outcomes using technology hidden from the public.",
]
_TOPICS = [
    "vaccine rollout", "election security", "climate policy", "housing market",
    "renewable energy", "public transit funding", "school curriculum reform",
    "banking regulation", "space exploration budget", "agricultural subsidy",
]
_ORGS = [
    "Ministry of Health", "Election Commission", "National Weather Service",
    "Department of Housing", "Energy Regulatory Authority", "Transit Authority",
    "Board of Education", "Central Bank", "Space Research Organisation", "Farmers' Council",
]


def _generate_template_articles(n: int = 40, seed: int = 7) -> list:
    rng = np.random.default_rng(seed)
    articles = []
    for i in range(n):
        is_fake = bool(rng.integers(0, 2))
        template = rng.choice(_FAKE_TEMPLATES if is_fake else _REAL_TEMPLATES)
        topic = rng.choice(_TOPICS)
        org = rng.choice(_ORGS)
        text = str(template).format(topic=topic, org=org)
        idx = len(_SEED_ARTICLES) + i + 1
        articles.append({
            "id": f"news_{idx:03d}",
            "title": text.split(".")[0][:90],
            "text": text,
            "label": 1 if is_fake else 0,
            "source": "Synthetic-Demo-Corpus",
            "author": "Demo Generator",
            "published_date": "2026-06-01",
        })
    return articles


DEMO_ARTICLES = _SEED_ARTICLES + _generate_template_articles()


def generate_demo_images(output_dir: str = "datasets/sample_images"):
    """Generates synthetic authentic and doctored images for demonstration."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_files = {}

    img1 = Image.new("RGB", (300, 300), color=(41, 128, 185))
    draw = ImageDraw.Draw(img1)
    draw.rectangle([50, 50, 250, 250], fill=(236, 240, 241), outline=(52, 73, 94), width=4)
    draw.text((80, 140), "CLEAN NEWS GRAPHIC", fill=(44, 62, 80))
    p1 = out_path / "authentic_news.jpg"
    img1.save(p1, "JPEG", quality=95)
    img_files["authentic"] = str(p1)

    img2 = img1.copy()
    draw2 = ImageDraw.Draw(img2)
    forged_patch = Image.new("RGB", (80, 80), color=(231, 76, 60))
    img2.paste(forged_patch, (160, 60))
    p2 = out_path / "doctored_forgery.jpg"
    img2.save(p2, "JPEG", quality=70)
    img_files["doctored"] = str(p2)

    logger.info(f"Generated sample demo images in {out_path}")
    return img_files


def get_demo_df() -> pd.DataFrame:
    """Returns a pandas DataFrame of demo news records."""
    img_map = generate_demo_images()
    df = pd.DataFrame(DEMO_ARTICLES)
    df["image_path"] = [img_map["authentic"] if label == 0 else img_map["doctored"] for label in df["label"]]
    return df
