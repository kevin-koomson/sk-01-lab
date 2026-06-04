import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pipeline")

CONFIG = {
    "input_dir": Path("data/raw"),
    "output_dir": Path("data/processed"),
    "crm_api_url": "https://api.shopstream.example.com/v2/customers",
    "crm_api_key": "sk-xxxx",
    "valid_regions": ["US", "EU", "APAC"],
    "email_regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "quality_threshold": 0.95,
    "source_priority": {"crm": 1, "website": 2, "erp": 3, "marketing": 4},
}

for d in [CONFIG["input_dir"], CONFIG["output_dir"]]:
    d.mkdir(parents=True, exist_ok=True)

STANDARD_SCHEMA = [
    "email", "first_name", "last_name", "phone", "region",
    "registration_date", "opt_out", "source"
]

REGION_MAP = {
    "us": "US", "usa": "US", "united states": "US", "north america": "US",
    "na": "US", "amer": "US", "america": "US",
    "eu": "EU", "europe": "EU", "emea": "EU", "eur": "EU", "european union": "EU",
    "apac": "APAC", "asia": "APAC", "asia pacific": "APAC",
    "ap": "APAC", "asia-pacific": "APAC",
}
