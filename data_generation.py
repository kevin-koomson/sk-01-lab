import json

import numpy as np
import pandas as pd

from config import CONFIG, logger


def generate_synthetic_data():
    """
    Generate 4 source datasets with realistic data quality problems.
    Run once to create test data files in data/raw/.
    """
    np.random.seed(42)
    n = 1000  # Smaller for the lab; in production this would be 200K+

    emails_pool = [
        f"customer{i}@{'gmail' if i % 3 == 0 else 'yahoo' if i % 3 == 1 else 'company'}.com"
        for i in range(800)
    ]
    emails_pool += ["not-an-email", "missing@", "@nodomain.com", "", "double@@sign.com"]

    first_names = [
        "Maria", "José", "André", "Léa", "François", "Müller", "O'Brien",
        "John", "Jane", "Mike", "Sarah", "Alex", "Chris", "Pat", "Sam"
    ] * 70
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Martínez", "Díaz", "López", "González", "Wang", "Kim"
    ] * 84

    regions_messy = (
        ["US", "us", "USA", "united states", "North America"] * 150 +
        ["EU", "eu", "Europe", "EMEA", "europe"] * 150 +
        ["APAC", "apac", "Asia Pacific", "Asia", "AP"] * 100 +
        [None, "", "N/A"] * 80
    )
    np.random.shuffle(regions_messy)

    phones_messy = (
        ["+1 (555) 123-4567", "555.123.4567", "5551234567",
         "+44 20 7946 0958", "020 7946 0958",
         "+81-3-1234-5678", "invalid-phone", None] * 125
    )
    np.random.shuffle(phones_messy)

    # --- SOURCE 1: Website CSV (ISO-8859-1 encoded) ---
    website_df = pd.DataFrame({
        "CustomerEmail": np.random.choice(emails_pool, n),
        "First Name": [first_names[i] for i in np.random.randint(0, len(first_names), n)],
        "Last Name": [last_names[i] for i in np.random.randint(0, len(last_names), n)],
        "Phone": [phones_messy[i % len(phones_messy)] for i in range(n)],
        "Region": [regions_messy[i % len(regions_messy)] for i in range(n)],
        "Registration Date": pd.date_range("2020-01-01", periods=n, freq="4h").strftime("%Y-%m-%d"),
        "OptOut": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })
    test_accounts = pd.DataFrame({
        "CustomerEmail": [f"test{i}@test.shopstream.com" for i in range(20)],
        "First Name": ["Test"] * 20,
        "Last Name": ["Account"] * 20,
        "Phone": [None] * 20,
        "Region": ["US"] * 20,
        "Registration Date": ["2023-01-01"] * 20,
        "OptOut": [0] * 20,
    })
    website_df = pd.concat([website_df, test_accounts], ignore_index=True)
    website_df.to_csv(CONFIG["input_dir"] / "website_customers.csv",
                      index=False, encoding="iso-8859-1")
    logger.info(f"Generated website CSV: {len(website_df)} records")

    # --- SOURCE 2: CRM Export (JSON format) ---
    crm_records = []
    for i in range(n // 2):
        crm_records.append({
            "id": f"CRM-{i:06d}",
            "email": np.random.choice(emails_pool),
            "profile": {
                "first_name": first_names[np.random.randint(0, len(first_names))],
                "last_name": last_names[np.random.randint(0, len(last_names))],
            },
            "phone": phones_messy[i % len(phones_messy)],
            "region": regions_messy[i % len(regions_messy)],
            "registration_date": f"202{np.random.randint(0,4)}-{np.random.randint(1,13):02d}-01",
            "opt_out": bool(np.random.choice([0, 1], p=[0.85, 0.15])),
            "lifetime_value": round(np.random.uniform(50, 5000), 2),
        })
    crm_path = CONFIG["input_dir"] / "crm_export.json"
    crm_path.write_text(json.dumps({"customers": crm_records}))
    logger.info(f"Generated CRM JSON: {len(crm_records)} records")

    # --- SOURCE 3: ERP Fixed-Width ---
    erp_lines = []
    for i in range(n // 4):
        email = np.random.choice(emails_pool)
        name = f"{first_names[i % len(first_names)]} {last_names[i % len(last_names)]}"
        phone = str(phones_messy[i % len(phones_messy)] or "")
        region = str(regions_messy[i % len(regions_messy)] or "")
        date = f"2019-{np.random.randint(1,13):02d}-01"
        status = np.random.choice(["ACTIV", "INACT"])
        line = (
            f"{str(i):>10}"
            f"{name:<50}"
            f"{email:<60}"
            f"{phone:<20}"
            f"{region:<5}"
            f"{date:<10}"
            f"{status:<5}"
        )
        erp_lines.append(line)
    (CONFIG["input_dir"] / "erp_customers.txt").write_text("\n".join(erp_lines), encoding="utf-8")
    logger.info(f"Generated ERP fixed-width: {len(erp_lines)} records")

    logger.info("Synthetic data generation complete.")
