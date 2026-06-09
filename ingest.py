import json

import numpy as np
import pandas as pd
import requests

from config import CONFIG, STANDARD_SCHEMA, logger


def _normalize_columns(df):
    """Lowercase, strip, and snake_case all column names."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def ingest_website_csv(filepath):
    """
    Ingest the website registration CSV export.
    Handles ISO-8859-1 encoding, removes test accounts, normalises column names.
    """
    logger.info(f"Ingesting website CSV: {filepath}")

    df = pd.read_csv(
        filepath,
        encoding="iso-8859-1",
        dtype={"Phone": str},
        parse_dates=["Registration Date"],
        na_values=["", "N/A", "null", "NULL", "none", "NaN"],
    ).pipe(_normalize_columns)

    df = df.rename(columns={
        "customeremail": "email",
        "optout": "opt_out",
    })

    test_mask = df["email"].str.contains(r"@test\.shopstream\.com$", na=False, case=False)
    removed = test_mask.sum()
    df = df[~test_mask].copy()
    logger.info(f"  Removed {removed} test accounts")

    df["source"] = "website"
    logger.info(f"  Ingested {len(df)} records from website CSV")
    return df


def ingest_crm_json(filepath):
    """
    Ingest customer data from the CRM JSON export.
    Flattens nested profile keys using pd.json_normalize.
    """
    logger.info(f"Ingesting CRM JSON: {filepath}")

    raw = json.loads(filepath.read_text())
    df = pd.json_normalize(raw["customers"], sep="_")

    df = df.rename(columns={
        "profile_first_name": "first_name",
        "profile_last_name": "last_name",
        "registration_date": "registration_date",
    })

    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["source"] = "crm"

    logger.info(f"  Ingested {len(df)} records from CRM JSON")
    return df


def ingest_crm_api(api_url, api_key):
    """
    Ingest customer data from the CRM REST API with pagination.
    Use this in production instead of ingest_crm_json().
    """
    logger.info("Ingesting CRM API (paginated)...")
    all_records = []
    page = 1

    while True:
        response = requests.get(
            api_url,
            headers={"Authorization": f"Bearer {api_key}"},
            params={"page": page, "per_page": 500},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("customers"):
            break

        all_records.extend(data["customers"])
        logger.info(f"  Fetched page {page} ({len(data['customers'])} records)")
        page += 1

        if page > data.get("total_pages", 1):
            break

    df = pd.json_normalize(all_records, sep="_")
    df["source"] = "crm"
    logger.info(f"  Ingested {len(df)} records from CRM API")
    return df


def ingest_erp_fixed_width(filepath):
    """
    Ingest the legacy ERP fixed-width text file.

    Field layout (ERP system spec v3.2):
        [0:10]    customer_id
        [10:60]   full_name
        [60:120]  email
        [120:140] phone
        [140:145] region_code
        [145:155] registration_date
        [155:160] status
    """
    logger.info(f"Ingesting ERP fixed-width: {filepath}")

    colspecs = [
        (0, 10),
        (10, 60),
        (60, 120),
        (120, 140),
        (140, 145),
        (145, 155),
        (155, 160),
    ]
    col_names = ["customer_id", "full_name", "email", "phone",
                 "region_code", "registration_date", "status"]

    df = pd.read_fwf(
        filepath,
        colspecs=colspecs,
        names=col_names,
        dtype=str,
        encoding="utf-8",
    )

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    name_split = df["full_name"].str.split(n=1, expand=True)
    df["first_name"] = name_split[0] if 0 in name_split.columns else np.nan
    df["last_name"] = name_split[1] if 1 in name_split.columns else np.nan

    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["region"] = df["region_code"]
    df["source"] = "erp"

    logger.info(f"  Ingested {len(df)} records from ERP")
    return df


def align_schema(df):
    """Align a source DataFrame to STANDARD_SCHEMA. Missing cols → NaN, extras dropped."""
    for col in STANDARD_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan
    return df[STANDARD_SCHEMA].copy()


def ingest_all_sources():
    """Ingest all sources and combine into a single raw DataFrame."""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Ingestion")

    frames = []

    website_df = ingest_website_csv(CONFIG["input_dir"] / "website_customers.csv")
    frames.append(align_schema(website_df))

    crm_df = ingest_crm_json(CONFIG["input_dir"] / "crm_export.json")
    frames.append(align_schema(crm_df))

    erp_df = ingest_erp_fixed_width(CONFIG["input_dir"] / "erp_customers.txt")
    frames.append(align_schema(erp_df))

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total records combined: {len(combined)}")
    for source in combined["source"].unique():
        count = (combined["source"] == source).sum()
        logger.info(f"  {source}: {count} records")

    return combined
