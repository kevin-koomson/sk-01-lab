import pandas as pd
import pytest

from dedup import deduplicate_customers


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame with the columns dedup expects."""
    defaults = {
        "email": None, "first_name": None, "last_name": None,
        "phone": None, "region": None, "registration_date": None,
        "opt_out": False, "source": "website", "email_valid": True,
        "email_raw": None,
    }
    records = [{**defaults, **row} for row in rows]
    return pd.DataFrame(records)


# --- basic deduplication ---

def test_exact_email_match_merged_to_one_record():
    df = make_df([
        {"email": "a@b.com", "source": "website"},
        {"email": "a@b.com", "source": "crm"},
    ])
    result = deduplicate_customers(df)
    valid = result[result["email_valid"] == True]
    assert len(valid[valid["email"] == "a@b.com"]) == 1


def test_distinct_emails_both_retained():
    df = make_df([
        {"email": "a@b.com", "source": "website"},
        {"email": "x@y.com", "source": "crm"},
    ])
    result = deduplicate_customers(df)
    assert len(result[result["email_valid"] == True]) == 2


# --- source priority ---

def test_source_priority_crm_over_website():
    df = make_df([
        {"email": "a@b.com", "source": "website", "first_name": "Web"},
        {"email": "a@b.com", "source": "crm",     "first_name": "CRM"},
    ])
    result = deduplicate_customers(df)
    merged = result[result["email"] == "a@b.com"].iloc[0]
    assert merged["first_name"] == "CRM"


def test_fills_missing_field_from_lower_priority_source():
    df = make_df([
        {"email": "a@b.com", "source": "crm",     "phone": None,         "region": "US"},
        {"email": "a@b.com", "source": "website",  "phone": "5551234567", "region": "US"},
    ])
    result = deduplicate_customers(df)
    merged = result[result["email"] == "a@b.com"].iloc[0]
    assert merged["phone"] == "5551234567"


# --- GDPR opt-out ---

def test_gdpr_opt_out_any_source_opted_out():
    df = make_df([
        {"email": "a@b.com", "source": "crm",     "opt_out": False},
        {"email": "a@b.com", "source": "website",  "opt_out": True},
    ])
    result = deduplicate_customers(df)
    merged = result[result["email"] == "a@b.com"].iloc[0]
    assert merged["opt_out"] == True


def test_gdpr_opt_out_all_false_stays_false():
    df = make_df([
        {"email": "a@b.com", "source": "crm",     "opt_out": False},
        {"email": "a@b.com", "source": "website",  "opt_out": False},
    ])
    result = deduplicate_customers(df)
    merged = result[result["email"] == "a@b.com"].iloc[0]
    assert merged["opt_out"] == False


# --- source tracking ---

def test_sources_column_lists_all_contributing_sources():
    df = make_df([
        {"email": "a@b.com", "source": "crm"},
        {"email": "a@b.com", "source": "website"},
        {"email": "a@b.com", "source": "erp"},
    ])
    result = deduplicate_customers(df)
    merged = result[result["email"] == "a@b.com"].iloc[0]
    sources = set(merged["sources"].split(","))
    assert sources == {"crm", "website", "erp"}


def test_source_count_matches_unique_sources():
    df = make_df([
        {"email": "a@b.com", "source": "crm"},
        {"email": "a@b.com", "source": "website"},
    ])
    result = deduplicate_customers(df)
    merged = result[result["email"] == "a@b.com"].iloc[0]
    assert merged["source_count"] == 2


# --- invalid emails ---

def test_invalid_emails_not_deduplicated():
    df = make_df([
        {"email": "bad-email", "source": "website", "email_valid": False},
        {"email": "bad-email", "source": "crm",     "email_valid": False},
    ])
    result = deduplicate_customers(df)
    invalid = result[result["email_valid"] == False]
    assert len(invalid) == 2
