import numpy as np
import pandas as pd
import pytest

from clean import (
    standardize_emails,
    standardize_names,
    standardize_phone_numbers,
    standardize_regions,
    validate_emails,
)


# --- standardize_emails ---

def test_standardize_emails_lowercases():
    result = standardize_emails(pd.Series(["USER@EXAMPLE.COM"]))
    assert result.iloc[0] == "user@example.com"


def test_standardize_emails_strips_whitespace():
    result = standardize_emails(pd.Series([" user@example.com "]))
    assert result.iloc[0] == "user@example.com"


def test_standardize_emails_removes_internal_spaces():
    result = standardize_emails(pd.Series(["user @example.com"]))
    assert result.iloc[0] == "user@example.com"


def test_standardize_emails_placeholder_nan_to_nan():
    result = standardize_emails(pd.Series(["nan", "none", ""]))
    assert result.isna().all()


# --- validate_emails ---

def test_validate_emails_valid():
    series = pd.Series(["user@example.com", "a.b+c@domain.co.uk"])
    assert validate_emails(series).all()


def test_validate_emails_invalid():
    series = pd.Series(["not-an-email", "missing@", "@nodomain.com", "double@@sign.com"])
    assert not validate_emails(series).any()


def test_validate_emails_nan_returns_false():
    # validate_emails is always called after standardize_emails, which produces
    # an object-dtype series where NaN values coexist with strings.
    result = validate_emails(pd.Series([np.nan], dtype=object))
    assert result.iloc[0] == False


# --- standardize_phone_numbers ---

def test_standardize_phone_strips_formatting():
    result = standardize_phone_numbers(pd.Series(["555.123.4567"]))
    assert result.iloc[0] == "5551234567"


def test_standardize_phone_preserves_international_prefix():
    result = standardize_phone_numbers(pd.Series(["+1 (555) 123-4567"]))
    assert result.iloc[0] == "+15551234567"


def test_standardize_phone_too_short_returns_nan():
    result = standardize_phone_numbers(pd.Series(["123"]))
    assert pd.isna(result.iloc[0])


def test_standardize_phone_non_numeric_returns_nan():
    result = standardize_phone_numbers(pd.Series(["invalid-phone"]))
    assert pd.isna(result.iloc[0])


def test_standardize_phone_nan_input_returns_nan():
    result = standardize_phone_numbers(pd.Series([np.nan]))
    assert pd.isna(result.iloc[0])


# --- standardize_names ---

def test_standardize_names_title_case():
    result = standardize_names(pd.Series(["john smith"]))
    assert result.iloc[0] == "John Smith"


def test_standardize_names_collapses_extra_whitespace():
    result = standardize_names(pd.Series(["  john   smith  "]))
    assert result.iloc[0] == "John Smith"


def test_standardize_names_placeholder_to_nan():
    result = standardize_names(pd.Series(["nan", "none", ""]))
    assert result.isna().all()


# --- standardize_regions ---

def test_standardize_regions_us_variants():
    variants = pd.Series(["us", "USA", "united states", "North America", "AMER"])
    result = standardize_regions(variants)
    assert (result == "US").all()


def test_standardize_regions_eu_variants():
    variants = pd.Series(["eu", "europe", "EMEA", "European Union"])
    result = standardize_regions(variants)
    assert (result == "EU").all()


def test_standardize_regions_apac_variants():
    variants = pd.Series(["apac", "Asia Pacific", "asia", "AP"])
    result = standardize_regions(variants)
    assert (result == "APAC").all()


def test_standardize_regions_unmapped_returns_nan():
    result = standardize_regions(pd.Series(["Mars", "unknown"]))
    assert result.isna().all()
