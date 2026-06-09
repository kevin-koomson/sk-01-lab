import pandas as pd
import pytest

from validate import DataQualityValidator


def make_validator(data: dict, threshold=0.95) -> DataQualityValidator:
    return DataQualityValidator(pd.DataFrame(data), threshold=threshold)


# --- check_not_null ---

def test_check_not_null_pass():
    v = make_validator({"email": ["a@b.com", "c@d.com"]})
    result = v.check_not_null("email", "desc")
    assert result["status"] == "PASS"
    assert result["failed"] == 0


def test_check_not_null_fail():
    v = make_validator({"email": ["a@b.com", None, None, None, None]}, threshold=0.95)
    result = v.check_not_null("email", "desc")
    assert result["status"] == "FAIL"
    assert result["failed"] == 4


def test_check_not_null_pass_rate_calculation():
    v = make_validator({"col": ["a", "b", None, None, "c", "d", "e", "f", "g", "h"]})
    result = v.check_not_null("col", "desc")
    assert result["pass_rate"] == pytest.approx(0.8, abs=0.01)


# --- check_unique ---

def test_check_unique_pass():
    v = make_validator({"email": ["a@b.com", "c@d.com", "e@f.com"]})
    result = v.check_unique("email", "desc")
    assert result["status"] == "PASS"
    assert result["failed"] == 0


def test_check_unique_fail():
    v = make_validator({"email": ["dup@b.com"] * 10 + ["unique@b.com"]}, threshold=0.95)
    result = v.check_unique("email", "desc")
    assert result["status"] == "FAIL"


def test_check_unique_ignores_nan():
    v = make_validator({"email": ["a@b.com", None, None]})
    result = v.check_unique("email", "desc")
    assert result["failed"] == 0


# --- check_regex ---

def test_check_regex_pass():
    v = make_validator({"email": ["a@b.com", "x@y.org"]})
    result = v.check_regex("email", r"^[^@]+@[^@]+\.[^@]+$", "desc")
    assert result["status"] == "PASS"


def test_check_regex_fail():
    v = make_validator({"email": ["not-valid"] * 5 + ["ok@b.com"]}, threshold=0.95)
    result = v.check_regex("email", r"^[^@]+@[^@]+\.[^@]+$", "desc")
    assert result["status"] == "FAIL"


# --- check_values_in_set ---

def test_check_values_in_set_pass():
    v = make_validator({"region": ["US", "EU", "APAC"]})
    result = v.check_values_in_set("region", ["US", "EU", "APAC"], "desc")
    assert result["status"] == "PASS"
    assert result["failed"] == 0


def test_check_values_in_set_fail():
    v = make_validator({"region": ["US", "INVALID"] * 5}, threshold=0.95)
    result = v.check_values_in_set("region", ["US", "EU", "APAC"], "desc")
    assert result["status"] == "FAIL"


# --- check_date_range ---

def test_check_date_range_pass():
    v = make_validator({"date": pd.to_datetime(["2021-01-01", "2022-06-15"])})
    result = v.check_date_range("date", "2020-01-01", "2030-01-01", "desc")
    assert result["status"] == "PASS"
    assert result["failed"] == 0


def test_check_date_range_fail():
    dates = pd.to_datetime(["1990-01-01"] * 5 + ["2022-01-01"])
    v = DataQualityValidator(pd.DataFrame({"date": dates}), threshold=0.95)
    result = v.check_date_range("date", "2000-01-01", "2030-01-01", "desc")
    assert result["status"] == "FAIL"


# --- generate_report ---

def test_generate_report_returns_dataframe():
    v = make_validator({"email": ["a@b.com", "c@d.com"], "region": ["US", "EU"]})
    v.check_not_null("email", "desc")
    v.check_values_in_set("region", ["US", "EU", "APAC"], "desc")
    report = v.generate_report()
    assert isinstance(report, pd.DataFrame)
    assert len(report) == 2
    assert set(report.columns) >= {"check", "status", "pass_rate", "failed", "passed"}


def test_generate_report_overall_pass():
    v = make_validator({"email": ["a@b.com", "c@d.com"]})
    v.check_not_null("email", "desc")
    report = v.generate_report()
    assert (report["status"] == "PASS").all()
