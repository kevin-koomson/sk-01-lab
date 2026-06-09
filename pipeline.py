"""
ShopStream Customer Data Quality Pipeline
==========================================
Ingests customer data from 4 sources, cleans, deduplicates, validates,
and produces a golden customer record dataset.

Author: Kevin Koomson
Run: python pipeline.py
"""
from datetime import datetime

from clean import clean_dataframe
from config import CONFIG, logger
from dedup import deduplicate_customers
from ingest import ingest_all_sources
from tests.generate_data import generate_synthetic_data
from validate import run_quality_checks
from visualize import generate_eda_report


def run_pipeline():
    """Orchestrates: Ingest → Clean → Deduplicate → Validate → Visualize → Export"""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("SHOPSTREAM CUSTOMER DATA QUALITY PIPELINE")
    logger.info(f"Run started: {start_time.isoformat()}")
    logger.info("=" * 60)

    combined = ingest_all_sources()
    input_count = len(combined)

    cleaned = clean_dataframe(combined)

    deduped = deduplicate_customers(cleaned)

    # Uncomment to enable AI region inference (requires ANTHROPIC_API_KEY):
    # from enrich import infer_region_with_llm
    # deduped = infer_region_with_llm(deduped)

    logger.info("STEP 4: Quality Validation")
    quality_report = run_quality_checks(deduped)

    generate_eda_report(deduped, CONFIG["output_dir"])

    logger.info("STEP 6: Exporting Results")

    deduped["opt_out"] = deduped["opt_out"].fillna(False).astype(bool)
    if "source_count" in deduped.columns:
        deduped["source_count"] = deduped["source_count"].fillna(1).astype(int)

    parquet_path = CONFIG["output_dir"] / "golden_customers.parquet"
    deduped.to_parquet(parquet_path, index=False, engine="pyarrow", compression="gzip")
    logger.info(f"  Parquet export: {parquet_path} ({parquet_path.stat().st_size / 1024:.1f} KB)")

    csv_path = CONFIG["output_dir"] / "golden_customers.csv"
    deduped.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"  CSV export: {csv_path}")

    report_path = CONFIG["output_dir"] / "quality_report.csv"
    quality_report.to_csv(report_path, index=False)
    logger.info(f"  Quality report: {report_path}")

    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Input records:           {input_count:,}")
    logger.info(f"  Output (golden) records: {len(deduped):,}")
    logger.info(f"  Duplicates removed:      {input_count - len(deduped):,}")
    logger.info(f"  Quality checks passed:   {(quality_report['status'] == 'PASS').sum()}/{len(quality_report)}")
    logger.info(f"  Duration:                {duration:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_synthetic_data()
    run_pipeline()
