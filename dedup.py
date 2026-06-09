import pandas as pd

from config import CONFIG, logger


def deduplicate_customers(df):
    """
    Deduplicate customer records across sources.

    Strategy:
        1. Sort by source priority (CRM is most trusted)
        2. Group by standardized email
        3. Merge fields: take first non-null from highest-priority source
        4. GDPR: if ANY source has opt_out=True, the record is opted out
    """
    logger.info("STEP 3: Deduplication")
    initial_count = len(df)

    df["source_priority"] = df["source"].map(CONFIG["source_priority"]).fillna(99)
    df = df.sort_values("source_priority")

    def merge_group(group):
        best = group.iloc[0].copy()

        for col in ["phone", "region", "first_name", "last_name", "registration_date"]:
            if col in group.columns and pd.isna(best.get(col)):
                non_null = group[col].dropna()
                if len(non_null) > 0:
                    best[col] = non_null.iloc[0]

        if "opt_out" in group.columns:
            best["opt_out"] = group["opt_out"].fillna(0).astype(bool).any()

        best["sources"] = ",".join(group["source"].unique())
        best["source_count"] = len(group["source"].unique())

        return best

    valid_mask = df["email_valid"] == True
    valid_df = df[valid_mask]
    invalid_df = df[~valid_mask]

    # reset_index(drop=False) restores the groupby key ("email") as a column
    deduped = (
        valid_df
        .groupby("email", sort=False)
        .apply(merge_group, include_groups=False)
        .reset_index(drop=False)
    )

    result = pd.concat([deduped, invalid_df], ignore_index=True)

    removed = initial_count - len(result)
    logger.info(f"  Records before deduplication: {initial_count}")
    logger.info(f"  Duplicate records removed: {removed}")
    logger.info(f"  Records after deduplication: {len(result)}")
    return result
