import json

from config import CONFIG, logger


def infer_region_with_llm(df):
    """
    Use Claude to infer missing region values from available context.
    Only processes records with null region — never overwrites existing values.
    Requires: pip install anthropic and ANTHROPIC_API_KEY env var.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed. Skipping LLM region inference.")
        df["region_inferred"] = False
        return df

    client = anthropic.Anthropic()
    null_region_mask = df["region"].isna()
    null_count = null_region_mask.sum()

    if null_count == 0:
        logger.info("No null regions to infer.")
        df["region_inferred"] = False
        return df

    logger.info(f"Inferring region for {null_count} records using LLM...")
    df = df.copy()
    df["region_inferred"] = False

    null_records = df[null_region_mask].copy()
    batch_size = 10
    inferred_regions = {}

    for i in range(0, len(null_records), batch_size):
        batch = null_records.iloc[i:i + batch_size]
        records_text = "\n".join([
            f"  - Index {idx}: email={row.get('email', 'unknown')}, "
            f"name={row.get('first_name', '')} {row.get('last_name', '')}"
            for idx, row in batch.iterrows()
        ])

        prompt = f"""You are a data engineer. Based on the following customer records,
infer the most likely geographic region for each. The only valid regions are:
- US (United States and Canada)
- EU (Europe, Middle East, Africa)
- APAC (Asia Pacific, Australia, New Zealand)

Use email domains, name patterns, and any other available signals.
If you truly cannot infer, respond with UNKNOWN.

Records:
{records_text}

Respond ONLY in JSON format like:
{{"results": [{{"index": <index>, "region": "<US|EU|APAC|UNKNOWN>", "confidence": "<high|medium|low>", "reason": "<brief reason>"}}]}}"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            result = json.loads(response.content[0].text)
            for item in result["results"]:
                idx = item["index"]
                region = item["region"]
                if region in CONFIG["valid_regions"]:
                    inferred_regions[idx] = region
                    logger.info(
                        f"  Inferred region for index {idx}: {region} "
                        f"(confidence: {item['confidence']}, reason: {item['reason']})"
                    )
        except Exception as e:
            logger.warning(f"  LLM inference failed for batch {i}-{i+batch_size}: {e}")

    for idx, region in inferred_regions.items():
        df.at[idx, "region"] = region
        df.at[idx, "region_inferred"] = True

    inferred_count = df["region_inferred"].sum()
    still_null = df["region"].isna().sum()
    logger.info(f"  Regions inferred by LLM: {inferred_count}")
    logger.info(f"  Regions still null (UNKNOWN or inference failed): {still_null}")
    return df
