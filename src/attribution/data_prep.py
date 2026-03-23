"""
Extract user journeys from GA4 BigQuery sample dataset for attribution modeling.

Queries bigquery-public-data.ga4_obfuscated_sample_ecommerce to build
touchpoint sequences per user, identifying conversion paths.
"""
from google.cloud import bigquery
import pandas as pd
import sys
import os

# Add parent directory to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GCP_PROJECT, GA4_DATASET, CHANNEL_MAPPING, DEFAULT_CHANNEL, MIN_TOUCHPOINTS


def get_bigquery_client():
    """Create BigQuery client using project credentials."""
    return bigquery.Client(project=GCP_PROJECT)


def extract_journeys(client=None, min_touchpoints=MIN_TOUCHPOINTS):
    """
    Extract user journey data from GA4 sample dataset.

    Returns a DataFrame with columns:
    - user_id: unique user identifier
    - journey_path: string of channels separated by ' > '
    - channel_list: list of channel names in order
    - has_conversion: 1 if journey ended in purchase, 0 otherwise
    - conversion_value: total purchase value (0 if no conversion)
    - journey_length: number of touchpoints
    - first_visit_date: date of first touchpoint
    - last_visit_date: date of last touchpoint
    """
    if client is None:
        client = get_bigquery_client()

    query = f"""
    WITH session_traffic AS (
        -- Extract traffic source per session per user
        -- Use traffic_source fields (available in GA4 sample dataset)
        SELECT
            user_pseudo_id,
            (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS ga_session_id,
            event_date,
            -- Traffic source from the top-level traffic_source fields
            COALESCE(NULLIF(traffic_source.medium, ''), '(none)') AS channel_medium,
            COALESCE(NULLIF(traffic_source.source, ''), '(direct)') AS channel_source,
            -- Check if this event is a purchase
            CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END AS is_purchase,
            CASE WHEN event_name = 'purchase' THEN ecommerce.purchase_revenue ELSE 0 END AS purchase_value,
            event_timestamp
        FROM
            `{GA4_DATASET}.events_*`
        WHERE
            user_pseudo_id IS NOT NULL
    ),

    session_level AS (
        -- Aggregate to one row per session
        SELECT
            user_pseudo_id,
            ga_session_id,
            MIN(event_date) AS event_date,
            -- Take the first non-null medium/source per session
            ARRAY_AGG(channel_medium ORDER BY event_timestamp ASC LIMIT 1)[OFFSET(0)] AS channel_medium,
            ARRAY_AGG(channel_source ORDER BY event_timestamp ASC LIMIT 1)[OFFSET(0)] AS channel_source,
            MAX(is_purchase) AS has_purchase,
            MAX(purchase_value) AS purchase_value,
            MIN(event_timestamp) AS session_start_timestamp
        FROM session_traffic
        WHERE ga_session_id IS NOT NULL
        GROUP BY user_pseudo_id, ga_session_id
    ),

    user_journeys AS (
        -- Build journey per user: ordered sequence of channels
        SELECT
            user_pseudo_id AS user_id,
            STRING_AGG(channel_medium, ' > ' ORDER BY session_start_timestamp ASC) AS journey_medium_path,
            STRING_AGG(channel_source, ' > ' ORDER BY session_start_timestamp ASC) AS journey_source_path,
            MAX(has_purchase) AS has_conversion,
            SUM(purchase_value) AS conversion_value,
            COUNT(*) AS journey_length,
            MIN(event_date) AS first_visit_date,
            MAX(event_date) AS last_visit_date
        FROM session_level
        GROUP BY user_pseudo_id
        HAVING COUNT(*) >= {min_touchpoints}
    )

    SELECT *
    FROM user_journeys
    ORDER BY has_conversion DESC, journey_length DESC
    """

    print(f"Querying BigQuery ({GA4_DATASET})...")
    df = client.query(query).to_dataframe()
    print(f"Retrieved {len(df)} user journeys")

    # Map medium values to standard channel names
    df["journey_path"] = df["journey_medium_path"].apply(map_channels_in_path)
    df["channel_list"] = df["journey_path"].apply(lambda x: x.split(" > "))

    # Clean up conversion values
    df["conversion_value"] = df["conversion_value"].fillna(0).astype(float)
    df["has_conversion"] = df["has_conversion"].astype(int)

    # Summary stats
    n_converting = df["has_conversion"].sum()
    n_total = len(df)
    print(f"\nJourney Summary:")
    print(f"  Total journeys: {n_total}")
    print(f"  Converting journeys: {n_converting} ({n_converting/n_total*100:.1f}%)")
    print(f"  Non-converting journeys: {n_total - n_converting}")
    print(f"  Avg journey length: {df['journey_length'].mean():.1f} touchpoints")
    print(f"  Unique channels: {get_unique_channels(df)}")

    return df


def map_channel(medium):
    """Map a single medium value to a standard channel name."""
    medium_lower = str(medium).lower().strip()
    return CHANNEL_MAPPING.get(medium_lower, DEFAULT_CHANNEL)


def map_channels_in_path(path_str):
    """Map all channels in a journey path string."""
    channels = [map_channel(c.strip()) for c in str(path_str).split(" > ")]
    return " > ".join(channels)


def get_unique_channels(df):
    """Get sorted list of unique channels across all journeys."""
    all_channels = set()
    for channels in df["channel_list"]:
        all_channels.update(channels)
    return sorted(all_channels)


def prepare_mam_dataframe(df):
    """
    Convert journey DataFrame to the format expected by DP6's MAM library.

    MAM with group_channels=True expects:
    - journey_id: unique journey identifier
    - conversion: boolean (True/False)
    - channel: the channel for this session/touchpoint

    One row per session per user.
    """
    rows = []
    for _, journey in df.iterrows():
        channels = journey["channel_list"]
        user_id = journey["user_id"]
        has_conv = bool(journey["has_conversion"])

        for i, channel in enumerate(channels):
            rows.append({
                "journey_id": f"{user_id}_{i}",
                "user_id": user_id,
                "touchpoint_number": i + 1,
                "channel": channel,
                "conversion": has_conv,
                "conversion_value": journey["conversion_value"] if has_conv else 0,
            })

    mam_df = pd.DataFrame(rows)
    print(f"\nMAM DataFrame: {len(mam_df)} rows ({len(df)} journeys expanded to touchpoints)")
    return mam_df


if __name__ == "__main__":
    """Test the data extraction pipeline."""
    print("=" * 60)
    print("Marketing Attribution Agent - Data Extraction Test")
    print("=" * 60)

    # Install python-dotenv if not already installed
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Installing python-dotenv...")
        os.system("pip install python-dotenv")

    # Extract journeys
    df = extract_journeys()

    # Show sample data
    print("\n--- Sample Converting Journeys ---")
    converting = df[df["has_conversion"] == 1].head(5)
    for _, row in converting.iterrows():
        print(f"  {row['journey_path']} -> ${row['conversion_value']:.2f}")

    print("\n--- Sample Non-Converting Journeys ---")
    non_converting = df[df["has_conversion"] == 0].head(5)
    for _, row in non_converting.iterrows():
        print(f"  {row['journey_path']}")

    # Prepare MAM format
    mam_df = prepare_mam_dataframe(df)
    print("\n--- MAM DataFrame Sample ---")
    print(mam_df.head(10).to_string(index=False))

    print("\n✅ Data extraction complete!")
