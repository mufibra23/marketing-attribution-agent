"""Configuration for Marketing Attribution Agent."""
import os
from dotenv import load_dotenv

load_dotenv()

# BigQuery
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "galvanic-smoke-489914-u7")
GA4_DATASET = "bigquery-public-data.ga4_obfuscated_sample_ecommerce"

# Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Attribution model settings
CHANNEL_MAPPING = {
    "organic": "organic_search",
    "cpc": "paid_search",
    "social": "social",
    "email": "email",
    "(direct)": "direct",
    "(none)": "direct",
    "referral": "referral",
    "display": "display",
    "affiliate": "affiliate",
}

# Default to "other" for any unmapped channels
DEFAULT_CHANNEL = "other"

# Minimum journey length for attribution (single-touchpoint = no attribution needed)
MIN_TOUCHPOINTS = 2

# Maximum journey length for LSTM padding
MAX_SEQ_LENGTH = 8
