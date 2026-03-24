# Marketing Attribution Agent

An AI-powered marketing attribution analysis platform that runs **7 statistical models + 1 LSTM deep learning model** on Google Analytics 4 data, with a conversational AI agent for interactive insights.

Built for the **Hack2Skill** hackathon.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit Dashboard                         │
│  ┌──────────┬────────────┬──────────┬──────────┬───────────────┐   │
│  │ Overview  │ Attribution│ Channel  │  LSTM    │  AI Agent     │   │
│  │          │  Models    │ Deep Dive│  D.L.    │  Chat         │   │
│  └──────────┴────────────┴──────────┴──────────┴───────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│  BigQuery     │  │  Attribution    │  │  LangGraph Agent    │
│  GA4 Data     │  │  Engine         │  │  (Gemini 2.5 Flash) │
│  Extraction   │  │                 │  │                     │
│               │  │  7 Statistical  │  │  ┌───────────────┐  │
│  User Journey │  │  Models (MAM)   │  │  │ Tool: Compare │  │
│  Builder      │  │  + Shapley      │  │  │ Tool: Budget  │  │
│               │  │                 │  │  │ Tool: LSTM    │  │
└───────┬───────┘  └────────┬────────┘  │  └───────────────┘  │
        │                   │           └─────────────────────┘
        │                   │
        ▼                   ▼
┌───────────────────────────────────────┐
│          LSTM Deep Learning           │
│  Masking → LSTM(64) → LSTM(32)       │
│  → Dense(sigmoid)                    │
│  tf.GradientTape Attribution         │
└───────────────────────────────────────┘
```

## Features

- **7 Statistical Attribution Models** — First-Click, Last-Click, Linear, Time-Decay, Position-Based, Markov Chain, Shapley Value
- **LSTM Deep Learning Attribution** — Gradient-based channel attribution using TensorFlow/Keras
- **Conversational AI Agent** — LangGraph + Gemini 2.5 Flash for natural language analysis of results
- **Interactive Dashboard** — 5-tab Streamlit app with Plotly visualizations
- **BigQuery Integration** — Pulls real GA4 sample ecommerce data from Google's public dataset
- **Budget Recommendations** — Data-driven budget reallocation based on Markov vs Last-Click deltas
- **Model Disagreement Analysis** — Identifies channels where models disagree (high uncertainty)

## Tech Stack

| Layer | Technology |
|---|---|
| **Data** | Google BigQuery, GA4 Obfuscated Sample Ecommerce |
| **Attribution** | marketing-attribution-models (MAM) by DP6 |
| **Deep Learning** | TensorFlow 2.21, Keras 3, LSTM + GradientTape |
| **AI Agent** | LangGraph, LangChain, Gemini 2.5 Flash Lite |
| **Dashboard** | Streamlit, Plotly |
| **Language** | Python 3.13 |

## Setup

### Prerequisites

- Python 3.13+
- Google Cloud project with BigQuery access
- Google API key (for Gemini agent)

### Installation

```bash
git clone https://github.com/mufibra23/marketing-attribution-agent.git
cd marketing-attribution-agent
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_API_KEY=your-gemini-api-key
```

> The BigQuery dataset (`bigquery-public-data.ga4_obfuscated_sample_ecommerce`) is publicly accessible — you only need a GCP project for billing.

### Run

**Dashboard (recommended):**
```bash
streamlit run app.py
```

**Attribution models only:**
```bash
python src/attribution/models.py
```

**Train LSTM model:**
```bash
python src/deep_learning/train.py
```

**CLI agent chat:**
```bash
python -m src.agent.graph
```

## Attribution Models

| Model | Type | Description |
|---|---|---|
| First-Click | Rule-based | 100% credit to first touchpoint |
| Last-Click | Rule-based | 100% credit to last touchpoint (GA4 default) |
| Linear | Rule-based | Equal credit to all touchpoints |
| Time-Decay | Rule-based | More credit to touchpoints closer to conversion |
| Position-Based | Rule-based | 40% first, 40% last, 20% middle |
| Markov Chain | Data-driven | Transition probability-based removal effect |
| Shapley Value | Game theory | Marginal contribution of each channel |
| LSTM | Deep learning | Gradient-based attribution from conversion prediction model |

## Project Structure

```
marketing-attribution-agent/
├── app.py                          # Streamlit dashboard (5 tabs)
├── src/
│   ├── config.py                   # GCP, channel mapping, settings
│   ├── attribution/
│   │   ├── data_prep.py            # BigQuery GA4 journey extraction
│   │   └── models.py               # 7 attribution models (MAM)
│   ├── deep_learning/
│   │   ├── lstm_model.py           # LSTM architecture & training
│   │   ├── sequence_prep.py        # Journey → padded sequences
│   │   ├── attribution.py          # GradientTape attribution
│   │   └── train.py                # Training script
│   └── agent/
│       ├── graph.py                # LangGraph agent with ReAct loop
│       ├── state.py                # Agent state schema
│       └── tools.py                # LangChain tools for agent
├── models/                         # Saved LSTM model (.keras)
├── requirements.txt
└── .streamlit/config.toml
```

## Data Source

Uses Google's public **GA4 Obfuscated Sample Ecommerce** dataset from BigQuery (`bigquery-public-data.ga4_obfuscated_sample_ecommerce`). This contains real (anonymized) Google Analytics 4 event data from the Google Merchandise Store.

## License

MIT
