# P6: Marketing Attribution Agent
## Run
- Python 3.13, `venv\Scripts\activate`
- Models: `python src/attribution/models.py`
- Agent: `python -m src.agent.graph`
- LSTM: `python src/deep_learning/train.py`
- Dashboard: `streamlit run app.py`
## Key
- src/attribution/ — data_prep.py (BigQuery), models.py (7 models, MAM tuple index [1])
- src/deep_learning/ — LSTM (AUC 0.72, models/lstm_attribution.keras)
- src/agent/ — LangGraph agent, gemini-2.5-flash-lite, pre-computed data
## Gotchas
- MAM: tuple[1] not [0], seaborn-white patched to seaborn-v0_8-white
- Windows: sys.stdout.reconfigure(encoding='utf-8')
- GCP: galvanic-smoke-489914-u7, bigquery-public-data.ga4_obfuscated_sample_ecommerce
