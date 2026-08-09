# AI Cattle Health Monitoring

This repository contains generator scripts for a synthetic cattle health monitoring dataset and associated analysis pipelines.

## Structure

- `config.py`: global configuration values
- `baseline_generator.py`: generates a healthy baseline dataset
- `cow_profile_generator.py`: creates synthetic cow profiles
- `physiology_generator.py`: generates physiology samples with realistic variation
- `preprocessing.py`, `train_xgboost.py`, `risk_score_engine.py`, `xai_engine.py`: downstream modeling and explainability components

## Getting started

1. Create a Python environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the generators as needed.

## Notes

Generated datasets are intentionally ignored by Git so they do not get pushed to the repository.
