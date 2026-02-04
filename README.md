# Animal Health Disease Prediction Model

This project provides a veterinary disease prediction web app powered by an XGBoost classifier. It includes
data generation utilities, a training pipeline, and a Flask-based API with a futuristic UI dashboard.

## What's Included

- **Model training** via `train_model.py` (exports `animal_model.pkl`, `label_encoder.pkl`, `model_features.pkl`)
- **Metrics persistence** to `training_metrics.json` for transparency and monitoring
- **Data ingestion** helper `data_ingest.py` to normalize and merge external datasets
- **Interactive UI** in `templates/index.html` with top predictions and error states

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (using the bundled dataset):

   ```bash
   python train_model.py --dataset enhanced_animal_disease.csv
   ```

3. Start the API server:

   ```bash
   python app.py
   ```

4. Open the UI in your browser:

   ```
   http://localhost:10000
   ```

## Data Ingestion (Optional)

To ingest external datasets into a unified CSV:

1. Copy the example config:

   ```bash
   cp data_sources.example.json data_sources.json
   ```

2. Update URLs and column mappings in `data_sources.json`.

3. Run ingestion:

   ```bash
   python data_ingest.py
   ```

This will produce `data/combined_dataset.csv`, which can be passed into `train_model.py`.

## API Endpoints

- `GET /status` - service status and latest training metrics (if available)
- `POST /predict` - returns prediction, confidence, reasoning, and top 3 predictions

## Notes

- Accuracy and metrics depend on the dataset used. See `training_metrics.json` after training.
- The UI expects the backend to be running locally on port `10000`.
