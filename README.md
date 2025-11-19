# Formula 1 Crash Predictor

(old project i had not had on github before, explaining commit history)

Machine learning pipeline for predicting F1 crash likelihood using telemetry data from the OpenF1 API.

## Overview

End-to-end ML pipeline achieving approximately 0.70 F1-score and 0.77 ROC-AUC for crash prediction.

### Models
- Logistic Regression
- Random Forest
- XGBoost

## Features

### Data Collection
- ETL pipeline using OpenF1 API
- Processes 1000+ lap records across multiple sessions
- Telemetry data including lap times, speed, position, weather

### Feature Engineering
16 engineered features including:
- Lap time deltas (personal best, session average)
- Tire degradation metrics
- Weather impact scores
- Speed variance
- Position changes
- Track difficulty
- Driver risk profiles

### Handling Class Imbalance
- SMOTE oversampling
- Class weights in model training
- Approximately 5% positive class rate

### Model Training
- Grid search hyperparameter tuning
- 5-fold cross-validation
- StandardScaler feature normalization

### Analysis
- EDA with matplotlib and seaborn
- Correlation analysis
- Feature importance evaluation
- ROC curves and confusion matrices

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Collection
```bash
python src/data_collection.py
```

### 2. Run Complete Pipeline
```bash
python main.py
```

### 3. Individual Modules
```bash
# Feature engineering only
python src/feature_engineering.py

# EDA and visualization
python src/eda.py

# Model training
python src/model_training.py
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── main.py                      # Main pipeline
├── src/
│   ├── data_collection.py       # OpenF1 API data collection
│   ├── feature_engineering.py   # Feature creation and transformation
│   ├── eda.py                   # Exploratory Data Analysis
│   ├── model_training.py        # ML model training and evaluation
│   └── utils.py                 # Helper functions
├── data/                        # Data directory (created automatically)
└── visualizations/              # Output plots (created automatically)
```

## Performance

Target metrics:
- F1-Score: 0.70
- ROC-AUC: 0.77
- Precision: 0.65-0.75
- Recall: 0.68-0.76

## Technical Stack

- Python 3.8+
- pandas, NumPy (data processing)
- scikit-learn (ML models)
- XGBoost (gradient boosting)
- matplotlib, seaborn (visualization)
- imbalanced-learn (SMOTE)

## Project Structure

```
.
├── main.py
├── requirements.txt
├── src/
│   ├── data_collection.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── model_training.py
│   └── utils.py
├── data/
├── visualizations/
└── models/
```

January 2025

