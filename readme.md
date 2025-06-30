# Fantasy Football Player Ranking Predictor

## Overview

A machine learning pipeline that predicts fantasy football player rankings using historical performance data, achieving **63.37% Spearman correlation** vs ESPN's 58.15% in real-world 2024 season evaluation.

**Key Finding**: Linear Regression surprisingly outperformed complex ensemble methods in cross-validation testing, highlighting the importance of feature engineering over algorithm complexity.

## Features

- **Modular Architecture**: Clean project structure with separation of concerns
- **Configuration Management**: YAML-based parameter management for easy experimentation
- **Multi-Model Comparison**: Automated comparison of 4+ ML algorithms
- **Advanced Feature Engineering**: 200+ engineered features including lagged statistics, trends, and efficiency metrics
- **Comprehensive Evaluation**: Multiple ranking metrics (Spearman, MAE)
- **Production-Ready**: Type hints, logging, error handling, and comprehensive documentation

## Model Performance

### Cross-Validation Results (Training Phase)

| Model | Spearman Correlation | MAE | R² Score |
|-------|---------------------|-----|----------|
| **Linear Regression** | **0.7536** | 71.2 | 0.78 |
| Random Forest | 0.7234 | 69.8 | 0.82 |
| XGBoost | 0.7156 | 70.1 | 0.81 |
| Gradient Boosting | 0.7089 | 72.3 | 0.79 |

### Real-World Evaluation (2024 Season)
**Model vs Actual Results**: 63.37% Spearman correlation  
**ESPN vs Actual Results**: 58.15% Spearman correlation

*Note: Real-world performance typically differs from cross-validation due to data distribution shifts and unforeseen events.*

## Technical Architecture

```
fantasy-football-predictor/
├── config/                 # Configuration management
│   ├── config.yaml        # Model parameters, data paths, features
│   └── __init__.py
├── src/                   # Source code modules
│   ├── data/              # Data loading and preprocessing
│   │   ├── data_loader.py # Professional data loading with validation
│   │   └── __init__.py
│   ├── features/          # Feature engineering
│   │   ├── feature_engineering.py # 200+ engineered features
│   │   └── __init__.py
│   ├── models/            # Model training and prediction
│   │   ├── train.py       # Multi-model comparison pipeline
│   │   └── __init__.py
│   ├── utils/             # Utilities and helpers
│   │   ├── config.py      # Configuration management
│   │   ├── demo.py        # Pipeline demonstration
│   │   └── __init__.py
│   └── __init__.py
├── data/                  # Data storage
│   ├── raw/              # Original CSV files (2004-2024)
│   ├── processed/        # Cleaned and processed data
│   └── external/         # ESPN rankings for comparison
├── models/               # Trained models
│   └── trained_models/   # Serialized model files
├── notebooks/            # Jupyter notebooks for exploration
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tyrandall28/fantasy-football-predictions.git
cd fantasy-football-predictor

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
from src.data.data_loader import load_and_clean_data
from src.features.feature_engineering import FantasyFeatureEngineer
from src.models.train import train_best_model

# Load and prepare data
data = load_and_clean_data([2020, 2021, 2022, 2023])

# Engineer features
feature_engineer = FantasyFeatureEngineer()
features = feature_engineer.create_training_features(data)

# Train and compare models
best_model, results = train_best_model(X, y)

# Make predictions
predictions = best_model.predict(X_new)
```

### Configuration-Driven Approach

```yaml
# config/config.yaml
model:
  random_forest:
    n_estimators: 500
    max_depth: 10
    random_state: 42

data:
  seasons_for_training: [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
  test_size: 0.2
  n_lag_seasons: 3

features:
  stats_to_lag: ["age", "games", "ppr_fantasy_points", "rush_yards", ...]
```

## Advanced Features

### 1. Sophisticated Feature Engineering
- **Lagged Statistics**: Multi-season historical performance (S-1, S-2, S-3)
- **Derived Metrics**: Per-game averages, efficiency rates, catch rates
- **Trend Analysis**: Year-over-year changes and momentum indicators
- **Aggregate Features**: Multi-season averages, volatility measures
- **Position-Specific Features**: Tailored features for QB, RB, WR, TE

### 2. Multi-Model Architecture
```python
# Automated model comparison
comparison = ModelComparison()
results = comparison.train_all_models(X, y, 
    models=['linear_regression', 'random_forest', 'xgboost', 'gradient_boosting'])
best_model = comparison.get_best_model()
```

### 3. Professional Code Practices
- **Type Hints**: Full type annotation for better code clarity
- **Logging**: Comprehensive logging throughout the pipeline
- **Error Handling**: Graceful error handling and validation
- **Documentation**: Extensive docstrings and examples
- **Modular Design**: Clean separation of concerns

## Key Insights

### The Linear Regression Surprise
My analysis revealed that **Linear Regression significantly outperformed complex ensemble methods**:

- **75.36% Spearman correlation** vs 71.56% for XGBoost
- Demonstrates that **feature engineering > algorithm complexity**
- Validates the importance of domain knowledge in feature creation

### Feature Importance Findings
1. **Multi-season aggregates** (29.03% avg importance)
2. **Lagged PPR points** (27.04% avg importance) 
3. **Per-game averages** and **efficiency metrics**
4. **Trend indicators** for momentum detection

### Business vs Technical Optimization
The model correctly identifies QBs as highest-scoring players but reveals the distinction between:
- **Technical accuracy**: Predicting total points
- **Draft strategy**: Considering positional scarcity and opportunity cost

## Technical Implementation

### Machine Learning Pipeline
- Multi-model comparison framework with automated evaluation
- Advanced feature engineering including lagged statistics, derived metrics, and trend analysis
- Time series cross-validation with chronological train/test splits
- Multiple ranking metrics for comprehensive performance assessment

### Software Architecture
- Modular project structure with clean separation of concerns
- Configuration-driven approach with YAML parameter management
- Comprehensive logging and error handling throughout pipeline
- Full type annotation and documentation standards

## Project Evolution

This project currently represents a **V2.0 professional refactor** of an initial working model:

- **V1.0**: Achieved competitive performance vs ESPN rankings
- **V2.0**: Complete restructure for production-ready, maintainable code
- **Future**: Position-specific models, advanced features, API deployment

## Future Work & Experiments

### Priority Experiments
1. **Time Window Optimization**: Investigate optimal training window lengths (e.g., 3-year vs 10-year historical data). Initial analysis suggests longer windows may significantly outperform recent data due to statistical power vs recency trade-offs. Big picture: do player point trends change enough over time as the game evolves to make the historical data no longer as useful to include in the model training? (Think how teams would heavily lean on a single workhorse RB in the late 2000s vs how teams now routinely carry 2-3 RBs who share the workload. This may have an effect on the model accuracy for predictions today)

2. **Position-Specific Models**: Train separate models for QB, RB, WR, TE to capture position-specific performance patterns.

3. **Advanced Feature Engineering**: Incorporate team context, coaching changes, and market share metrics.

## Methodology

### Data Strategy
- **Training Data**: Historical seasons (2013-2023) based on actual performance
- **Prediction Pool**: ESPN's 2024 preseason rankings (avoiding expert bias in training)
- **Evaluation**: Compare 2024 predictions against actual season outcomes

### Model Selection Criteria
1. **Spearman Correlation**: Primary metric for ranking accuracy
2. **Mean Absolute Error**: Point prediction accuracy
3. **R² Score**: Variance explanation capability
4. **Cross-Validation**: Generalization assessment

---

*Built with Python, scikit-learn, XGBoost, and professional ML engineering practices.*