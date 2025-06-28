# Professional ML Project Structure Guide

## Project Directory Structure
```
fantasy-football-predictor/
├── README.md
├── requirements.txt
├── setup.py (optional but impressive)
├── .gitignore
├── config/
│   ├── __init__.py
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── tests/
│   ├── __init__.py
│   └── test_data_processing.py
├── models/
│   └── trained_models/
└── docs/
    └── model_documentation.md
```

## Key Principles
- **Notebooks vs Python Files**: Use notebooks for exploration/experimentation, convert stable code to .py files
- **Separate concerns**: data loading, preprocessing, feature engineering, training, evaluation
- **Make it modular**: each function should do one thing well
- **Configuration-driven**: use config files instead of hardcoded values

## Code Structure Examples

### data/data_loader.py
```python
"""
Data loading and initial validation utilities.
"""
import pandas as pd
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def load_player_data(file_path: str, season: Optional[int] = None) -> pd.DataFrame:
    """
    Load fantasy football player data from CSV.
    
    Args:
        file_path: Path to the data file
        season: Optional season filter
        
    Returns:
        DataFrame with player statistics
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        
        # Validate required columns
        required_cols = ['player_name', 'position', 'fantasy_points']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if season:
            df = df[df['season'] == season]
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
```

### models/train.py
```python
"""
Model training pipeline.
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class FantasyFootballPredictor:
    """
    Random Forest model for predicting fantasy football player rankings.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """Initialize the predictor with model parameters."""
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.feature_columns = None
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        """
        Train the model and return performance metrics.
        
        Args:
            X: Feature matrix
            y: Target variable (fantasy points)
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info("Starting model training...")
        
        # Store feature columns for prediction consistency
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_score': cross_val_score(self.model, X_train, y_train, cv=5).mean()
        }
        
        logger.info(f"Training completed. MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
        return metrics
        
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
```

## Commenting Best Practices

### 1. Docstrings for all functions/classes
```python
def calculate_rolling_average(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Calculate rolling average of fantasy points over specified window.
    
    This helps smooth out week-to-week variance and identify trends
    in player performance.
    
    Args:
        df: DataFrame with player statistics
        window: Number of games for rolling window (default: 3)
        
    Returns:
        DataFrame with additional rolling average columns
        
    Example:
        >>> df_with_rolling = calculate_rolling_average(player_df, window=4)
        >>> print(df_with_rolling.columns)
        ['player_name', 'fantasy_points', 'fantasy_points_rolling_avg']
    """
```

### 2. Inline comments for complex logic
```python
# Handle missing values using forward fill for time series continuity
# This assumes missing weeks mean the player didn't play (injury/bye)
df['fantasy_points'] = df.groupby('player_name')['fantasy_points'].fillna(method='ffill')

# Calculate opponent strength adjustment
# Stronger defenses typically allow fewer fantasy points
df['opponent_adj_points'] = df['fantasy_points'] * df['opponent_def_ranking']
```

### 3. Avoid obvious comments
```python
# BAD
x = x + 1  # increment x by 1

# GOOD  
# Adjust for 1-based ranking system used in fantasy platforms
ranking = ranking + 1
```

## Professional Touches

### 1. Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 2. Type hints
```python
from typing import List, Dict, Optional, Tuple
import pandas as pd

def process_features(df: pd.DataFrame, 
                    feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
```

### 3. Configuration files
```yaml
# config/config.yaml
model:
  n_estimators: 100
  max_depth: 10
  random_state: 42

data:
  train_file: "data/processed/train_data.csv"
  test_size: 0.2
  
features:
  numeric: ["passing_yards", "rushing_yards", "receptions"]
  categorical: ["position", "team"]
```

### 4. Tests
```python
# tests/test_data_processing.py
import pytest
from src.data.preprocessing import clean_player_data

def test_clean_player_data():
    # Test data cleaning function
    pass
```

## README Template
```markdown
# Fantasy Football Player Ranking Predictor

## Overview
Machine learning pipeline that predicts fantasy football player rankings using historical performance data, opponent matchups, and advanced statistics.

## Features
- Data preprocessing and feature engineering
- Multiple model comparison (Random Forest, XGBoost, Linear Regression)
- Cross-validation and hyperparameter tuning
- Performance visualization and analysis

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.models.train import FantasyFootballPredictor

# Load and train model
predictor = FantasyFootballPredictor()
metrics = predictor.train(X_train, y_train)
predictor.save_model('models/ff_predictor.joblib')
```

## Model Performance
- Mean Absolute Error: 2.3 fantasy points
- R² Score: 0.78
- Cross-validation Score: 0.75

## Data Sources
- [Fantasy football statistics source]
- [Injury reports source]
```

## Key Takeaways
This structure demonstrates:
- Understanding of software engineering principles
- ML best practices and maintainable code
- Thinking beyond "getting the model to work" to building sustainable solutions
- Professional development practices (testing, documentation, configuration management)