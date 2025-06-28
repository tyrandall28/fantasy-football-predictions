import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Import XGBoost
from xgboost import XGBRegressor

# Import the existing data loading and feature engineering functions
import sys
sys.path.append('.')
from train import load_and_clean_data, engineer_features

# --- Model Configuration ---
MODELS_TO_COMPARE = {
    'Linear Regression': {
        'model': LinearRegression(),
        'description': 'Simple linear baseline model'
    },
    'Random Forest': {
        'model': RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'description': 'Ensemble of decision trees with bagging'
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'description': 'Current model - sequential ensemble with boosting'
    },
    'XGBoost': {
        'model': XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'description': 'Optimized gradient boosting implementation'
    }
}

def prepare_model_data(training_df):
    """Prepare features and target for model training."""
    # Same preprocessing as existing train.py
    FEATURES_TO_DROP = ['player_id', 'player_name', 'prediction_season', 'target_ppr_points']
    X = training_df.drop(columns=FEATURES_TO_DROP)
    y = training_df['target_ppr_points']

    # One-hot encode position
    X = pd.get_dummies(X, columns=['position'], prefix='pos', drop_first=True)

    # Chronological split (same as existing approach)
    X_train = X[training_df['prediction_season'] < 2023]
    y_train = y[training_df['prediction_season'] < 2023]
    X_test = X[training_df['prediction_season'] == 2023]
    y_test = y[training_df['prediction_season'] == 2023]
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, predictions):
    """Calculate comprehensive evaluation metrics."""
    # Basic regression metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Ranking metrics
    # Create rankings from predictions and actual values
    predicted_ranks = (-predictions).argsort().argsort() + 1  # Convert to ranks (1-based)
    actual_ranks = (-y_test).argsort().argsort() + 1
    
    # Spearman correlation for ranking accuracy
    spearman_corr, _ = spearmanr(predicted_ranks, actual_ranks)
    
    # Mean absolute rank error
    rank_mae = mean_absolute_error(actual_ranks, predicted_ranks)
    
    return {
        'MAE (Points)': mae,
        'R² Score': r2,
        'Spearman Correlation': spearman_corr,
        'Rank MAE': rank_mae
    }

def train_and_evaluate_model(name, model_config, X_train, X_test, y_train, y_test):
    """Train a single model and return evaluation results."""
    print(f"\nTraining {name}...")
    print(f"Description: {model_config['description']}")
    
    # Time the training
    start_time = time.time()
    
    # Train the model
    model = model_config['model']
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, predictions)
    metrics['Training Time (s)'] = training_time
    
    print(f"  MAE: {metrics['MAE (Points)']:.2f} points")
    print(f"  R²: {metrics['R² Score']:.3f}")
    print(f"  Spearman: {metrics['Spearman Correlation']:.3f}")
    print(f"  Training Time: {training_time:.1f}s")
    
    return model, metrics, predictions

def create_comparison_table(results):
    """Create a formatted comparison table of all models."""
    df = pd.DataFrame(results).T
    
    # Round for better display
    df['MAE (Points)'] = df['MAE (Points)'].round(2)
    df['R² Score'] = df['R² Score'].round(3)
    df['Spearman Correlation'] = df['Spearman Correlation'].round(4)
    df['Rank MAE'] = df['Rank MAE'].round(1)
    df['Training Time (s)'] = df['Training Time (s)'].round(1)
    
    # Sort by Spearman correlation (most important metric)
    df = df.sort_values('Spearman Correlation', ascending=False)
    
    return df

def save_results(comparison_df, models, predictions_dict, X_test, y_test):
    """Save all results for further analysis."""
    # Save comparison table
    comparison_df.to_csv('model_comparison_results.csv')
    
    # Save the best model
    best_model_name = comparison_df.index[0]
    best_model = models[best_model_name]
    joblib.dump(best_model, f'best_model_{best_model_name.lower().replace(" ", "_")}.joblib')
    
    # Save detailed predictions for analysis
    detailed_results = pd.DataFrame({
        'actual_ppr': y_test.values,
        'actual_rank': (-y_test).argsort().argsort() + 1
    })
    
    for model_name, predictions in predictions_dict.items():
        detailed_results[f'{model_name.lower().replace(" ", "_")}_pred'] = predictions
        detailed_results[f'{model_name.lower().replace(" ", "_")}_rank'] = (-predictions).argsort().argsort() + 1
    
    detailed_results.to_csv('detailed_predictions_comparison.csv', index=False)
    
    print(f"\nResults saved:")
    print(f"  - Model comparison: model_comparison_results.csv")
    print(f"  - Best model: best_model_{best_model_name.lower().replace(' ', '_')}.joblib")
    print(f"  - Detailed predictions: detailed_predictions_comparison.csv")

def main():
    """Main function to run model comparison."""
    print("=== Fantasy Football Model Comparison ===\n")
    print(f"Comparing {len(MODELS_TO_COMPARE)} models:")
    for name, config in MODELS_TO_COMPARE.items():
        print(f"  - {name}: {config['description']}")
    
    # Load and prepare data using existing pipeline
    print("\nLoading and preparing data...")
    master_df = load_and_clean_data()
    training_df = engineer_features(master_df)
    X_train, X_test, y_train, y_test = prepare_model_data(training_df)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Train and evaluate all models
    results = {}
    models = {}
    predictions_dict = {}
    
    for name, config in MODELS_TO_COMPARE.items():
        try:
            model, metrics, predictions = train_and_evaluate_model(
                name, config, X_train, X_test, y_train, y_test
            )
            results[name] = metrics
            models[name] = model
            predictions_dict[name] = predictions
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    # Create and display comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    comparison_df = create_comparison_table(results)
    print(comparison_df.to_string())
    
    # Highlight the winner
    best_model = comparison_df.index[0]
    print(f"\n  BEST MODEL: {best_model}")
    print(f"   Spearman Correlation: {comparison_df.loc[best_model, 'Spearman Correlation']}")
    print(f"   MAE: {comparison_df.loc[best_model, 'MAE (Points)']} points")
    
    # Save all results
    save_results(comparison_df, models, predictions_dict, X_test, y_test)
    
    print(f"\n Model comparison complete!")

if __name__ == '__main__':
    main() 