"""
Demonstration script for the professional fantasy football prediction pipeline.

Shows how to use the modular components to load data, engineer features,
train models, and make predictions.
"""

import logging
import pandas as pd
from pathlib import Path

from ..data.data_loader import load_and_clean_data
from ..features.feature_engineering import FantasyFeatureEngineer
from ..models.train import train_best_model
from ..utils.config import load_config, validate_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline_demo():
    """
    Run a complete demonstration of the fantasy football prediction pipeline.
    
    This function demonstrates:
    1. Loading and validating configuration
    2. Loading and cleaning data
    3. Engineering features
    4. Training and comparing models
    5. Making predictions
    """
    
    try:
        logger.info("Starting Fantasy Football Prediction Pipeline Demo")
        
        # Step 1: Load and validate configuration
        logger.info("Loading configuration...")
        config = load_config()
        validate_config(config)
        logger.info("Configuration loaded and validated")
        
        # Step 2: Load and clean data
        logger.info("Loading fantasy football data...")
        seasons_to_load = [2020, 2021, 2022, 2023]  # Smaller subset for demo
        clean_data = load_and_clean_data(seasons_to_load, config)
        logger.info(f"Loaded data: {clean_data.shape[0]} records, {clean_data.shape[1]} columns")
        
        # Step 3: Engineer features
        logger.info("Engineering features...")
        feature_engineer = FantasyFeatureEngineer(config)
        feature_data = feature_engineer.create_training_features(clean_data)
        logger.info(f"Features created: {feature_data.shape[0]} samples, {feature_data.shape[1]} features")
        
        # Step 4: Prepare training data
        logger.info("Preparing training data...")
        
        # Remove non-predictive columns
        feature_columns_to_drop = ['player_id', 'player_name', 'prediction_season', 'target_ppr_points']
        X = feature_data.drop(columns=feature_columns_to_drop)
        y = feature_data['target_ppr_points']
        
        # Handle categorical variables (position)
        X = pd.get_dummies(X, columns=['position'], prefix='pos', drop_first=True)
        
        logger.info(f"Training data prepared: {X.shape[1]} features")
        
        # Step 5: Train and compare models
        logger.info("Training and comparing models...")
        best_model, comparison_results = train_best_model(X, y, config)
        
        logger.info("Model training completed!")
        logger.info("Model Comparison Results:")
        print(comparison_results[['model_name', 'test_spearman', 'test_mae', 'test_r2']].round(4))
        
        # Step 6: Save the best model
        logger.info("Saving best model...")
        model_path = best_model.save_model()
        logger.info(f"Best model saved: {model_path}")
        
        # Step 7: Demonstrate prediction capability
        logger.info("Demonstrating prediction capability...")
        
        # Use a subset of the data for prediction demo
        sample_data = X.head(10)
        predictions = best_model.predict(sample_data)
        
        # Show some example predictions
        prediction_df = pd.DataFrame({
            'predicted_ppr_points': predictions,
            'actual_ppr_points': y.head(10).values
        })
        
        logger.info("Sample Predictions:")
        print(prediction_df.round(2))
        
        logger.info("Pipeline demonstration completed successfully!")
        
        return {
            'best_model': best_model,
            'comparison_results': comparison_results,
            'feature_data': feature_data,
            'clean_data': clean_data
        }
        
    except Exception as e:
        logger.error(f"Pipeline demonstration failed: {e}")
        raise


def quick_data_summary():
    """
    Generate a quick summary of the available data.
    
    Useful for understanding the data before running the full pipeline.
    """
    try:
        logger.info("Generating quick data summary...")
        
        # Load configuration
        config = load_config()
        
        # Load data
        clean_data = load_and_clean_data(config=config)
        
        # Basic statistics
        summary = {
            'total_records': len(clean_data),
            'seasons': sorted(clean_data['Season'].unique()),
            'positions': clean_data['position'].value_counts().to_dict(),
            'columns': list(clean_data.columns),
            'sample_records': clean_data.head()
        }
        
        logger.info(f"Data Summary Generated")
        logger.info(f"Total Records: {summary['total_records']}")
        logger.info(f"Seasons: {summary['seasons']}")
        logger.info(f"Positions: {summary['positions']}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Data summary generation failed: {e}")
        raise


def test_configuration():
    """
    Test the configuration system.
    
    Verifies that configuration files can be loaded and are valid.
    """
    try:
        logger.info("Testing configuration system...")
        
        # Load config
        config = load_config()
        
        # Validate config
        validate_config(config)
        
        # Test specific config functions
        from ..utils.config import get_model_config, get_data_config, get_feature_config
        
        model_config = get_model_config('random_forest', config)
        data_config = get_data_config(config)
        feature_config = get_feature_config(config)
        
        logger.info("Configuration system test passed!")
        logger.info(f"Available models: {list(config['model'].keys())}")
        logger.info(f"Training seasons: {data_config['seasons_for_training']}")
        logger.info(f"Features to lag: {len(feature_config['stats_to_lag'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False


if __name__ == "__main__":
    # You can run individual components by uncommenting the desired function
    
    # Test configuration
    test_configuration()
    
    # Generate data summary
    quick_data_summary()
    
    # Run full pipeline demo (commented out as it takes longer)
    run_full_pipeline_demo() 