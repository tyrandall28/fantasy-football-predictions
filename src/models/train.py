"""
Model training pipeline for fantasy football prediction.

Provides classes and functions for training multiple models, comparing performance,
and selecting the best model based on evaluation metrics.
"""

import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ..utils.config import get_model_config, get_data_config

logger = logging.getLogger(__name__)


class FantasyFootballModel:
    """
    Base class for fantasy football prediction models.
    
    Provides common functionality for training, prediction, and evaluation
    that can be extended by specific model implementations.
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        """
        Initialize the model with configuration.
        
        Args:
            model_name: Name of the model (e.g., 'random_forest', 'linear_regression')
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.config = get_model_config(model_name, config)
        self.data_config = get_data_config(config)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self.training_metrics = {}
        
    def _create_model(self):
        """Create the appropriate model based on model_name."""
        if self.model_name == 'random_forest':
            self.model = RandomForestRegressor(**self.config)
        elif self.model_name == 'linear_regression':
            self.model = LinearRegression(**self.config)
        elif self.model_name == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**self.config)
        elif self.model_name == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
            self.model = xgb.XGBRegressor(**self.config)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = None, scale_features: bool = None) -> Dict[str, float]:
        """
        Train the model and return performance metrics.
        
        Args:
            X: Feature matrix
            y: Target variable (fantasy points)
            test_size: Proportion of data for testing
            scale_features: Whether to scale features
            
        Returns:
            Dictionary containing performance metrics
        """
        if test_size is None:
            test_size = self.data_config['test_size']
        
        if scale_features is None:
            scale_features = self.model_name in ['linear_regression']
        
        logger.info(f"Training {self.model_name} model...")
        
        # Store feature columns for prediction consistency
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.data_config['random_state']
        )
        
        # Scale features if needed
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Create and train model
        self._create_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            y_train, y_pred_train, y_test, y_pred_test,
            X_train_scaled, y_train
        )
        
        self.training_metrics = metrics
        logger.info(f"{self.model_name} training completed. "
                   f"Test MAE: {metrics['test_mae']:.2f}, "
                   f"Test R²: {metrics['test_r2']:.3f}")
        
        return metrics
    
    def _calculate_metrics(self, y_train: pd.Series, y_pred_train: np.ndarray,
                          y_test: pd.Series, y_pred_test: np.ndarray,
                          X_train: np.ndarray, y_train_for_cv: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic regression metrics
        metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)
        metrics['test_mae'] = mean_absolute_error(y_test, y_pred_test)
        metrics['train_r2'] = r2_score(y_train, y_pred_train)
        metrics['test_r2'] = r2_score(y_test, y_pred_test)
        
        # Ranking metrics (more important for fantasy football)
        metrics['train_spearman'] = spearmanr(y_train, y_pred_train)[0]
        metrics['test_spearman'] = spearmanr(y_test, y_pred_test)[0]
        metrics['train_kendall'] = kendalltau(y_train, y_pred_train)[0]
        metrics['test_kendall'] = kendalltau(y_test, y_pred_test)[0]
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(
                self.model, X_train, y_train_for_cv, 
                cv=5, scoring='neg_mean_absolute_error'
            )
            metrics['cv_mae'] = -cv_scores.mean()
            metrics['cv_mae_std'] = cv_scores.std()
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            metrics['cv_mae'] = None
            metrics['cv_mae_std'] = None
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure feature consistency
        if self.feature_columns:
            missing_features = set(self.feature_columns) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            X = X[self.feature_columns]
        
        # Scale if necessary
        if hasattr(self.scaler, 'scale_'):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def save_model(self, filepath: str = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model. If None, uses default naming
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            models_dir = Path("models/trained_models")
            models_dir.mkdir(parents=True, exist_ok=True)
            filepath = models_dir / f"ff_predictor_{self.model_name}_{timestamp}.joblib"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_name': self.model_name,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'FantasyFootballModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(model_data['model_name'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_columns = model_data['feature_columns']
        instance.config = model_data['config']
        instance.training_metrics = model_data.get('training_metrics', {})
        instance.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        return instance


class ModelComparison:
    """
    Class for comparing multiple fantasy football models.
    
    Trains multiple models and provides comprehensive comparison metrics
    to help select the best performing model.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model comparison.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.comparison_results = None
        
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        models_to_train: List[str] = None) -> pd.DataFrame:
        """
        Train multiple models and compare their performance.
        
        Args:
            X: Feature matrix
            y: Target variable
            models_to_train: List of model names to train
            
        Returns:
            DataFrame with comparison results
        """
        if models_to_train is None:
            models_to_train = ['linear_regression', 'random_forest', 
                             'gradient_boosting']
            if XGBOOST_AVAILABLE:
                models_to_train.append('xgboost')
        
        logger.info(f"Training and comparing {len(models_to_train)} models")
        
        results = []
        
        for model_name in models_to_train:
            try:
                logger.info(f"Training {model_name}...")
                
                # Create and train model
                model = FantasyFootballModel(model_name, self.config)
                metrics = model.train(X, y)
                
                # Store model and results
                self.models[model_name] = model
                
                result = {
                    'model_name': model_name,
                    **metrics
                }
                results.append(result)
                
                logger.info(f"✓ {model_name} completed")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Create comparison DataFrame
        self.comparison_results = pd.DataFrame(results)
        
        if not self.comparison_results.empty:
            # Sort by primary metric (Spearman correlation is most important for rankings)
            self.comparison_results = self.comparison_results.sort_values(
                'test_spearman', ascending=False
            )
            
            logger.info("Model comparison completed")
            self._log_comparison_summary()
        
        return self.comparison_results
    
    def _log_comparison_summary(self):
        """Log a summary of model comparison results."""
        if self.comparison_results is None or self.comparison_results.empty:
            return
        
        logger.info("=== MODEL COMPARISON RESULTS ===")
        
        for _, row in self.comparison_results.iterrows():
            logger.info(f"{row['model_name']}:")
            logger.info(f"  Spearman Correlation: {row['test_spearman']:.4f}")
            logger.info(f"  MAE: {row['test_mae']:.2f}")
            logger.info(f"  R²: {row['test_r2']:.4f}")
            logger.info("")
        
        best_model = self.comparison_results.iloc[0]
        logger.info(f"Best Model: {best_model['model_name']} "
                   f"(Spearman: {best_model['test_spearman']:.4f})")
    
    def get_best_model(self) -> FantasyFootballModel:
        """
        Get the best performing model based on Spearman correlation.
        
        Returns:
            Best performing model instance
        """
        if self.comparison_results is None or self.comparison_results.empty:
            raise ValueError("No models have been trained yet")
        
        best_model_name = self.comparison_results.iloc[0]['model_name']
        return self.models[best_model_name]
    
    def save_comparison_results(self, filepath: str = None) -> str:
        """
        Save comparison results to CSV.
        
        Args:
            filepath: Path to save results
            
        Returns:
            Path where results were saved
        """
        if self.comparison_results is None:
            raise ValueError("No comparison results to save")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"model_comparison_results_{timestamp}.csv"
        
        self.comparison_results.to_csv(filepath, index=False)
        logger.info(f"Comparison results saved to {filepath}")
        return filepath


def train_best_model(X: pd.DataFrame, y: pd.Series, 
                    config: Dict[str, Any] = None) -> Tuple[FantasyFootballModel, pd.DataFrame]:
    """
    Convenience function to train multiple models and return the best one.
    
    Args:
        X: Feature matrix
        y: Target variable
        config: Configuration dictionary
        
    Returns:
        Tuple of (best_model, comparison_results)
        
    Example:
        >>> best_model, results = train_best_model(X_train, y_train)
        >>> predictions = best_model.predict(X_test)
    """
    comparison = ModelComparison(config)
    results = comparison.train_all_models(X, y)
    best_model = comparison.get_best_model()
    
    return best_model, results 