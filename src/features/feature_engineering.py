"""
Feature engineering utilities for fantasy football prediction.

Creates lagged features, rolling statistics, trends, and derived metrics
from historical player performance data.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..utils.config import get_feature_config, get_data_config

logger = logging.getLogger(__name__)


class FantasyFeatureEngineer:
    """
    Feature engineering class for fantasy football prediction model.
    
    Creates sophisticated features from historical player data including:
    - Lagged statistics from previous seasons
    - Rolling averages and trends
    - Per-game averages and efficiency metrics
    - Volatility and consistency measures
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Configuration dictionary. If None, loads default config.
        """
        self.feature_config = get_feature_config(config)
        self.data_config = get_data_config(config)
        self.stats_to_lag = self.feature_config['stats_to_lag']
        self.n_lag_seasons = self.data_config['n_lag_seasons']
        self.target_variable = 'ppr_fantasy_points'
    
    def create_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for model training.
        
        Args:
            df: Clean player data with multiple seasons
            
        Returns:
            DataFrame with engineered features for training
        """
        logger.info("Starting comprehensive feature engineering")
        
        # Sort data by player and season for proper lagging
        df_sorted = df.sort_values(['player_id', 'Season']).copy()
        
        all_features = []
        grouped = df_sorted.groupby('player_id')
        
        total_players = len(grouped)
        processed_players = 0
        
        for player_id, player_df in grouped:
            player_features = self._create_player_features(player_df)
            all_features.extend(player_features)
            
            processed_players += 1
            if processed_players % 100 == 0:
                logger.info(f"Processed features for {processed_players}/{total_players} players")
        
        if not all_features:
            raise ValueError("No features were generated")
        
        # Create final training DataFrame
        training_df = pd.DataFrame(all_features).fillna(0)
        logger.info(f"Feature engineering completed. Shape: {training_df.shape}")
        
        return training_df
    
    def _create_player_features(self, player_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create features for a single player across all their seasons.
        
        Args:
            player_df: DataFrame with single player's data across seasons
            
        Returns:
            List of feature dictionaries, one per season
        """
        player_features = []
        
        # Need at least 2 seasons to create features (current + 1 lag)
        for i in range(1, len(player_df)):
            target_season_data = player_df.iloc[i]
            
            features = {
                'player_id': target_season_data['player_id'],
                'player_name': target_season_data['player'],
                'position': target_season_data['position'],
                'prediction_season': int(target_season_data['Season']),
                'target_ppr_points': target_season_data[self.target_variable]
            }
            
            # Create lagged features
            features.update(self._create_lagged_features(player_df, i))
            
            # Create derived features
            features.update(self._create_derived_features(features))
            
            player_features.append(features)
        
        return player_features
    
    def _create_lagged_features(self, player_df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """
        Create lagged features from previous seasons.
        
        Args:
            player_df: Player's historical data
            current_idx: Index of current season
            
        Returns:
            Dictionary of lagged features
        """
        lagged_features = {}
        
        for lag in range(1, self.n_lag_seasons + 1):
            if current_idx - lag >= 0:
                past_season = player_df.iloc[current_idx - lag]
                for stat in self.stats_to_lag:
                    feature_name = f"{stat}_s_minus_{lag}"
                    lagged_features[feature_name] = past_season[stat]
            else:
                # Player doesn't have enough history - fill with 0
                for stat in self.stats_to_lag:
                    feature_name = f"{stat}_s_minus_{lag}"
                    lagged_features[feature_name] = 0
        
        return lagged_features
    
    def _create_derived_features(self, base_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create derived features from base lagged features.
        
        Args:
            base_features: Dictionary with base lagged features
            
        Returns:
            Dictionary of derived features
        """
        derived_features = {}
        
        # Per-game averages
        derived_features.update(self._create_per_game_features(base_features))
        
        # Multi-season aggregates
        derived_features.update(self._create_aggregate_features(base_features))
        
        # Trend features
        derived_features.update(self._create_trend_features(base_features))
        
        # Efficiency metrics
        derived_features.update(self._create_efficiency_features(base_features))
        
        return derived_features
    
    def _create_per_game_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Create per-game average features."""
        per_game_features = {}
        
        for lag in range(1, self.n_lag_seasons + 1):
            games_played = features.get(f'games_s_minus_{lag}', 0)
            
            if games_played > 0:
                # PPR points per game
                ppr_total = features.get(f'ppr_fantasy_points_s_minus_{lag}', 0)
                per_game_features[f'ppr_per_game_s_minus_{lag}'] = ppr_total / games_played
                
                # Yards per game
                rush_yards = features.get(f'rush_yards_s_minus_{lag}', 0)
                per_game_features[f'rush_yards_per_game_s_minus_{lag}'] = rush_yards / games_played
                
                rec_yards = features.get(f'rec_yds_s_minus_{lag}', 0)
                per_game_features[f'rec_yards_per_game_s_minus_{lag}'] = rec_yards / games_played
                
                pass_yards = features.get(f'pass_yards_s_minus_{lag}', 0)
                per_game_features[f'pass_yards_per_game_s_minus_{lag}'] = pass_yards / games_played
            else:
                per_game_features[f'ppr_per_game_s_minus_{lag}'] = 0
                per_game_features[f'rush_yards_per_game_s_minus_{lag}'] = 0
                per_game_features[f'rec_yards_per_game_s_minus_{lag}'] = 0
                per_game_features[f'pass_yards_per_game_s_minus_{lag}'] = 0
        
        return per_game_features
    
    def _create_aggregate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Create multi-season aggregate features."""
        aggregate_features = {}
        
        # Last 2 seasons average
        last_2_seasons_points = [
            features.get(f'ppr_fantasy_points_s_minus_{j}', 0) 
            for j in range(1, min(3, self.n_lag_seasons + 1))
        ]
        valid_points_l2 = [p for p in last_2_seasons_points if p > 0]
        
        if valid_points_l2:
            aggregate_features['avg_ppr_last_2_seasons'] = np.mean(valid_points_l2)
        else:
            aggregate_features['avg_ppr_last_2_seasons'] = 0
        
        # Last 3 seasons average and volatility
        last_3_seasons_points = [
            features.get(f'ppr_fantasy_points_s_minus_{j}', 0) 
            for j in range(1, min(4, self.n_lag_seasons + 1))
        ]
        valid_points_l3 = [p for p in last_3_seasons_points if p > 0]
        
        if valid_points_l3:
            aggregate_features['avg_ppr_last_3_seasons'] = np.mean(valid_points_l3)
            aggregate_features['std_dev_ppr_last_3_seasons'] = np.std(valid_points_l3)
            aggregate_features['max_ppr_last_3_seasons'] = np.max(valid_points_l3)
            aggregate_features['min_ppr_last_3_seasons'] = np.min(valid_points_l3)
        else:
            aggregate_features['avg_ppr_last_3_seasons'] = 0
            aggregate_features['std_dev_ppr_last_3_seasons'] = 0
            aggregate_features['max_ppr_last_3_seasons'] = 0
            aggregate_features['min_ppr_last_3_seasons'] = 0
        
        return aggregate_features
    
    def _create_trend_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Create year-over-year trend features."""
        trend_features = {}
        
        # PPR points trend (S-1 vs S-2)
        s1_points = features.get('ppr_fantasy_points_s_minus_1', 0)
        s2_points = features.get('ppr_fantasy_points_s_minus_2', 0)
        
        if s1_points > 0 and s2_points > 0:
            trend_features['ppr_trend_s1_vs_s2'] = s1_points - s2_points
            trend_features['ppr_trend_pct_s1_vs_s2'] = (s1_points - s2_points) / s2_points
        else:
            trend_features['ppr_trend_s1_vs_s2'] = 0
            trend_features['ppr_trend_pct_s1_vs_s2'] = 0
        
        # Games played trend
        s1_games = features.get('games_s_minus_1', 0)
        s2_games = features.get('games_s_minus_2', 0)
        trend_features['games_trend_s1_vs_s2'] = s1_games - s2_games
        
        return trend_features
    
    def _create_efficiency_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Create efficiency and rate-based features."""
        efficiency_features = {}
        
        for lag in range(1, self.n_lag_seasons + 1):
            # Yards per attempt metrics
            rush_attempts = features.get(f'rush_attempts_s_minus_{lag}', 0)
            rush_yards = features.get(f'rush_yards_s_minus_{lag}', 0)
            
            if rush_attempts > 0:
                efficiency_features[f'yards_per_rush_s_minus_{lag}'] = rush_yards / rush_attempts
            else:
                efficiency_features[f'yards_per_rush_s_minus_{lag}'] = 0
            
            # Reception efficiency
            targets = features.get(f'targets_s_minus_{lag}', 0)
            receptions = features.get(f'receptions_s_minus_{lag}', 0)
            
            if targets > 0:
                efficiency_features[f'catch_rate_s_minus_{lag}'] = receptions / targets
            else:
                efficiency_features[f'catch_rate_s_minus_{lag}'] = 0
            
            # TD rates
            rush_tds = features.get(f'rush_tds_s_minus_{lag}', 0)
            if rush_attempts > 0:
                efficiency_features[f'rush_td_rate_s_minus_{lag}'] = rush_tds / rush_attempts
            else:
                efficiency_features[f'rush_td_rate_s_minus_{lag}'] = 0
        
        return efficiency_features
    
    def create_prediction_features(self, player_data: pd.DataFrame, target_season: int) -> pd.DataFrame:
        """
        Create features for making predictions on new data.
        
        Args:
            player_data: Historical data for players to predict
            target_season: Season to make predictions for
            
        Returns:
            DataFrame with features for prediction
        """
        logger.info(f"Creating prediction features for {target_season} season")
        
        # Filter data to seasons before target season
        historical_data = player_data[player_data['Season'] < target_season].copy()
        
        if historical_data.empty:
            raise ValueError(f"No historical data available before {target_season}")
        
        # Group by player and create features
        prediction_features = []
        grouped = historical_data.groupby('player_id')
        
        for player_id, player_df in grouped:
            if len(player_df) == 0:
                continue
            
            # Use the most recent season as the "current" for feature creation
            most_recent = player_df.iloc[-1]
            
            features = {
                'player_id': player_id,
                'player_name': most_recent['player'],
                'position': most_recent['position'],
                'prediction_season': target_season
            }
            
            # Create lagged features using all available history
            for lag in range(1, self.n_lag_seasons + 1):
                if len(player_df) >= lag:
                    past_season = player_df.iloc[-lag]
                    for stat in self.stats_to_lag:
                        feature_name = f"{stat}_s_minus_{lag}"
                        features[feature_name] = past_season[stat]
                else:
                    for stat in self.stats_to_lag:
                        feature_name = f"{stat}_s_minus_{lag}"
                        features[feature_name] = 0
            
            # Create derived features
            features.update(self._create_derived_features(features))
            prediction_features.append(features)
        
        prediction_df = pd.DataFrame(prediction_features).fillna(0)
        logger.info(f"Created prediction features for {len(prediction_df)} players")
        
        return prediction_df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names that will be created.
        
        Returns:
            List of feature names
        """
        feature_names = ['player_id', 'player_name', 'position', 'prediction_season']
        
        # Add lagged features
        for lag in range(1, self.n_lag_seasons + 1):
            for stat in self.stats_to_lag:
                feature_names.append(f"{stat}_s_minus_{lag}")
        
        # Add derived features (this is a simplified list - actual implementation creates more)
        derived_features = [
            'ppr_per_game_s_minus_1', 'rush_yards_per_game_s_minus_1',
            'rec_yards_per_game_s_minus_1', 'pass_yards_per_game_s_minus_1',
            'avg_ppr_last_2_seasons', 'avg_ppr_last_3_seasons',
            'std_dev_ppr_last_3_seasons', 'ppr_trend_s1_vs_s2'
        ]
        
        feature_names.extend(derived_features)
        return feature_names 