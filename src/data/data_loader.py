"""
Data loading and initial validation utilities.

Provides functions to load fantasy football player data from various sources,
validate data integrity, and perform initial cleaning operations.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from ..utils.config import get_data_config

logger = logging.getLogger(__name__)


class FantasyDataLoader:
    """
    Fantasy football data loader with validation and cleaning capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data loader with configuration.
        
        Args:
            config: Configuration dictionary. If None, loads default config.
        """
        self.config = get_data_config(config)
        self.data_dir = Path(self.config['raw_data_dir'])
        
    def load_seasons_data(self, seasons: List[int] = None) -> pd.DataFrame:
        """
        Load fantasy football data for multiple seasons.
        
        Args:
            seasons: List of seasons to load. If None, uses config default.
            
        Returns:
            Combined DataFrame with all seasons data
            
        Raises:
            FileNotFoundError: If data files don't exist
            ValueError: If data validation fails
        """
        if seasons is None:
            seasons = self.config['seasons_for_training']
        
        logger.info(f"Loading data for seasons: {seasons}")
        
        season_dataframes = []
        
        for season in seasons:
            try:
                season_df = self._load_single_season(season)
                season_dataframes.append(season_df)
                logger.info(f"Loaded {len(season_df)} records for {season} season")
            except FileNotFoundError as e:
                logger.warning(f"Data file not found for season {season}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error loading data for season {season}: {e}")
                raise
        
        if not season_dataframes:
            raise ValueError("No season data was successfully loaded")
        
        # Combine all seasons
        combined_df = pd.concat(season_dataframes, ignore_index=True)
        logger.info(f"Successfully combined data from {len(season_dataframes)} seasons. "
                   f"Total records: {len(combined_df)}")
        
        # Validate the combined dataset
        self._validate_data(combined_df)
        
        return combined_df
    
    def _load_single_season(self, season: int) -> pd.DataFrame:
        """
        Load data for a single season.
        
        Args:
            season: Year of the season to load
            
        Returns:
            DataFrame with season data
            
        Raises:
            FileNotFoundError: If season file doesn't exist
        """
        file_path = self.data_dir / f"fantasy_{season}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            df['Season'] = season  # Add season identifier
            return df
        except Exception as e:
            raise ValueError(f"Error reading CSV file {file_path}: {e}")
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate that the loaded data has required columns and reasonable values.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        required_columns = [
            'player', 'position', 'ppr_fantasy_points', 'games', 'Season'
        ]
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types and ranges
        if df['Season'].isna().any():
            raise ValueError("Season column contains null values")
        
        if df['ppr_fantasy_points'].isna().sum() > len(df) * 0.1:
            logger.warning("More than 10% of PPR fantasy points are missing")
        
        # Check for reasonable value ranges
        if (df['games'] < 0).any() or (df['games'] > 20).any():
            logger.warning("Some players have unrealistic games played values")
        
        logger.info("Data validation completed successfully")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the raw data.
        
        Args:
            df: Raw DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process")
        df_clean = df.copy()
        
        # Clean player names - remove Pro Bowl (*) and All-Pro (+) indicators
        if 'player' in df_clean.columns:
            df_clean['player'] = df_clean['player'].str.replace(r'[\\*\\+]', '', regex=True).str.strip()
            logger.info("Cleaned player names")
        
        # Identify numeric and categorical columns
        id_columns = ['player', 'team', 'position', 'player_id']
        numeric_columns = [col for col in df_clean.columns if col not in id_columns]
        
        # Convert numeric columns and handle missing values
        for col in numeric_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # Convert appropriate columns to integers
        int_columns = [
            'rank', 'age', 'games', 'games_started', 'completions', 'pass_attempts',
            'pass_yards', 'pass_tds', 'ints', 'rush_attempts', 'rush_yards', 'rush_tds',
            'targets', 'receptions', 'rec_yds', 'rec_tds', 'fumbles', 'fumbles_lost',
            'rush_and_rec_td_total', 'two_point_convs_made', 'two_point_conv_passes',
            'pos_rank', 'overall_rank', 'Season'
        ]
        existing_int_columns = [col for col in int_columns if col in df_clean.columns]
        df_clean[existing_int_columns] = df_clean[existing_int_columns].astype(np.int64)
        
        logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def load_espn_rankings(self, season: int) -> pd.DataFrame:
        """
        Load ESPN pre-season rankings for comparison.
        
        Args:
            season: Season year for rankings
            
        Returns:
            DataFrame with ESPN rankings
        """
        external_dir = Path(self.config['external_data_dir'])
        file_path = external_dir / f"espn_{season}_top_300_ppr.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"ESPN rankings file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded ESPN rankings for {season}: {len(df)} players")
            return df
        except Exception as e:
            raise ValueError(f"Error loading ESPN rankings: {e}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the loaded data.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            'total_records': len(df),
            'seasons': sorted(df['Season'].unique()) if 'Season' in df.columns else [],
            'positions': df['position'].value_counts().to_dict() if 'position' in df.columns else {},
            'date_range': {
                'min_season': df['Season'].min() if 'Season' in df.columns else None,
                'max_season': df['Season'].max() if 'Season' in df.columns else None
            },
            'missing_data': df.isnull().sum().to_dict(),
            'columns': list(df.columns)
        }
        
        return summary


def load_and_clean_data(seasons: List[int] = None, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Convenience function to load and clean fantasy football data.
    
    Args:
        seasons: List of seasons to load
        config: Configuration dictionary
        
    Returns:
        Cleaned DataFrame ready for feature engineering
        
    Example:
        >>> df = load_and_clean_data([2020, 2021, 2022])
        >>> print(df.shape)
        (2500, 35)
    """
    loader = FantasyDataLoader(config)
    raw_data = loader.load_seasons_data(seasons)
    clean_data = loader.clean_data(raw_data)
    
    return clean_data 