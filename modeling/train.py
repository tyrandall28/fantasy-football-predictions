# Placeholder for model training script 

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Configuration ---
DATA_DIR = '../data_collection/player_data_csvs'
MODEL_DIR = 'modeling'
MODEL_FILENAME = 'fantasy_football_predictor.joblib'
SEASONS_TO_LOAD = range(2013, 2024)
N_LAG_SEASONS = 3

STATS_TO_LAG = [
    'age', 'games', 'games_started', 'completions', 'pass_attempts', 
    'pass_yards', 'pass_tds', 'ints', 'rush_attempts', 'rush_yards', 
    'yards_per_rush_attempt', 'rush_tds', 'targets', 'receptions', 'rec_yds', 
    'yards_per_rec', 'rec_tds', 'fumbles', 'fumbles_lost',
    'ppr_fantasy_points'
]
TARGET_VARIABLE = 'ppr_fantasy_points'

def load_and_clean_data():
    """Load and clean historical player data."""
    print(f"Looking for data in: {os.path.abspath(DATA_DIR)}")
    all_season_dfs = []
    
    for season in SEASONS_TO_LOAD:
        file_path = os.path.join(DATA_DIR, f'fantasy_{season}.csv')
        if os.path.exists(file_path):
            print(f"Loading data for {season} season...")
            try:
                season_df = pd.read_csv(file_path)
                season_df['Season'] = season
                all_season_dfs.append(season_df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: File not found for season {season} at {file_path}")

    if not all_season_dfs:
        raise ValueError("No data was loaded. Check file paths and season range.")

    # Combine all seasons
    master_df = pd.concat(all_season_dfs, ignore_index=True)
    
    # Clean data
    master_df['player'] = master_df['player'].str.replace(r'[\*\+]', '', regex=True).str.strip()
    
    # Convert numeric columns
    id_cols = ['player', 'team', 'position', 'player_id']
    numeric_cols = [col for col in master_df.columns if col not in id_cols]
    master_df[numeric_cols] = master_df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Convert integer columns
    int_cols = [
        'rank', 'age', 'games', 'games_started', 'completions', 'pass_attempts', 
        'pass_yards', 'pass_tds', 'ints', 'rush_attempts', 'rush_yards', 'rush_tds',
        'targets', 'receptions', 'rec_yds', 'rec_tds', 'fumbles', 'fumbles_lost',
        'rush_and_rec_td_total', 'two_point_convs_made', 'two_point_conv_passes',
        'vbd', 'pos_rank', 'overall_rank', 'Season'
    ]
    existing_int_cols = [col for col in int_cols if col in master_df.columns]
    master_df[existing_int_cols] = master_df[existing_int_cols].astype(np.int64)
    
    return master_df

def engineer_features(master_df):
    """Create features for model training."""
    print("Starting feature engineering...")
    master_df = master_df.sort_values(by=['player_id', 'Season'])
    all_features = []
    grouped = master_df.groupby('player_id')

    for player_id, player_df in grouped:
        for i in range(1, len(player_df)):
            target_season_data = player_df.iloc[i]
            
            features = {
                'player_id': player_id,
                'player_name': target_season_data['player'],
                'position': target_season_data['position'],
                'prediction_season': int(target_season_data['Season']),
                'target_ppr_points': target_season_data[TARGET_VARIABLE]
            }

            # Create lagged features
            for lag in range(1, N_LAG_SEASONS + 1):
                if i - lag >= 0:
                    past_season = player_df.iloc[i - lag]
                    for stat in STATS_TO_LAG:
                        features[f'{stat}_s_minus_{lag}'] = past_season[stat]
                else:
                    for stat in STATS_TO_LAG:
                        features[f'{stat}_s_minus_{lag}'] = 0
            
            # Create derived features
            for lag in range(1, N_LAG_SEASONS + 1):
                games_played = features.get(f'games_s_minus_{lag}', 0)
                if games_played > 0:
                    features[f'ppr_per_game_s_minus_{lag}'] = features[f'ppr_fantasy_points_s_minus_{lag}'] / games_played
                    features[f'rush_yards_per_game_s_minus_{lag}'] = features[f'rush_yards_s_minus_{lag}'] / games_played
                    features[f'rec_yards_per_game_s_minus_{lag}'] = features[f'rec_yds_s_minus_{lag}'] / games_played
                    features[f'pass_yards_per_game_s_minus_{lag}'] = features[f'pass_yards_s_minus_{lag}'] / games_played
                else:
                    features[f'ppr_per_game_s_minus_{lag}'] = 0
                    features[f'rush_yards_per_game_s_minus_{lag}'] = 0
                    features[f'rec_yards_per_game_s_minus_{lag}'] = 0
                    features[f'pass_yards_per_game_s_minus_{lag}'] = 0

            # Multi-season aggregates
            last_2_seasons_points = [features.get(f'ppr_fantasy_points_s_minus_{j}', 0) for j in range(1, 3)]
            valid_points_l2 = [p for p in last_2_seasons_points if p > 0]
            features['avg_ppr_last_2_seasons'] = np.mean(valid_points_l2) if valid_points_l2 else 0
            
            last_3_seasons_points = [features.get(f'ppr_fantasy_points_s_minus_{j}', 0) for j in range(1, 4)]
            valid_points_l3 = [p for p in last_3_seasons_points if p > 0]
            features['avg_ppr_last_3_seasons'] = np.mean(valid_points_l3) if valid_points_l3 else 0
            features['std_dev_ppr_last_3_seasons'] = np.std(valid_points_l3) if valid_points_l3 else 0

            # Year-over-year trends
            s1_points = features.get('ppr_fantasy_points_s_minus_1', 0)
            s2_points = features.get('ppr_fantasy_points_s_minus_2', 0)
            features['ppr_trend_s1_vs_s2'] = s1_points - s2_points if s1_points > 0 and s2_points > 0 else 0
            
            all_features.append(features)

    training_df = pd.DataFrame(all_features).fillna(0)
    return training_df

def train_model(training_df):
    """Train the Gradient Boosting model."""
    # Prepare features and target
    FEATURES_TO_DROP = ['player_id', 'player_name', 'prediction_season', 'target_ppr_points']
    X = training_df.drop(columns=FEATURES_TO_DROP)
    y = training_df['target_ppr_points']

    # One-hot encode position
    X = pd.get_dummies(X, columns=['position'], prefix='pos', drop_first=True)

    # Split data chronologically
    X_train = X[training_df['prediction_season'] < 2023]
    y_train = y[training_df['prediction_season'] < 2023]
    X_test = X[training_df['prediction_season'] == 2023]
    y_test = y[training_df['prediction_season'] == 2023]

    print("Data split into training and testing sets:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")

    # Train model
    print("\nTraining the Gradient Boosting Regressor model...")
    gbr = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    gbr.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate model
    print("\nEvaluating model performance on the unseen 2023 test set...")
    test_predictions = gbr.predict(X_test)
    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)

    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  R-squared (R²): {r2:.3f}")
    print("\n(MAE - on average, how many fantasy points the prediction was off by)")
    print("(R² - what percentage of the variance in the target variable the model explains)")

    return gbr

def save_model(model):
    """Save the trained model to disk."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {os.path.abspath(model_path)}")

def main():
    try:
        # Load and clean data
        master_df = load_and_clean_data()
        
        # Engineer features
        training_df = engineer_features(master_df)
        
        # Train model
        model = train_model(training_df)
        
        # Save model
        save_model(model)
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 