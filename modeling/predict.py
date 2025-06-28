import pandas as pd
import numpy as np
import os
import joblib

# --- Configuration ---
MODEL_PATH = 'fantasy_football_predictor.joblib'
HISTORICAL_DATA_PATH = '../data_collection/player_data_csvs/'
ESPN_RANKINGS_PATH = '../espn_top_300_ppr_by_year/espn_2024_top_300_ppr.csv'
OUTPUT_PATH = '2024_fantasy_predictions.csv'
N_LAG_SEASONS = 3 # Must be the same as used in training!

STATS_TO_LAG = [
    'age', 'games', 'games_started', 'completions', 'pass_attempts', 
    'pass_yards', 'pass_tds', 'ints', 'rush_attempts', 'rush_yards', 
    'yards_per_rush_attempt', 'rush_tds', 'targets', 'receptions', 'rec_yds', 
    'yards_per_rec', 'rec_tds', 'fumbles', 'fumbles_lost',
    'ppr_fantasy_points'
]

# --- Main Prediction Script ---
def load_data():
    """Loads and cleans all necessary data: historical stats, ESPN rankings, and the trained model."""
    print("Loading data...")
    seasons = range(2013, 2024) # 2013-2023
    all_dfs = []
    for season in seasons:
        path = os.path.join(HISTORICAL_DATA_PATH, f'fantasy_{season}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Season'] = season
            all_dfs.append(df)
        else:
            print(f"Warning: Data file not found for season {season}. Skipping.")

    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # --- Start: Added Data Cleaning Logic ---
    # Clean player names
    master_df['player'] = master_df['player'].str.replace(r'[\\*\\+]', '', regex=True).str.strip()
    
    # Define which columns should be numeric vs. identifiers
    id_cols = ['player', 'team', 'position', 'player_id']
    numeric_cols = [col for col in master_df.columns if col not in id_cols]
    
    # Convert columns to numeric, coercing errors, and filling resulting NaNs with 0
    master_df[numeric_cols] = master_df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Define columns that should be whole numbers and convert them
    int_cols = [
        'rank', 'age', 'games', 'games_started', 'completions', 'pass_attempts', 
        'pass_yards', 'pass_tds', 'ints', 'rush_attempts', 'rush_yards', 'rush_tds',
        'targets', 'receptions', 'rec_yds', 'rec_tds', 'fumbles', 'fumbles_lost',
        'rush_and_rec_td_total', 'two_point_convs_made', 'two_point_conv_passes',
        'vbd', 'pos_rank', 'overall_rank', 'Season'
    ]
    existing_int_cols = [col for col in int_cols if col in master_df.columns]
    master_df[existing_int_cols] = master_df[existing_int_cols].astype(np.int64)
    print("Historical data cleaned and prepared.")
    # --- End: Added Data Cleaning Logic ---

    # Load ESPN 2024 rankings
    espn_df = pd.read_csv(ESPN_RANKINGS_PATH)
    espn_df['player_name'] = espn_df['player_name'].str.strip()
    
    # Load the trained model (this was generated in the main notebook)
    model = joblib.load(MODEL_PATH)
    
    return master_df, espn_df, model

def generate_prediction_features(player_name, historical_df):
    """Generates the feature set for a single player for the 2024 season."""
    player_history = historical_df[historical_df['player'] == player_name].sort_values(by='Season', ascending=False)
    
    if player_history.empty:
        return None # Player not found in historical data

    features = {}
    
    # Create lagged features from the most recent seasons (2023, 2022, 2021)
    for lag in range(1, N_LAG_SEASONS + 1):
        if lag - 1 < len(player_history):
            past_season = player_history.iloc[lag - 1]
            for stat in STATS_TO_LAG:
                features[f'{stat}_s_minus_{lag}'] = past_season.get(stat, 0)
        else:
            for stat in STATS_TO_LAG:
                features[f'{stat}_s_minus_{lag}'] = 0
    
    # --- Create Derived Features (must match notebook exactly) ---
    for lag in range(1, N_LAG_SEASONS + 1):
        games = features.get(f'games_s_minus_{lag}', 0)
        if games > 0:
            features[f'ppr_per_game_s_minus_{lag}'] = features[f'ppr_fantasy_points_s_minus_{lag}'] / games
        else:
            features[f'ppr_per_game_s_minus_{lag}'] = 0
            
    last_2 = [features.get(f'ppr_fantasy_points_s_minus_{j}', 0) for j in range(1, 3)]
    features['avg_ppr_last_2_seasons'] = np.mean([p for p in last_2 if p > 0] or [0])
    
    last_3 = [features.get(f'ppr_fantasy_points_s_minus_{j}', 0) for j in range(1, 4)]
    features['avg_ppr_last_3_seasons'] = np.mean([p for p in last_3 if p > 0] or [0])
    features['std_dev_ppr_last_3_seasons'] = np.std([p for p in last_3 if p > 0] or [0])
    
    s1 = features.get('ppr_fantasy_points_s_minus_1', 0)
    s2 = features.get('ppr_fantasy_points_s_minus_2', 0)
    features['ppr_trend_s1_vs_s2'] = s1 - s2 if s1 > 0 and s2 > 0 else 0

    # Add position from the most recent season's data
    features['position'] = player_history.iloc[0]['position']

    return features

def main():
    historical_df, espn_df, model = load_data()
    
    all_prediction_features = []
    skipped_players = []

    print(f"Generating features for {len(espn_df)} players from ESPN list...")
    for _, row in espn_df.iterrows():
        player_name = row['player_name']
        if pd.isna(player_name):
            continue
        
        features = generate_prediction_features(player_name, historical_df)
        
        if features:
            features['player_name'] = player_name
            features['espn_rank'] = row['espn_rank']
            all_prediction_features.append(features)
        else:
            skipped_players.append(player_name)

    print(f"\\nSkipped {len(skipped_players)} players (no historical data found):")
    print(skipped_players)

    if not all_prediction_features:
        print("No players to predict. Exiting.")
        return

    # Convert to DataFrame and prepare for prediction
    predict_df = pd.DataFrame(all_prediction_features)
    
    # Align columns with the model's training columns
    X_predict = predict_df.drop(columns=['player_name', 'espn_rank'])
    X_predict = pd.get_dummies(X_predict, columns=['position'], prefix='pos', drop_first=True)
    
    # Ensure all columns from training are present
    model_cols = model.feature_names_in_
    X_predict = X_predict.reindex(columns=model_cols, fill_value=0)

    # Make predictions
    print("\\nMaking 2024 predictions...")
    predictions = model.predict(X_predict)
    
    # Add predictions to the DataFrame
    predict_df['predicted_ppr_2024'] = predictions
    
    # Rank players based on our model's predictions
    predict_df = predict_df.sort_values(by='predicted_ppr_2024', ascending=False)
    predict_df['model_rank'] = np.arange(1, len(predict_df) + 1)
    
    # Final cleanup for the output file
    output_df = predict_df[['model_rank', 'espn_rank', 'player_name', 'predicted_ppr_2024', 'position']]
    
    # Save the results
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\\nSuccessfully saved 2024 predictions to: {os.path.abspath(OUTPUT_PATH)}")

if __name__ == '__main__':
    main() 