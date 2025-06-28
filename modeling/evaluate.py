import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr

# --- Configuration ---
PREDICTIONS_PATH = '2024_fantasy_predictions.csv'
ACTUAL_2024_DATA_PATH = '../data_collection/player_data_csvs/fantasy_2024.csv'

# --- Main Evaluation Script ---
def load_data_for_evaluation():
    """Loads the model's predictions and the actual 2024 season results."""
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"Error: Predictions file not found at {PREDICTIONS_PATH}")
        return None, None
        
    if not os.path.exists(ACTUAL_2024_DATA_PATH):
        print(f"Error: Actual 2024 data not found at {ACTUAL_2024_DATA_PATH}")
        return None, None
        
    print("Loading predictions and actual 2024 data...")
    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    actual_df = pd.read_csv(ACTUAL_2024_DATA_PATH)
    
    # Clean player names in actual data for consistent merging
    actual_df['player'] = actual_df['player'].str.replace(r'[\\*\\+]', '', regex=True).str.strip()
    
    # Only need player name and their actual PPR points from the 2024 data
    actual_df = actual_df[['player', 'ppr_fantasy_points']]
    actual_df = actual_df.rename(columns={'ppr_fantasy_points': 'actual_ppr_2024'})
    
    # Create the actual 2024 rank based on points scored
    actual_df = actual_df.sort_values(by='actual_ppr_2024', ascending=False)
    actual_df['actual_rank_2024'] = np.arange(1, len(actual_df) + 1)
    
    return predictions_df, actual_df

def main():
    predictions_df, actual_df = load_data_for_evaluation()
    
    if predictions_df is None or actual_df is None:
        print("Exiting due to missing data.")
        return
        
    # Merge predictions with the actual results based on player name
    # Use an inner merge to only evaluate players present in both datasets - eliminates issues with players like 2024 rookies who are not in the historical data
    eval_df = pd.merge(predictions_df, actual_df, left_on='player_name', right_on='player', how='inner')
    
    print(f"\\nSuccessfully merged {len(eval_df)} players for evaluation.")
    
    # --- Evaluation Metric Calculation ---
    
    # 1. Spearman's Rank Correlation Coefficient
    # This measures the statistical dependence between two sets of rankings.
    # A value of +1 is a perfect positive correlation, -1 is perfect negative, and 0 is no correlation.
    
    # Ensure no NaN values in rank columns before calculating correlation
    eval_df_cleaned = eval_df.dropna(subset=['model_rank', 'espn_rank', 'actual_rank_2024'])
    
    model_spearman, _ = spearmanr(eval_df_cleaned['model_rank'], eval_df_cleaned['actual_rank_2024'])
    espn_spearman, _ = spearmanr(eval_df_cleaned['espn_rank'], eval_df_cleaned['actual_rank_2024'])
    
    # 2. Mean Absolute Error of Ranks
    # On average, how far off was the rank prediction from the actual final rank?
    eval_df_cleaned['model_rank_error'] = abs(eval_df_cleaned['model_rank'] - eval_df_cleaned['actual_rank_2024'])
    eval_df_cleaned['espn_rank_error'] = abs(eval_df_cleaned['espn_rank'] - eval_df_cleaned['actual_rank_2024'])
    
    model_mae_rank = eval_df_cleaned['model_rank_error'].mean()
    espn_mae_rank = eval_df_cleaned['espn_rank_error'].mean()
    
    # --- Print Results ---
    print("\\n--- Model vs. ESPN Benchmark (2024 Season) ---")
    print("\\n** Spearman's Rank Correlation (Higher is Better) **")
    print(f"  Your Model: {model_spearman:.4f}")
    print(f"  ESPN      : {espn_spearman:.4f}")
    
    print("\\n** Mean Absolute Rank Error (Lower is Better) **")
    print(f"  Your Model: {model_mae_rank:.2f} ranks")
    print(f"  ESPN      : {espn_mae_rank:.2f} ranks")
    
    if model_spearman > espn_spearman and model_mae_rank < espn_mae_rank:
        print("\\nVerdict: Congratulations! Your model outperformed the ESPN pre-season rankings.")
    elif model_spearman < espn_spearman and model_mae_rank > espn_mae_rank:
        print("\\nVerdict: The ESPN pre-season rankings were more accurate than the model's predictions.")
    else:
        print("\\nVerdict: The results are mixed. One metric is better for the model, the other for ESPN.")

    # Save the detailed evaluation file for further analysis
    eval_output_path = '2024_evaluation_results.csv'
    eval_df_cleaned.to_csv(eval_output_path, index=False)
    print(f"\\nDetailed evaluation data saved to: {os.path.abspath(eval_output_path)}")

if __name__ == '__main__':
    main() 