# This script is used to correct the headers of the csv files pulled from pro-football-reference.com
# The data was pulled from their website using the csv and copy & paste export option

# The data in the project has already been cleaned with this script, but it is kept here for reference

import pandas as pd
import os

# --- Configuration ---
# Path to the directory containing the CSV files
DATA_DIR = '../data_collection/player_data_csvs/'

# The file with the correct, manually-edited header row
TEMPLATE_FILE = os.path.join(DATA_DIR, 'fantasy_2023.csv')

# The range of files to apply the new header to
SEASONS_TO_FIX = [2024] # Temporarily targeting only the 2024 file

# --- Main Script ---

try:
    print(f"Reading correct header from: {TEMPLATE_FILE}")
    template_df = pd.read_csv(TEMPLATE_FILE, nrows=1)
    correct_header = template_df.columns.tolist()
    print("Successfully read template header.")
except Exception as e:
    print(f"Error: Could not read the template file. {e}")
    exit() 

print(f"\\nHeader to be applied: {correct_header}")

# Loop through the old csvs and apply the new header
for season in SEASONS_TO_FIX:
    file_path = os.path.join(DATA_DIR, f'fantasy_{season}.csv')
    
    if os.path.exists(file_path):
        print(f"\\nProcessing {file_path}...")
        try:
            # Read the CSV, skipping the first two incorrect header rows and assign the correct header from manually cleaned template file
            season_df = pd.read_csv(file_path, header=None, skiprows=2, names=correct_header)
            
            # Drop rows where 'rank' is not a digit (removes any lingering bad data)
            season_df = season_df[pd.to_numeric(season_df['rank'], errors='coerce').notna()]
            
            # Overwrite the original file with the corrected data
            season_df.to_csv(file_path, index=False)
            
            print(f"Successfully corrected and saved {file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    else:
        print(f"Warning: File not found for season {season} at {file_path}")

print("\\n\\nCorrection script finished.") 