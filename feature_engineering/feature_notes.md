# Data Handling & Feature Engineering Notes

Below is a summary of key decisions and considerations for data handling and feature engineering in the fantasy football prediction model.

## I. Data Handling Strategy

### A. Role of ESPN preseason Top 300/N Lists:

1.  **Target Prediction Year (e.g., 2024 season prediction):**
    * **Primary Use:** The ESPN preseason list for the *target year* will be used to define the **pool of relevant and active players for whom predictions will be made**.
    * This acts as a practical filter for players who are retired, expected to be out for the season (due to preseason injury if known by ESPN), or are otherwise not considered fantasy-relevant by experts for the upcoming season.

2.  **Historical Training Data (e.g., seasons 2004-2023):**
    * **Avoid Strict Filtering:** Do NOT use historical ESPN preseason top 300 lists to filter *historical training data*.
    * **Why We Do This:** The model needs to learn from actual past performances. Filtering based on past expert predictions would exclude players who had breakout seasons despite not being on the preseason radar, or players experts were wrong about. This would mean losing valuable training instances. Think James Robinson in 2020. ESPN PPR top 300 had him at 139. He finished the season as 18 overall and the number 7 RB. Filtering only the top 100 players because just because they tend to be consistent season-to-season would miss this useful piece of data.

### B. Defining the Historical Training Player Pool:

1.  **Based on Actual Outcomes:** For each historical season used in training, the player pool should be defined by *actual performance* in that season.
    * **Method:** Use the CSVs of all NFL players and select the "top 200 actual PPR performers" for each historical season. Alternatively, filter by players who played a minimum number of games in that season.
    * **Target Variable Context:** The target variable for training (e.g., `Actual_Fantasy_Points_Season_S`) will be the actual points scored by these players in that historical season `S`.

### C. Handling Player Volatility (Injuries, Performance Dips, Comebacks):

1.  **Through Feature Engineering:** The model will learn to account for these scenarios based on the numerical patterns in the features provided. It does not "know" a player was injured in the way we talk about it, for example.
2.  **Importance of `Games_Played`:** Including `Games_Played` in past seasons as features is crucial. This helps the model differentiate between:
    * A player scoring low points due to playing few games (think injury).
    * A player scoring low points despite playing many games (think perhaps a decline in skill or role).
3.  **Example (McCaffrey Scenario):**
    * Strong history + high games played (S1-S3) -> good prediction.
    * Sudden drop in points + low games played (S4 due to injury).
    * When S4 becomes historical data for predicting S5, the features will reflect the strong S1-S3 *and* the injury-contextualized S4. The model will learn if players with such profiles tend to bounce back, especially if other features (like age) are still favorable.

### D. Handling Retirements & Long-Term Irrelevance:

1.  **During Training:**
    * If a player was great for N seasons and then retires (doesn't play in season N+1), their data from the N active seasons is valuable training input.
    * They simply *won't have an actual performance target* for season N+1 in the training set because they didn't play. The model isn't "penalized" by a zero-score for a non-playing season. That data point just doesn't exist for them as an active player in N+1.
2.  **For Prediction (Target Year):**
    * The ESPN preseason list for the target year should ideally already have filtered out players who have retired or are known to be out for the season. The model will therefore not be asked to make a prediction for them.

## II. Key Features for Model Training

The goal is to create a feature set for each player `P` for whom we are predicting performance in a target season `S`. These features will be based on their historical performance in seasons `S-1`, `S-2`, `S-3`, etc.

### A. Player Identifiers & Context (for Season S):

* `PlayerID`: Unique identifier for the player.
* `PlayerName`: Name of the player.
* `Season_of_Prediction`: The NFL season for which the prediction is being made (e.g., 2024).
* `Age_at_Start_of_Season_S`: Player's age at the start of season `S`.
* `Years_of_NFL_Experience_at_Start_of_Season_S`: Number of seasons played in the NFL prior to season `S`.
* `Position`: (e.g., QB, RB, WR, TE) - Potentially one-hot encode or use embeddings.

### B. Lagged Performance Statistics (from prior seasons `S-1`, `S-2`, `S-3`...):

* `Games_Played_Sn`: (e.g., `Games_Played_S-1`, `Games_Played_S-2`)
* `Games_Started_Sn`: (e.g., `Games_Started_S-1`, `Games_Started_S-2`)
* `Fantasy_Points_PPR_Total_Sn`: Total PPR fantasy points in season `S-n`.
* **Core NFL Statistics (repeat for `S-1`, `S-2`, etc.):**
    * **Passing (for QBs primarily):**
        * `Pass_Attempts_Sn`, `Pass_Completions_Sn`, `Passing_Yards_Sn`, `Passing_TDs_Sn`, `Interceptions_Thrown_Sn`, `Sacks_Taken_Sn`
    * **Rushing (for RBs, QBs, some WRs):**
        * `Rushing_Attempts_Sn`, `Rushing_Yards_Sn`, `Rushing_TDs_Sn`, `Fumbles_Sn`, `Fumbles_Lost_Sn`
    * **Receiving (for WRs, TEs, RBs):**
        * `Targets_Sn`, `Receptions_Sn`, `Receiving_Yards_Sn`, `Receiving_TDs_Sn`
    * *(Will likely add more specific stats as available, such as yards per attempt, completion percentage, target share, etc.)*

### C. Derived & Trend Features (calculated from lagged stats):

* **Per-Game Averages (for `S-1`, `S-2`...):**
    * `Fantasy_Points_PPR_per_Game_Sn` (e.g., `Fantasy_Points_PPR_Total_S-1 / Games_Played_S-1`)
    * `Rushing_Yards_per_Game_Sn`, `Receiving_Yards_per_Game_Sn`, etc.
* **Efficiency Metrics (for `S-1`, `S-2`...):**
    * `Yards_per_Rush_Attempt_Sn`, `Yards_per_Reception_Sn`, `Yards_per_Pass_Attempt_Sn`
    * `Catch_Rate_Sn` (Receptions_Sn / Targets_Sn)
    * `TD_Rate_Passing_Sn` (Passing_TDs_Sn / Pass_Attempts_Sn)
* **Multi-Season Aggregates/Averages:**
    * `Avg_Fantasy_Points_Last_N_Seasons` (e.g., N=2 or 3)
    * `Avg_Games_Played_Last_N_Seasons`
    * `Total_TDs_Last_N_Seasons`
* **Volatility/Consistency Metrics:**
    * `StdDev_Fantasy_Points_Last_N_Seasons`
    * `Min_Fantasy_Points_Last_N_Seasons`, `Max_Fantasy_Points_Last_N_Seasons`
* **Year-over-Year Trends:**
    * `Fantasy_Points_Change_S-1_vs_S-2` (Fantasy_Points_PPR_Total_S-1 - Fantasy_Points_PPR_Total_S-2)
    * `Games_Played_Change_S-1_vs_S-2`
    * Percentage change for key stats.
* **Health/Participation Indicators:**
    * `Percentage_of_Possible_Games_Played_Sn` (Games_Played_Sn / League_Max_Games_in_Season_Sn)

### D. Target Variable (What the model predicts for Season `S`):

* **`Actual_Fantasy_Points_PPR_Season_S`**: The actual total PPR fantasy points scored by the player in the target season `S`.

## III. Important Considerations:

* **Handling Missing Data:** Need to decide on a strategy for players with limited history (e.g., 2nd-year players will only have S-1 data). Imputation (e.g., with 0 or mean/median for a position) or specific flags might be needed.
* **Feature Scaling:** Standardize or normalize numerical features before feeding them to many ML models.
* **Curse of Dimensionality:** Be mindful not to add too many features, especially if the number of player-seasons in the training data is not vast. Feature selection or dimensionality reduction techniques might be useful later.
* **Positional Differences:** Patterns for QBs, RBs, WRs, and TEs can be very different. Will likely need to consider:
    * Including `Position` as a feature.
    * Potentially training separate models per position if initial results with a combined model are not satisfactory (though this might add unnecessary complexity).