# Fantasy Football Performance Predictor (ML Project)

## 1. Project Goal

The primary goal of this project is to develop a machine learning model that predicts and ranks the top 100 fantasy football players (PPR scoring) for the 2024 NFL season. The aim is to achieve a higher predictive accuracy than ESPN's pre-season top 300 player rankings when compared against the actual performance outcomes of the 2024 season. This model will then be utilized to predict future seasons.

## 2. Core Concept & Methodology

The model will leverage historical NFL player statistics from the past 15 seasons. Instead of relying solely on past fantasy point totals, this project will analyze granular player performance metrics (e.g., rushing yards, passing completion rate, touchdowns, fumbles, receptions) to identify patterns indicative of future fantasy success.

The core methodology is as follows:
1.  **Data Collection:** Gather detailed player statistics and PPR fantasy scores for approximately the top 100 performing players for each of the last 15 NFL seasons.
2.  **Feature Engineering:** Create features for each player based on their historical performance (e.g., stats from season N-1, N-2, multi-year averages, age, experience).
3.  **Model Training:** Train a regression model to predict a player's PPR fantasy points for an upcoming season using their engineered historical features.
4.  **Player Pool for Prediction:** For predicting the 2024 season, the initial player pool will be derived from ESPN's 2024 pre-season top 300 rankings. This serves as a practical filter to identify players expected to be active and relevant for the target season.
5.  **Prediction & Ranking:** The trained model will predict the 2024 PPR fantasy points for each player in the filtered pool. These players will then be ranked based on their predicted scores to generate a top 100 list.
6.  **Evaluation:** The model's predicted rankings for 2024 will be compared against the actual 2024 season-end player performances and rankings. This will also be benchmarked against ESPN's pre-season rankings' accuracy.

## 3. Data Details

* **Input Data:** Historical season-by-season data for individual NFL players. This includes:
    * Standard offensive statistics (passing, rushing, receiving).
    * Turnovers (fumbles, interceptions).
    * Player metadata (age, seasons played).
    * Historical PPR fantasy points.
* **Data Format:** For ease of use with `pandas` and initial data exploration, the primary data format will be **CSV (Comma Separated Values)**. Data might be stored as a single large CSV or as separate CSVs per season.
* **Focus Group:** The model training will primarily focus on patterns observed in players who finished among the top 100 PPR scorers in previous seasons.
* **Features for Prediction:**
    * Lagged statistics (e.g., performance metrics from 1, 2, and potentially 3 seasons prior).
    * Calculated trend indicators (e.g., year-over-year changes).
    * Player age and NFL experience.
    * For simplicity in this initial version, the model will primarily focus on players with existing NFL statistics. Rookies or players with no recent NFL history will not be the primary focus of the *model training*, though they might be part of the ESPN player pool to be ranked.
* **Target Variable:** The model will be trained to predict `Actual PPR Fantasy Points` for a player in the target season.

## 4. Technology Stack

* **Language:** Python
* **Core Libraries:**
    * `pandas` for data manipulation and analysis.
    * `scikit-learn` for machine learning model implementation, feature processing, and evaluation.
    * (Potentially `numpy` for numerical operations).

## 5. Project Structure (High-Level Workflow)

1.  **`data_collection/`**: Scripts and potentially raw data for player stats and fantasy scores. (Note: Data acquisition is a significant initial step).
2.  **`data_preprocessing/`**: Scripts for cleaning, transforming, and merging datasets.
3.  **`feature_engineering/`**: Scripts to create predictive features from the processed data.
4.  **`modeling/`**:
    * `train.py`: Script for training the ML model.
    * `predict.py`: Script to make predictions for a new season.
    * `evaluate.py`: Script to evaluate model performance against actuals and benchmarks.
5.  **`notebooks/`**: Jupyter notebooks for exploratory data analysis (EDA), model experimentation, and visualization.

## 6. Key Challenges & Simplifications

* **Data Acquisition:** Obtaining consistent and accurate historical data for 15 seasons is a primary challenge.
* **Player Churn:** Managing player retirements, new entries, and long-term injuries.
    * *Simplification:* Using ESPN's top 300 pre-season list as a primary filter for the player pool to be ranked for the target season (2024). This assumes ESPN has accounted for most retirements/non-active players.
* **Rookies:** Predicting performance for players with no NFL history is complex.
    * *Simplification:* The model will primarily be trained on players with existing NFL stats. The performance on ranking rookies will depend on their inclusion in the ESPN list and how the model generalizes (or doesn't) to missing historical features.
* **Injuries:** Unpredictable injuries significantly impact fantasy outcomes and are inherently difficult to model.

## 7. Evaluation Metrics

The model's success will be measured by:

* **Predictive Accuracy of Points:**
    * Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE) between predicted and actual fantasy points for the 2024 season.
* **Ranking Accuracy:**
    * Spearman's Rank Correlation Coefficient or Kendall's Tau to compare the model's top 100 player rankings against actual 2024 end-of-season rankings.
    * Hit Rate within Tiers (e.g., percentage of model's predicted top 10/25/50 players that actually finished in the top 10/25/50).
* **Benchmarking:** All metrics will be compared against the predictive accuracy of ESPN's pre-season rankings for the 2024 season.

## 8. Skills Showcased

This project aims to clearly demonstrate fundamentals of a machine learning project lifecycle:
* Problem Definition & Scoping
* Data Collection and Preprocessing Considerations
* Thoughtful Feature Engineering
* Model Selection and Training (Regression)
* Prediction on New Data
* Robust Evaluation against a Clear Benchmark
* Clear Documentation and Code Structure in Python
