import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Import existing functions
sys.path.append('..')
from modeling.train import load_and_clean_data, engineer_features

# Configuration
plt.style.use('default')
sns.set_palette("husl")

def prepare_data():
    """Load and prepare data using existing pipeline."""
    print("Loading and preparing data...")
    master_df = load_and_clean_data()
    training_df = engineer_features(master_df)
    
    # Same preprocessing as model training
    FEATURES_TO_DROP = ['player_id', 'player_name', 'prediction_season', 'target_ppr_points']
    X = training_df.drop(columns=FEATURES_TO_DROP)
    y = training_df['target_ppr_points']
    
    # One-hot encode position
    X = pd.get_dummies(X, columns=['position'], prefix='pos', drop_first=True)
    
    # Use only training data (before 2023) for feature importance
    X_train = X[training_df['prediction_season'] < 2023]
    y_train = y[training_df['prediction_season'] < 2023]
    
    return X_train, y_train, X.columns.tolist()

def get_linear_regression_importance(X_train, y_train, feature_names):
    """Get feature importance for Linear Regression using coefficient magnitudes."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Use absolute coefficients as importance
    importance = np.abs(model.coef_)
    
    return importance, model

def get_tree_based_importance(model_class, X_train, y_train, feature_names, model_name):
    """Get feature importance from tree-based models."""
    if model_name == 'XGBoost':
        model = model_class(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            n_jobs=-1
        )
    elif model_name == 'Random Forest':
        model = model_class(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:  # Gradient Boosting
        model = model_class(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    
    return importance, model

def create_feature_importance_dataframe(importances_dict, feature_names):
    """Create a comprehensive DataFrame with all model importances."""
    df = pd.DataFrame({
        'feature': feature_names
    })
    
    for model_name, importance in importances_dict.items():
        # Normalize to 0-100 scale for easier comparison
        normalized_importance = 100 * importance / importance.max()
        df[f'{model_name}_importance'] = normalized_importance
    
    # Calculate average importance across all models
    importance_cols = [col for col in df.columns if col.endswith('_importance')]
    df['avg_importance'] = df[importance_cols].mean(axis=1)
    
    # Sort by average importance
    df = df.sort_values('avg_importance', ascending=False)
    
    return df

def categorize_features(feature_names):
    """Categorize features into logical groups for analysis."""
    categories = {
        'Lagged PPR Points': [],
        'Lagged Games': [],
        'Lagged Passing': [],
        'Lagged Rushing': [],
        'Lagged Receiving': [],
        'Per-Game Metrics': [],
        'Multi-Season Aggregates': [],
        'Trend Features': [],
        'Position Encoding': [],
        'Other': []
    }
    
    for feature in feature_names:
        if 'ppr_fantasy_points_s_minus' in feature:
            categories['Lagged PPR Points'].append(feature)
        elif 'games' in feature and 's_minus' in feature and 'per_game' not in feature:
            categories['Lagged Games'].append(feature)
        elif any(x in feature for x in ['pass_', 'completions', 'ints']) and 's_minus' in feature:
            categories['Lagged Passing'].append(feature)
        elif any(x in feature for x in ['rush_', 'fumbles']) and 's_minus' in feature:
            categories['Lagged Rushing'].append(feature)
        elif any(x in feature for x in ['rec_', 'targets', 'receptions']) and 's_minus' in feature:
            categories['Lagged Receiving'].append(feature)
        elif 'per_game' in feature:
            categories['Per-Game Metrics'].append(feature)
        elif any(x in feature for x in ['avg_', 'std_dev_']):
            categories['Multi-Season Aggregates'].append(feature)
        elif 'trend' in feature:
            categories['Trend Features'].append(feature)
        elif 'pos_' in feature:
            categories['Position Encoding'].append(feature)
        else:
            categories['Other'].append(feature)
    
    return categories

def plot_top_features(importance_df, top_n=20):
    """Create visualization of top N most important features."""
    top_features = importance_df.head(top_n).copy()
    
    plt.figure(figsize=(12, 10))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    
    # Plot bars for each model
    bar_width = 0.2
    models = ['Linear Regression', 'Random Forest', 'XGBoost', 'Gradient Boosting']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (model, color) in enumerate(zip(models, colors)):
        col_name = f'{model}_importance'
        if col_name in top_features.columns:
            plt.barh(y_pos + i * bar_width, top_features[col_name], 
                    bar_width, label=model, color=color, alpha=0.8)
    
    plt.xlabel('Feature Importance (Normalized to 0-100)')
    plt.ylabel('Features')
    plt.title(f'Top {top_n} Most Important Features Across All Models')
    plt.yticks(y_pos + bar_width * 1.5, top_features['feature'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Save the plot
    plt.savefig('top_features_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_categories(importance_df, feature_categories):
    """Plot importance by feature category."""
    category_importance = {}
    
    for category, features in feature_categories.items():
        if features:  # Only process non-empty categories
            category_features = importance_df[importance_df['feature'].isin(features)]
            if not category_features.empty:
                category_importance[category] = category_features['avg_importance'].mean()
    
    # Create bar plot
    categories = list(category_importance.keys())
    importances = list(category_importance.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, importances, color='skyblue', alpha=0.8)
    plt.xlabel('Feature Category')
    plt.ylabel('Average Importance')
    plt.title('Feature Importance by Category')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, importance in zip(bars, importances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{importance:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('feature_categories_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_linear_regression_coefficients(importance_df, X_train, y_train):
    """Special analysis for Linear Regression coefficients."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Get top positive and negative coefficients
    coef_df = pd.DataFrame({
        'feature': importance_df['feature'],
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\n" + "="*60)
    print("LINEAR REGRESSION COEFFICIENT ANALYSIS")
    print("="*60)
    
    print("\nTop 10 POSITIVE coefficients (increase PPR points):")
    positive_coefs = coef_df[coef_df['coefficient'] > 0].head(10)
    for _, row in positive_coefs.iterrows():
        print(f"  {row['feature']:<40} +{row['coefficient']:.3f}")
    
    print("\nTop 10 NEGATIVE coefficients (decrease PPR points):")
    negative_coefs = coef_df[coef_df['coefficient'] < 0].head(10)
    for _, row in negative_coefs.iterrows():
        print(f"  {row['feature']:<40} {row['coefficient']:.3f}")
    
    return coef_df

def generate_feature_importance_report(importance_df, feature_categories):
    """Generate a comprehensive text report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    print(f"\nAnalyzed {len(importance_df)} features across 4 different models")
    print("Models: Linear Regression, Random Forest, XGBoost, Gradient Boosting")
    
    print("\n" + "-"*50)
    print("TOP 15 MOST IMPORTANT FEATURES (Average Across All Models)")
    print("-"*50)
    
    for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<40} {row['avg_importance']:.2f}")
    
    print("\n" + "-"*50)
    print("FEATURE CATEGORY ANALYSIS")
    print("-"*50)
    
    for category, features in feature_categories.items():
        if features:
            category_features = importance_df[importance_df['feature'].isin(features)]
            if not category_features.empty:
                avg_imp = category_features['avg_importance'].mean()
                count = len(features)
                print(f"{category:<25} {count:3d} features, avg importance: {avg_imp:.2f}")

def main():
    """Main function to run feature importance analysis."""
    print("=== Fantasy Football Feature Importance Analysis ===\n")
    
    # Prepare data
    X_train, y_train, feature_names = prepare_data()
    print(f"Analyzing {len(feature_names)} features on {len(X_train)} training samples")
    
    # Get importance from all models
    print("\nTraining models to extract feature importance...")
    
    importances = {}
    
    # Linear Regression (coefficient magnitudes)
    print("  - Linear Regression...")
    lr_importance, lr_model = get_linear_regression_importance(X_train, y_train, feature_names)
    importances['Linear Regression'] = lr_importance
    
    # Random Forest
    print("  - Random Forest...")
    rf_importance, rf_model = get_tree_based_importance(RandomForestRegressor, X_train, y_train, feature_names, 'Random Forest')
    importances['Random Forest'] = rf_importance
    
    # XGBoost
    print("  - XGBoost...")
    xgb_importance, xgb_model = get_tree_based_importance(XGBRegressor, X_train, y_train, feature_names, 'XGBoost')
    importances['XGBoost'] = xgb_importance
    
    # Gradient Boosting
    print("  - Gradient Boosting...")
    gb_importance, gb_model = get_tree_based_importance(GradientBoostingRegressor, X_train, y_train, feature_names, 'Gradient Boosting')
    importances['Gradient Boosting'] = gb_importance
    
    # Create comprehensive DataFrame
    importance_df = create_feature_importance_dataframe(importances, feature_names)
    
    # Categorize features
    feature_categories = categorize_features(feature_names)
    
    # Generate report
    generate_feature_importance_report(importance_df, feature_categories)
    
    # Linear Regression specific analysis
    coef_df = analyze_linear_regression_coefficients(importance_df, X_train, y_train)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_top_features(importance_df, top_n=20)
    plot_feature_categories(importance_df, feature_categories)
    
    # Save results
    importance_df.to_csv('feature_importance_analysis.csv', index=False)
    coef_df.to_csv('linear_regression_coefficients.csv', index=False)
    
    print(f"\nResults saved:")
    print(f"  - Feature importance: feature_importance_analysis.csv")
    print(f"  - Linear regression coefficients: linear_regression_coefficients.csv")
    print(f"  - Visualizations: top_features_comparison.png, feature_categories_importance.png")
    
    print("\nFeature importance analysis complete!")

if __name__ == '__main__':
    main() 