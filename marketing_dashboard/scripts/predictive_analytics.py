#!/usr/bin/env python3
"""
Marketing Campaign Predictive Analytics Script
This script uses machine learning to predict future campaign performance
and identify factors that influence marketing success.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_integrated_data(file_path):
    """
    Load the integrated marketing data
    
    Args:
        file_path (str): Path to the integrated data file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    print(f"Loading integrated marketing data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records.")
    return df

def prepare_data_for_modeling(df, target_metric='roas'):
    """
    Prepare the data for predictive modeling
    
    Args:
        df (pandas.DataFrame): Integrated marketing data
        target_metric (str): Target metric to predict
        
    Returns:
        tuple: (X, y, feature_names) - Features, target, and feature names
    """
    print(f"Preparing data for predicting {target_metric}...")
    
    # Create a copy to avoid modifying the original
    model_df = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['campaign_date', 'start_date', 'end_date']
    for col in date_columns:
        if col in model_df.columns:
            model_df[col] = pd.to_datetime(model_df[col])
    
    # Filter to only matched records with valid target
    model_df = model_df[
        (model_df['match_quality'] == 'Matched') & 
        (model_df[target_metric].notna())
    ].reset_index(drop=True)
    
    print(f"Using {len(model_df)} records for modeling after filtering.")
    
    # Define features to use
    numeric_features = [
        'ad_spend', 'ad_clicks', 'ad_impressions', 'ad_CTR', 'ad_conversion_rate',
        'social_engagements', 'social_reach', 'social_CTR',
        'email_recipients', 'email_opens', 'email_clicks', 'email_open_rate', 'email_click_rate',
        'campaign_duration', 'budget_utilization'
    ]
    
    # Filter to only include features that exist in the dataframe
    numeric_features = [f for f in numeric_features if f in model_df.columns]
    
    categorical_features = [
        'ad_platform', 'social_platform', 'channel'
    ]
    
    # Filter to only include features that exist in the dataframe
    categorical_features = [f for f in categorical_features if f in model_df.columns]
    
    # Extract temporal features
    if 'campaign_date' in model_df.columns:
        model_df['month'] = model_df['campaign_date'].dt.month
        model_df['quarter'] = model_df['campaign_date'].dt.quarter
        model_df['year'] = model_df['campaign_date'].dt.year
        model_df['day_of_week'] = model_df['campaign_date'].dt.dayofweek
        
        numeric_features.extend(['month', 'quarter', 'year'])
        categorical_features.append('day_of_week')
    
    # Combine all features
    all_features = numeric_features + categorical_features
    
    # Create feature matrix and target vector
    X = model_df[all_features]
    y = model_df[target_metric]
    
    print(f"Created feature matrix with {X.shape[1]} features.")
    return X, y, all_features

def build_prediction_pipeline(X, categorical_features):
    """
    Build a scikit-learn pipeline for preprocessing and modeling
    
    Args:
        X (pandas.DataFrame): Feature matrix
        categorical_features (list): List of categorical feature names
        
    Returns:
        tuple: (numeric_transformer, categorical_transformer, preprocessor) - Preprocessing components
    """
    # Get numeric feature names
    numeric_features = [col for col in X.columns if col not in categorical_features]
    
    # Create transformers for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return numeric_transformer, categorical_transformer, preprocessor

def train_and_evaluate_models(X, y, categorical_features):
    """
    Train and evaluate multiple regression models
    
    Args:
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        categorical_features (list): List of categorical feature names
        
    Returns:
        tuple: (best_model, best_pipeline, model_results) - Best model, pipeline, and results
    """
    print("Training and evaluating predictive models...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build preprocessing pipeline
    _, _, preprocessor = build_prediction_pipeline(X, categorical_features)
    
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Store results
    model_results = {}
    best_score = -np.inf
    best_model_name = None
    best_model = None
    best_pipeline = None
    
    # Evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        model_results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Pipeline': pipeline
        }
        
        print(f"  {name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Update best model if this one is better
        if r2 > best_score:
            best_score = r2
            best_model_name = name
            best_model = model
            best_pipeline = pipeline
    
    print(f"\nBest model: {best_model_name} with R² = {best_score:.4f}")
    return best_model, best_pipeline, model_results

def tune_best_model(X, y, best_model, categorical_features):
    """
    Tune the hyperparameters of the best model
    
    Args:
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        best_model: Best model from initial evaluation
        categorical_features (list): List of categorical feature names
        
    Returns:
        tuple: (tuned_model, tuned_pipeline) - Tuned model and pipeline
    """
    print("Tuning hyperparameters for the best model...")
    
    # Build preprocessing pipeline
    _, _, preprocessor = build_prediction_pipeline(X, categorical_features)
    
    # Define parameter grid based on model type
    if isinstance(best_model, RandomForestRegressor):
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif isinstance(best_model, GradientBoostingRegressor):
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5, 10]
        }
    elif isinstance(best_model, (Ridge, Lasso)):
        param_grid = {
            'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
    else:  # Linear Regression doesn't have hyperparameters to tune
        print("Linear Regression doesn't have hyperparameters to tune.")
        return best_model, Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', best_model)
        ])
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_.named_steps['model'], grid_search.best_estimator_

def analyze_feature_importance(model, feature_names, categorical_features, preprocessor):
    """
    Analyze and visualize feature importance
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        categorical_features (list): List of categorical feature names
        preprocessor: Fitted preprocessor
        
    Returns:
        pandas.DataFrame: Feature importance dataframe
    """
    print("Analyzing feature importance...")
    
    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
        
        # Get feature names after preprocessing
        numeric_features = [f for f in feature_names if f not in categorical_features]
        
        # Get one-hot encoded feature names
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        encoded_features = cat_encoder.get_feature_names_out(categorical_features)
        
        # Combine all feature names
        all_features = np.concatenate([numeric_features, encoded_features])
        
        # Create dataframe of feature importances
        feature_importance = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        
        # Save the plot
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plot_path = os.path.join(base_dir, 'data', 'feature_importance.png')
        plt.savefig(plot_path)
        print(f"Feature importance plot saved to {plot_path}")
        
        return feature_importance
    else:
        # For linear models
        print("Feature importance not available for this model type.")
        return None

def forecast_future_performance(model_pipeline, X, y, forecast_periods=3):
    """
    Forecast future campaign performance
    
    Args:
        model_pipeline: Trained model pipeline
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        forecast_periods (int): Number of periods to forecast
        
    Returns:
        pandas.DataFrame: Forecast dataframe
    """
    print(f"Forecasting performance for the next {forecast_periods} periods...")
    
    # Get the latest data point
    latest_data = X.iloc[-1].copy()
    
    # Create a dataframe to store forecasts
    forecasts = []
    
    # Generate forecasts for future periods
    for i in range(1, forecast_periods + 1):
        # Create a copy of the latest data
        future_data = latest_data.copy()
        
        # Update temporal features if they exist
        if 'month' in future_data:
            # Calculate future date
            if 'campaign_date' in X.columns:
                latest_date = pd.to_datetime(X['campaign_date'].iloc[-1])
                future_date = latest_date + pd.DateOffset(months=i)
                
                # Update temporal features
                future_data['month'] = future_date.month
                future_data['quarter'] = (future_date.month - 1) // 3 + 1
                future_data['year'] = future_date.year
                future_data['day_of_week'] = future_date.dayofweek
        
        # Make prediction
        future_data_df = pd.DataFrame([future_data])
        prediction = model_pipeline.predict(future_data_df)[0]
        
        # Store forecast
        forecast = {
            'Period': i,
            'Predicted Value': prediction
        }
        
        # Add features to forecast
        for col in future_data.index:
            forecast[col] = future_data[col]
        
        forecasts.append(forecast)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame(forecasts)
    
    return forecast_df

def save_model_and_results(model_pipeline, model_results, feature_importance, forecast_df, target_metric):
    """
    Save the trained model, results, and forecasts
    
    Args:
        model_pipeline: Trained model pipeline
        model_results (dict): Model evaluation results
        feature_importance (pandas.DataFrame): Feature importance dataframe
        forecast_df (pandas.DataFrame): Forecast dataframe
        target_metric (str): Target metric that was predicted
    """
    print("Saving model and results...")
    
    # Create paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'data')
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f'{target_metric}_prediction_model.joblib')
    joblib.dump(model_pipeline, model_path)
    print(f"Model saved to {model_path}")
    
    # Save model results
    results_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'MSE': [results['MSE'] for results in model_results.values()],
        'RMSE': [results['RMSE'] for results in model_results.values()],
        'MAE': [results['MAE'] for results in model_results.values()],
        'R²': [results['R²'] for results in model_results.values()]
    })
    
    results_path = os.path.join(results_dir, 'model_evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Model evaluation results saved to {results_path}")
    
    # Save feature importance if available
    if feature_importance is not None:
        importance_path = os.path.join(results_dir, 'feature_importance.csv')
        feature_importance.to_csv(importance_path, index=False)
        print(f"Feature importance saved to {importance_path}")
    
    # Save forecasts
    forecast_path = os.path.join(results_dir, f'{target_metric}_forecasts.csv')
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Forecasts saved to {forecast_path}")

def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    integrated_file = os.path.join(base_dir, 'data', 'integrated_marketing_data.csv')
    
    # Define target metrics to predict
    target_metrics = ['roas', 'ctr', 'cpc', 'cpa']
    
    # Process each target metric
    for target_metric in target_metrics:
        print(f"\n=== Predicting {target_metric.upper()} ===")
        
        # Load integrated data
        df = load_integrated_data(integrated_file)
        
        # Prepare data for modeling
        X, y, feature_names = prepare_data_for_modeling(df, target_metric)
        
        # Get categorical features
        categorical_features = [col for col in X.columns if X[col].dtype == 'object']
        
        # Train and evaluate models
        best_model, best_pipeline, model_results = train_and_evaluate_models(X, y, categorical_features)
        
        # Tune the best model
        tuned_model, tuned_pipeline = tune_best_model(X, y, best_model, categorical_features)
        
        # Analyze feature importance
        feature_importance = analyze_feature_importance(
            tuned_model, 
            feature_names, 
            categorical_features, 
            tuned_pipeline.named_steps['preprocessor']
        )
        
        # Forecast future performance
        forecast_df = forecast_future_performance(tuned_pipeline, X, y)
        
        # Save model and results
        save_model_and_results(tuned_pipeline, model_results, feature_importance, forecast_df, target_metric)
    
    print("\nPredictive analytics completed successfully!")

if __name__ == "__main__":
    main() 