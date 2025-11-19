#!/usr/bin/env python3
"""
F1 Crash Predictor - Main Pipeline
End-to-end ML pipeline for predicting F1 crash likelihood using telemetry data.

January 2025
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collection import OpenF1DataCollector
from src.feature_engineering import F1FeatureEngineer
from src.eda import F1EDA
from src.model_training import F1CrashPredictor
from src.utils import create_directories, print_data_summary


def print_header():
    """Print pipeline header."""
    print("\n" + "="*70)
    print("F1 CRASH PREDICTOR - ML PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70 + "\n")


def print_footer(start_time: float):
    """Print pipeline footer with execution time."""
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "="*70)
    print(f"Pipeline Complete")
    print(f"Execution Time: {minutes}m {seconds}s")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def run_complete_pipeline(
    year: int = 2024,
    max_sessions: int = 5,
    run_eda: bool = True,
    tune_hyperparameters: bool = True,
    balance_method: str = 'smote'
):
    """
    Run the complete F1 crash prediction pipeline.
    
    Args:
        year: Year to collect F1 data from
        max_sessions: Maximum number of sessions to collect
        run_eda: Whether to run exploratory data analysis
        tune_hyperparameters: Whether to perform hyperparameter tuning
        balance_method: Method for handling class imbalance ('smote', 'class_weight', None)
    """
    start_time = time.time()
    
    print_header()
    
    # Step 1: Setup
    print_section("STEP 1: Environment Setup")
    create_directories()
    
    # Step 2: Data Collection
    print_section("STEP 2: Data Collection from OpenF1 API")
    collector = OpenF1DataCollector()
    df_raw = collector.collect_comprehensive_data(year=year, max_sessions=max_sessions)
    
    # Save raw data
    raw_data_path = 'data/raw/f1_telemetry_raw.csv'
    df_raw.to_csv(raw_data_path, index=False)
    print(f"\nRaw data saved: {raw_data_path}")
    print_data_summary(df_raw, "Raw Data Summary")
    
    # Step 3: Feature Engineering
    print_section("STEP 3: Feature Engineering (12+ Features)")
    engineer = F1FeatureEngineer()
    df_features = engineer.create_all_features(df_raw)
    df_labeled = engineer.label_crashes(df_features, method='synthetic')
    
    # Save processed data
    processed_data_path = 'data/processed/f1_telemetry_features.csv'
    df_labeled.to_csv(processed_data_path, index=False)
    print(f"\nProcessed data saved: {processed_data_path}")
    print_data_summary(df_labeled, "Processed Data Summary")
    
    # Get feature matrix
    X, y = engineer.get_feature_matrix(df_labeled)
    
    # Step 4: Exploratory Data Analysis
    if run_eda:
        print_section("STEP 4: Exploratory Data Analysis & Visualization")
        eda = F1EDA(df_labeled)
        eda.run_complete_eda()
    
    # Step 5: Model Training & Evaluation
    print_section("STEP 5: Model Training & Evaluation")
    print("Models: Logistic Regression, Random Forest, XGBoost")
    print(f"Hyperparameter Tuning: {'Enabled' if tune_hyperparameters else 'Disabled'}")
    print(f"Class Balancing: {balance_method if balance_method else 'None'}\n")
    
    predictor = F1CrashPredictor()
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(
        X, y, test_size=0.2, balance_method=balance_method
    )
    
    # Train all models
    predictor.train_all_models(X_train, y_train, X_test, y_test, tune_hyperparameters)
    
    # Step 6: Results & Visualization
    print_section("STEP 6: Results Visualization & Model Persistence")
    predictor.plot_results(X_test, y_test)
    predictor.save_models()
    predictor.generate_report()
    
    # Step 7: Final Summary
    print_section("STEP 7: Final Results Summary")
    print("\nMODEL PERFORMANCE SUMMARY")
    print("-" * 70)
    print(f"{'Model':<20} {'F1-Score':<12} {'ROC-AUC':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 70)
    
    for model_name, results in predictor.results.items():
        metrics = results['metrics']
        print(f"{model_name:<20} {metrics['f1_score']:<12.4f} {metrics['roc_auc']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")
    
    print("-" * 70)
    print(f"\nBEST MODEL: {predictor.best_model_name}")
    best_metrics = predictor.results[predictor.best_model_name]['metrics']
    print(f"   F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:  {best_metrics['roc_auc']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall:    {best_metrics['recall']:.4f}")
    
    print("\nOUTPUT FILES:")
    print("   - Data: data/raw/, data/processed/")
    print("   - Visualizations: visualizations/")
    print("   - Models: models/")
    print("   - Report: model_performance_report.txt")
    
    print_footer(start_time)
    
    return predictor, df_labeled


def quick_prediction_demo(predictor: F1CrashPredictor):
    """Demo function showing model usage for predictions."""
    import pandas as pd
    import numpy as np
    
    print("\n" + "="*70)
    print("PREDICTION DEMO")
    print("="*70 + "\n")
    
    # Create sample telemetry data
    sample_data = pd.DataFrame({
        'lap_time_delta_personal': [2.5],
        'lap_time_delta_session': [1.8],
        'tire_degradation_rate': [0.15],
        'speed_variance': [25.0],
        'position_change': [2.0],
        'position_change_rate': [0.5],
        'sector_consistency': [1.2],
        'weather_impact': [3.5],
        'tire_age_normalized': [0.65],
        'driver_risk_score': [1.8],
        'track_difficulty_normalized': [0.72],
        'lap_progression': [0.68],
        'speed_position_ratio': [28.5],
        'temp_deviation': [8.2],
        'is_raining': [0],
        'sector_performance_var': [1.5]
    })
    
    # Scale the data
    sample_scaled = predictor.scaler.transform(sample_data)
    sample_df = pd.DataFrame(sample_scaled, columns=sample_data.columns)
    
    # Make prediction
    prediction = predictor.best_model.predict(sample_df)[0]
    probability = predictor.best_model.predict_proba(sample_df)[0]
    
    print("Sample Telemetry Data:")
    print(sample_data.T)
    print(f"\nPrediction: {'CRASH RISK' if prediction == 1 else 'SAFE'}")
    print(f"Crash Probability: {probability[1]:.2%}")
    print(f"Safe Probability: {probability[0]:.2%}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 Crash Predictor ML Pipeline')
    parser.add_argument('--year', type=int, default=2024, help='F1 season year')
    parser.add_argument('--sessions', type=int, default=5, help='Max sessions to collect')
    parser.add_argument('--no-eda', action='store_true', help='Skip EDA')
    parser.add_argument('--quick', action='store_true', help='Quick run (no hyperparameter tuning)')
    parser.add_argument('--balance', type=str, default='smote', 
                       choices=['smote', 'class_weight', 'none'],
                       help='Class balancing method')
    parser.add_argument('--demo', action='store_true', help='Run prediction demo')
    
    args = parser.parse_args()
    
    balance_method = None if args.balance == 'none' else args.balance
    
    # Run pipeline
    predictor, df = run_complete_pipeline(
        year=args.year,
        max_sessions=args.sessions,
        run_eda=not args.no_eda,
        tune_hyperparameters=not args.quick,
        balance_method=balance_method
    )
    
    # Run demo if requested
    if args.demo:
        quick_prediction_demo(predictor)
    
    print("\nReady for deployment. Saved models available in 'models/' directory.")
    print("Tip: Run 'python main.py --demo' to see a prediction example.\n")

