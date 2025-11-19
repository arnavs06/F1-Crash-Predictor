"""
Utility functions for the F1 Crash Predictor project.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import pickle


def create_directories():
    """Create necessary directories for the project."""
    directories = ['data', 'data/raw', 'data/processed', 'visualizations', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Created project directories")


def save_model(model, filepath: str):
    """Save a trained model to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load a trained model from disk."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def calculate_percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change between two values."""
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100


def identify_crash_laps(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Identify potential crash laps based on telemetry anomalies.
    
    Args:
        df: DataFrame with telemetry data
        threshold: Standard deviation threshold for anomaly detection
    
    Returns:
        DataFrame with crash labels
    """
    df = df.copy()
    
    # Initialize crash label
    df['crash'] = 0
    
    # Detect crashes based on multiple criteria
    if 'lap_time' in df.columns:
        # Abnormally slow laps (potential crash/incident)
        lap_time_mean = df['lap_time'].mean()
        lap_time_std = df['lap_time'].std()
        slow_laps = df['lap_time'] > (lap_time_mean + threshold * lap_time_std)
        df.loc[slow_laps, 'crash'] = 1
    
    # Sudden speed drops
    if 'speed' in df.columns:
        df['speed_change'] = df.groupby('driver_number')['speed'].diff()
        speed_threshold = df['speed_change'].std() * threshold
        sudden_drops = df['speed_change'] < -speed_threshold
        df.loc[sudden_drops, 'crash'] = 1
    
    # Position losses (dropping multiple positions in a lap)
    if 'position' in df.columns:
        df['position_change'] = df.groupby('driver_number')['position'].diff()
        major_losses = df['position_change'] > 3  # Lost 3+ positions
        df.loc[major_losses, 'crash'] = 1
    
    return df


def print_data_summary(df: pd.DataFrame, title: str = "Data Summary"):
    """Print a summary of the DataFrame."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("None")
    print(f"{'='*60}\n")


def balance_classes(X: pd.DataFrame, y: pd.Series, method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance classes using various resampling techniques.
    
    Args:
        X: Feature matrix
        y: Target variable
        method: Resampling method ('smote', 'random_oversample', 'random_undersample')
    
    Returns:
        Balanced X and y
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    
    print(f"Original class distribution: {dict(y.value_counts())}")
    
    if method == 'smote':
        sampler = SMOTE(random_state=42)
    elif method == 'random_oversample':
        sampler = RandomOverSampler(random_state=42)
    elif method == 'random_undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    print(f"Resampled class distribution: {dict(pd.Series(y_resampled).value_counts())}")
    
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

