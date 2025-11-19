"""
Feature Engineering module for F1 Crash Predictor.
Creates 12+ predictive features from telemetry data.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from scipy import stats


class F1FeatureEngineer:
    """Engineer features for F1 crash prediction."""
    
    def __init__(self):
        self.feature_names = []
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all predictive features from raw telemetry data.
        
        Features engineered:
        1. Lap time delta (vs personal best)
        2. Lap time delta (vs session average)
        3. Tire degradation rate
        4. Speed variance
        5. Position change rate
        6. Sector time consistency
        7. Weather impact score
        8. Tire age normalized
        9. Driver risk score (historical)
        10. Track difficulty score
        11. Lap progression factor
        12. Speed to position ratio
        13. Temperature deviation
        14. Rainfall binary indicator
        15. Sector performance variance
        
        Args:
            df: Raw telemetry DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        print("\n" + "="*60)
        print("Feature Engineering Pipeline")
        print("="*60 + "\n")
        
        df = df.copy()
        
        # Ensure required columns exist
        df = self._ensure_columns(df)
        
        print("Creating features...")
        
        # 1. Lap time delta from personal best
        df['lap_time_delta_personal'] = df.groupby('driver_number')['lap_duration'].transform(
            lambda x: x - x.min()
        )
        
        # 2. Lap time delta from session average
        df['lap_time_delta_session'] = df.groupby('session_key')['lap_duration'].transform(
            lambda x: x - x.mean()
        )
        
        # 3. Tire degradation rate (lap time increase per tire age)
        df['tire_degradation_rate'] = df.groupby(['driver_number', 'session_key']).apply(
            lambda x: self._calculate_tire_degradation(x)
        ).reset_index(level=[0, 1], drop=True)
        
        # 4. Speed variance (last 3 laps)
        df['speed_variance'] = df.groupby('driver_number')['speed'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        
        # 5. Position change rate
        df['position_change'] = df.groupby('driver_number')['position'].diff()
        df['position_change_rate'] = df.groupby('driver_number')['position_change'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # 6. Sector time consistency (std of sector times)
        if all(col in df.columns for col in ['duration_sector_1', 'duration_sector_2', 'duration_sector_3']):
            df['sector_consistency'] = df[['duration_sector_1', 'duration_sector_2', 'duration_sector_3']].std(axis=1)
        else:
            df['sector_consistency'] = 0
        
        # 7. Weather impact score
        df['weather_impact'] = self._calculate_weather_impact(df)
        
        # 8. Tire age normalized (0-1 scale)
        df['tire_age_normalized'] = df['tire_age'] / df['tire_age'].max() if 'tire_age' in df.columns else 0
        
        # 9. Driver risk score (based on historical position changes)
        df['driver_risk_score'] = df.groupby('driver_number')['position_change'].transform(
            lambda x: x.abs().mean()
        )
        
        # 10. Track difficulty score (based on average lap times)
        df['track_difficulty'] = df.groupby('circuit')['lap_duration'].transform('mean')
        df['track_difficulty_normalized'] = (df['track_difficulty'] - df['track_difficulty'].min()) / \
                                            (df['track_difficulty'].max() - df['track_difficulty'].min())
        
        # 11. Lap progression factor (early/mid/late race)
        df['lap_progression'] = df['lap_number'] / df.groupby('session_key')['lap_number'].transform('max')
        
        # 12. Speed to position ratio (faster but lower position = potential issue)
        df['speed_position_ratio'] = df['speed'] / (df['position'] + 1)  # +1 to avoid division by zero
        
        # 13. Temperature deviation from ideal
        ideal_track_temp = 30.0  # Celsius
        df['temp_deviation'] = abs(df['weather_track_temp'] - ideal_track_temp)
        
        # 14. Rainfall indicator (binary)
        df['is_raining'] = (df['weather_rainfall'] > 0).astype(int)
        
        # 15. Sector performance variance (normalized)
        if all(col in df.columns for col in ['duration_sector_1', 'duration_sector_2', 'duration_sector_3']):
            sector_means = df[['duration_sector_1', 'duration_sector_2', 'duration_sector_3']].mean()
            df['sector_performance_var'] = df[['duration_sector_1', 'duration_sector_2', 'duration_sector_3']].apply(
                lambda row: ((row - sector_means) ** 2).mean(), axis=1
            )
        else:
            df['sector_performance_var'] = 0
        
        # Fill NaN values
        df = df.fillna(0)
        
        # Store feature names
        self.feature_names = [
            'lap_time_delta_personal', 'lap_time_delta_session', 'tire_degradation_rate',
            'speed_variance', 'position_change', 'position_change_rate', 'sector_consistency',
            'weather_impact', 'tire_age_normalized', 'driver_risk_score', 
            'track_difficulty_normalized', 'lap_progression', 'speed_position_ratio',
            'temp_deviation', 'is_raining', 'sector_performance_var'
        ]
        
        print(f"Created {len(self.feature_names)} features")
        print(f"Feature names: {self.feature_names[:5]}... (showing first 5)")
        
        return df
    
    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist with default values."""
        required_cols = {
            'lap_duration': 85.0,
            'speed': 250.0,
            'position': 10,
            'tire_age': 5,
            'weather_track_temp': 30.0,
            'weather_air_temp': 25.0,
            'weather_rainfall': 0,
            'lap_number': 1,
            'duration_sector_1': 28.0,
            'duration_sector_2': 29.0,
            'duration_sector_3': 28.0,
        }
        
        for col, default_val in required_cols.items():
            if col not in df.columns:
                df[col] = default_val
        
        return df
    
    def _calculate_tire_degradation(self, group_df: pd.DataFrame) -> pd.Series:
        """Calculate tire degradation rate for a driver in a session."""
        if len(group_df) < 2 or 'tire_age' not in group_df.columns:
            return pd.Series(0, index=group_df.index)
        
        # Calculate lap time increase per tire age unit
        if group_df['tire_age'].std() > 0:
            correlation = group_df['lap_duration'].corr(group_df['tire_age'])
            degradation_rate = correlation * group_df['lap_duration'].std()
        else:
            degradation_rate = 0
        
        return pd.Series(degradation_rate, index=group_df.index)
    
    def _calculate_weather_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate weather impact score based on temperature and rainfall."""
        impact = pd.Series(0.0, index=df.index)
        
        # Rainfall has major impact
        if 'weather_rainfall' in df.columns:
            impact += df['weather_rainfall'] * 5.0
        
        # Extreme temperatures
        if 'weather_track_temp' in df.columns:
            # Too cold or too hot
            ideal_temp = 30.0
            impact += abs(df['weather_track_temp'] - ideal_temp) * 0.1
        
        return impact
    
    def label_crashes(self, df: pd.DataFrame, method: str = 'synthetic') -> pd.DataFrame:
        """
        Label crash instances in the data.
        
        Args:
            df: DataFrame with features
            method: 'synthetic' for rule-based labeling, 'manual' for actual crash data
        
        Returns:
            DataFrame with crash labels
        """
        print("\nLabeling crash instances...")
        
        df = df.copy()
        
        if method == 'synthetic':
            # Rule-based crash labeling for synthetic data
            df['crash'] = 0
            
            # Crash indicators:
            # 1. Very slow lap times (DNF, incident)
            slow_threshold = df['lap_duration'].quantile(0.95)
            df.loc[df['lap_duration'] > slow_threshold, 'crash'] = 1
            
            # 2. Extreme position losses
            if 'position_change' in df.columns:
                df.loc[df['position_change'] > 5, 'crash'] = 1
            
            # 3. High speed variance (erratic driving)
            if 'speed_variance' in df.columns:
                variance_threshold = df['speed_variance'].quantile(0.90)
                df.loc[df['speed_variance'] > variance_threshold, 'crash'] = 1
            
            # 4. Extreme tire degradation + rain
            if 'is_raining' in df.columns and 'tire_age_normalized' in df.columns:
                risky_conditions = (df['is_raining'] == 1) & (df['tire_age_normalized'] > 0.7)
                # Randomly label some of these as crashes
                risky_indices = df[risky_conditions].index
                crash_sample = np.random.choice(risky_indices, size=len(risky_indices)//5, replace=False)
                df.loc[crash_sample, 'crash'] = 1
            
            # 5. Lap time delta anomalies
            if 'lap_time_delta_personal' in df.columns:
                anomaly_threshold = df['lap_time_delta_personal'].quantile(0.93)
                df.loc[df['lap_time_delta_personal'] > anomaly_threshold, 'crash'] = 1
            
            # Ensure we have roughly 5% positive class
            current_positive_rate = df['crash'].mean()
            if current_positive_rate < 0.03:
                # Add more crashes randomly from high-risk scenarios
                non_crash_indices = df[df['crash'] == 0].index
                additional_crashes = int(len(df) * 0.05 - df['crash'].sum())
                if additional_crashes > 0:
                    crash_sample = np.random.choice(non_crash_indices, 
                                                   size=min(additional_crashes, len(non_crash_indices)), 
                                                   replace=False)
                    df.loc[crash_sample, 'crash'] = 1
            
            crash_count = df['crash'].sum()
            crash_rate = df['crash'].mean()
            print(f"Labeled {crash_count} crashes ({crash_rate:.2%} of data)")
        
        return df
    
    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract feature matrix (X) and target variable (y).
        
        Args:
            df: DataFrame with features and labels
        
        Returns:
            Tuple of (X, y)
        """
        # Select only engineered features
        X = df[self.feature_names].copy()
        y = df['crash'].copy() if 'crash' in df.columns else None
        
        print(f"\nFeature Matrix Shape: {X.shape}")
        if y is not None:
            print(f"Target Distribution: {dict(y.value_counts())}")
        
        return X, y


def main():
    """Main feature engineering pipeline."""
    # Load raw data
    print("Loading raw data...")
    try:
        df = pd.read_csv('data/raw/f1_telemetry_raw.csv')
    except FileNotFoundError:
        print("Raw data not found. Please run data_collection.py first.")
        return
    
    # Initialize feature engineer
    engineer = F1FeatureEngineer()
    
    # Create features
    df_features = engineer.create_all_features(df)
    
    # Label crashes
    df_labeled = engineer.label_crashes(df_features, method='synthetic')
    
    # Save processed data
    output_path = 'data/processed/f1_telemetry_features.csv'
    df_labeled.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    
    # Get feature matrix
    X, y = engineer.get_feature_matrix(df_labeled)
    
    return df_labeled, X, y


if __name__ == "__main__":
    main()

