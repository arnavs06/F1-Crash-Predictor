"""
Exploratory Data Analysis (EDA) module for F1 Crash Predictor.
Comprehensive visualizations using matplotlib and seaborn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os


class F1EDA:
    """Exploratory Data Analysis for F1 crash prediction."""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = 'visualizations'):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def run_complete_eda(self):
        """Run complete EDA pipeline with all visualizations."""
        print("\n" + "="*60)
        print("Exploratory Data Analysis")
        print("="*60 + "\n")
        
        self.basic_statistics()
        self.plot_crash_distribution()
        self.plot_feature_distributions()
        self.plot_correlation_matrix()
        self.plot_feature_importance_correlation()
        self.plot_crash_by_conditions()
        self.plot_lap_analysis()
        self.plot_driver_analysis()
        
        print(f"\nAll visualizations saved to '{self.output_dir}/'")
    
    def basic_statistics(self):
        """Print basic statistical summary."""
        print("Basic Statistics:")
        print("-" * 60)
        print(f"Total Records: {len(self.df)}")
        print(f"Number of Features: {len(self.df.columns)}")
        
        if 'crash' in self.df.columns:
            crash_count = self.df['crash'].sum()
            crash_rate = self.df['crash'].mean()
            print(f"Crash Instances: {crash_count} ({crash_rate:.2%})")
            print(f"Non-Crash Instances: {len(self.df) - crash_count} ({1-crash_rate:.2%})")
        
        if 'session_key' in self.df.columns:
            print(f"Number of Sessions: {self.df['session_key'].nunique()}")
        
        if 'driver_number' in self.df.columns:
            print(f"Number of Drivers: {self.df['driver_number'].nunique()}")
        
        print("\nNumeric Features Summary:")
        print(self.df.select_dtypes(include=[np.number]).describe())
        print("-" * 60 + "\n")
    
    def plot_crash_distribution(self):
        """Plot crash vs non-crash distribution."""
        if 'crash' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        crash_counts = self.df['crash'].value_counts()
        axes[0].bar(['No Crash', 'Crash'], crash_counts.values, color=['green', 'red'], alpha=0.7)
        axes[0].set_ylabel('Count')
        axes[0].set_title('Crash Distribution (Count)')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add count labels
        for i, v in enumerate(crash_counts.values):
            axes[0].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        axes[1].pie(crash_counts.values, labels=['No Crash', 'Crash'], 
                   autopct='%1.1f%%', colors=['green', 'red'])
        axes[1].set_title('Crash Distribution (Percentage)')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_crash_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: crash_distribution.png")
    
    def plot_feature_distributions(self):
        """Plot distributions of key features."""
        # Select numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and non-feature columns
        exclude_cols = ['crash', 'session_key', 'driver_number', 'lap_number', 'position']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols][:12]  # First 12 features
        
        if not feature_cols:
            return
        
        n_features = len(feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, col in enumerate(feature_cols):
            if idx < len(axes):
                axes[idx].hist(self.df[col].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'{col}')
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(feature_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_distributions.png")
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of features."""
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude non-feature columns
        exclude_cols = ['session_key', 'driver_number', 'lap_number']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols][:15]  # Top 15
        
        if len(feature_cols) < 2:
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[feature_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: correlation_matrix.png")
    
    def plot_feature_importance_correlation(self):
        """Plot correlation of features with crash target."""
        if 'crash' not in self.df.columns:
            return
        
        # Select numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['crash', 'session_key', 'driver_number', 'lap_number']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not feature_cols:
            return
        
        # Calculate correlation with crash
        correlations = {}
        for col in feature_cols:
            if self.df[col].std() > 0:  # Only if there's variance
                corr, _ = stats.pearsonr(self.df[col].fillna(0), self.df['crash'])
                correlations[col] = corr
        
        # Sort by absolute correlation
        sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:15])
        
        # Plot
        plt.figure(figsize=(12, 8))
        colors = ['red' if v < 0 else 'green' for v in sorted_corr.values()]
        plt.barh(list(sorted_corr.keys()), list(sorted_corr.values()), color=colors, alpha=0.7)
        plt.xlabel('Correlation with Crash')
        plt.title('Feature Correlation with Crash Target', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_feature_crash_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_crash_correlation.png")
    
    def plot_crash_by_conditions(self):
        """Plot crash rates by different conditions."""
        if 'crash' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Crash by rainfall
        if 'is_raining' in self.df.columns or 'weather_rainfall' in self.df.columns:
            rain_col = 'is_raining' if 'is_raining' in self.df.columns else 'weather_rainfall'
            crash_by_rain = self.df.groupby(rain_col)['crash'].mean() * 100
            axes[0, 0].bar(crash_by_rain.index, crash_by_rain.values, color=['blue', 'darkblue'], alpha=0.7)
            axes[0, 0].set_xlabel('Rainfall (0=No, 1=Yes)')
            axes[0, 0].set_ylabel('Crash Rate (%)')
            axes[0, 0].set_title('Crash Rate by Rainfall')
            axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Crash by tire age
        if 'tire_age' in self.df.columns:
            tire_bins = pd.cut(self.df['tire_age'], bins=5)
            crash_by_tire = self.df.groupby(tire_bins)['crash'].mean() * 100
            axes[0, 1].bar(range(len(crash_by_tire)), crash_by_tire.values, color='orange', alpha=0.7)
            axes[0, 1].set_xlabel('Tire Age Group')
            axes[0, 1].set_ylabel('Crash Rate (%)')
            axes[0, 1].set_title('Crash Rate by Tire Age')
            axes[0, 1].set_xticklabels([f'G{i+1}' for i in range(len(crash_by_tire))])
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Crash by lap progression
        if 'lap_progression' in self.df.columns:
            progression_bins = pd.cut(self.df['lap_progression'], bins=5)
            crash_by_prog = self.df.groupby(progression_bins)['crash'].mean() * 100
            axes[1, 0].bar(range(len(crash_by_prog)), crash_by_prog.values, color='purple', alpha=0.7)
            axes[1, 0].set_xlabel('Race Progression')
            axes[1, 0].set_ylabel('Crash Rate (%)')
            axes[1, 0].set_title('Crash Rate by Race Progression')
            axes[1, 0].set_xticklabels(['Start', 'Early', 'Mid', 'Late', 'End'], rotation=45)
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Crash by track temperature
        if 'weather_track_temp' in self.df.columns:
            temp_bins = pd.cut(self.df['weather_track_temp'], bins=5)
            crash_by_temp = self.df.groupby(temp_bins)['crash'].mean() * 100
            axes[1, 1].bar(range(len(crash_by_temp)), crash_by_temp.values, color='red', alpha=0.7)
            axes[1, 1].set_xlabel('Track Temperature Group')
            axes[1, 1].set_ylabel('Crash Rate (%)')
            axes[1, 1].set_title('Crash Rate by Track Temperature')
            axes[1, 1].set_xticklabels([f'T{i+1}' for i in range(len(crash_by_temp))])
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_crash_by_conditions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: crash_by_conditions.png")
    
    def plot_lap_analysis(self):
        """Analyze lap times and performance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Lap time distribution by crash
        if 'lap_duration' in self.df.columns and 'crash' in self.df.columns:
            self.df.boxplot(column='lap_duration', by='crash', ax=axes[0, 0])
            axes[0, 0].set_xlabel('Crash (0=No, 1=Yes)')
            axes[0, 0].set_ylabel('Lap Duration (s)')
            axes[0, 0].set_title('Lap Duration by Crash')
            plt.sca(axes[0, 0])
            plt.xticks([1, 2], ['No Crash', 'Crash'])
        
        # 2. Speed distribution by crash
        if 'speed' in self.df.columns and 'crash' in self.df.columns:
            no_crash = self.df[self.df['crash'] == 0]['speed'].dropna()
            crash = self.df[self.df['crash'] == 1]['speed'].dropna()
            axes[0, 1].hist([no_crash, crash], bins=30, label=['No Crash', 'Crash'], 
                          color=['green', 'red'], alpha=0.6)
            axes[0, 1].set_xlabel('Speed (km/h)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Speed Distribution by Crash')
            axes[0, 1].legend()
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Position change distribution
        if 'position_change' in self.df.columns:
            axes[1, 0].hist(self.df['position_change'].dropna(), bins=30, color='skyblue', 
                          edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('Position Change')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Position Change Distribution')
            axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Lap time delta distribution
        if 'lap_time_delta_personal' in self.df.columns:
            axes[1, 1].hist(self.df['lap_time_delta_personal'].dropna(), bins=30, 
                          color='orange', edgecolor='black', alpha=0.7)
            axes[1, 1].set_xlabel('Lap Time Delta (s)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Lap Time Delta Distribution')
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_lap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: lap_analysis.png")
    
    def plot_driver_analysis(self):
        """Analyze driver performance and risk."""
        if 'driver_number' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Crash count by driver (top 10)
        if 'crash' in self.df.columns:
            crash_by_driver = self.df.groupby('driver_number')['crash'].sum().sort_values(ascending=False).head(10)
            axes[0, 0].bar(crash_by_driver.index.astype(str), crash_by_driver.values, 
                         color='red', alpha=0.7)
            axes[0, 0].set_xlabel('Driver Number')
            axes[0, 0].set_ylabel('Total Crashes')
            axes[0, 0].set_title('Top 10 Drivers by Crash Count')
            axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Average lap time by driver (top 10)
        if 'lap_duration' in self.df.columns:
            avg_lap_by_driver = self.df.groupby('driver_number')['lap_duration'].mean().sort_values().head(10)
            axes[0, 1].barh(avg_lap_by_driver.index.astype(str), avg_lap_by_driver.values, 
                          color='blue', alpha=0.7)
            axes[0, 1].set_ylabel('Driver Number')
            axes[0, 1].set_xlabel('Average Lap Time (s)')
            axes[0, 1].set_title('Top 10 Fastest Drivers (Avg Lap Time)')
            axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Driver risk score distribution
        if 'driver_risk_score' in self.df.columns:
            driver_risk = self.df.groupby('driver_number')['driver_risk_score'].mean().sort_values(ascending=False).head(15)
            axes[1, 0].bar(driver_risk.index.astype(str), driver_risk.values, 
                         color='orange', alpha=0.7)
            axes[1, 0].set_xlabel('Driver Number')
            axes[1, 0].set_ylabel('Risk Score')
            axes[1, 0].set_title('Top 15 Drivers by Risk Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Laps completed by driver
        laps_by_driver = self.df.groupby('driver_number').size().sort_values(ascending=False).head(15)
        axes[1, 1].bar(laps_by_driver.index.astype(str), laps_by_driver.values, 
                     color='green', alpha=0.7)
        axes[1, 1].set_xlabel('Driver Number')
        axes[1, 1].set_ylabel('Laps Completed')
        axes[1, 1].set_title('Top 15 Drivers by Laps Completed')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_driver_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: driver_analysis.png")


def main():
    """Main EDA pipeline."""
    # Load processed data
    print("Loading processed data...")
    try:
        df = pd.read_csv('data/processed/f1_telemetry_features.csv')
    except FileNotFoundError:
        print("Processed data not found. Please run feature_engineering.py first.")
        return
    
    # Run EDA
    eda = F1EDA(df)
    eda.run_complete_eda()
    
    print("\nEDA complete")


if __name__ == "__main__":
    main()

