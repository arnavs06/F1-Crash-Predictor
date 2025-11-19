"""
Model Training module for F1 Crash Predictor.
Implements Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, f1_score, precision_score, recall_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from typing import Dict, Tuple, Any


class F1CrashPredictor:
    """Train and evaluate F1 crash prediction models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                    balance_method: str = None) -> Tuple:
        """
        Prepare data for training with optional balancing.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Test set size
            balance_method: 'smote', 'class_weight', or None
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n" + "="*60)
        print("Data Preparation")
        print("="*60 + "\n")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training class distribution: {dict(y_train.value_counts())}")
        print(f"Test class distribution: {dict(y_test.value_counts())}")
        
        # Handle class imbalance
        if balance_method == 'smote':
            from imblearn.over_sampling import SMOTE
            print("\nApplying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {dict(pd.Series(y_train).value_counts())}")
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        print("Data preparation complete")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                  tune_hyperparameters: bool = True) -> LogisticRegression:
        """Train Logistic Regression model with optional hyperparameter tuning."""
        print("\n" + "-"*60)
        print("Training Logistic Regression")
        print("-"*60)
        
        if tune_hyperparameters:
            print("Performing grid search for hyperparameter tuning...")
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None]
            }
            
            lr = LogisticRegression(random_state=42, max_iter=1000)
            grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
        else:
            model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            model.fit(X_train, y_train)
        
        self.models['Logistic Regression'] = model
        print("Logistic Regression trained")
        
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           tune_hyperparameters: bool = True) -> RandomForestClassifier:
        """Train Random Forest model with optional hyperparameter tuning."""
        print("\n" + "-"*60)
        print("Training Random Forest")
        print("-"*60)
        
        if tune_hyperparameters:
            print("Performing grid search for hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
        else:
            model = RandomForestClassifier(n_estimators=200, random_state=42, 
                                         class_weight='balanced', n_jobs=-1)
            model.fit(X_train, y_train)
        
        self.models['Random Forest'] = model
        print("Random Forest trained")
        
        return model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                     tune_hyperparameters: bool = True) -> XGBClassifier:
        """Train XGBoost model with optional hyperparameter tuning."""
        print("\n" + "-"*60)
        print("Training XGBoost")
        print("-"*60)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        if tune_hyperparameters:
            print("Performing grid search for hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'scale_pos_weight': [scale_pos_weight]
            }
            
            xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
            grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
        else:
            model = XGBClassifier(n_estimators=200, random_state=42, 
                                scale_pos_weight=scale_pos_weight, n_jobs=-1, eval_metric='logloss')
            model.fit(X_train, y_train)
        
        self.models['XGBoost'] = model
        print("XGBoost trained")
        
        return model
    
    def evaluate_model(self, model: Any, model_name: str, 
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate a trained model."""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series, 
                        tune_hyperparameters: bool = True):
        """Train and evaluate all models."""
        print("\n" + "="*60)
        print("Training All Models")
        print("="*60)
        
        # Train models
        self.train_logistic_regression(X_train, y_train, tune_hyperparameters)
        self.train_random_forest(X_train, y_train, tune_hyperparameters)
        self.train_xgboost(X_train, y_train, tune_hyperparameters)
        
        # Evaluate all models
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model, model_name, X_test, y_test)
        
        # Identify best model
        best_f1 = 0
        for model_name, results in self.results.items():
            if results['metrics']['f1_score'] > best_f1:
                best_f1 = results['metrics']['f1_score']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        print(f"\nBest Model: {self.best_model_name} (F1-Score: {best_f1:.4f})")
    
    def plot_results(self, X_test: pd.DataFrame, y_test: pd.Series, 
                    output_dir: str = 'visualizations'):
        """Create comprehensive evaluation plots."""
        print("\n" + "="*60)
        print("Creating Evaluation Visualizations")
        print("="*60 + "\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Model comparison
        self._plot_model_comparison(output_dir)
        
        # 2. Confusion matrices
        self._plot_confusion_matrices(y_test, output_dir)
        
        # 3. ROC curves
        self._plot_roc_curves(y_test, output_dir)
        
        # 4. Feature importance
        self._plot_feature_importance(X_test, output_dir)
        
        print(f"All evaluation plots saved to '{output_dir}/'")
    
    def _plot_model_comparison(self, output_dir: str):
        """Plot comparison of model metrics."""
        metrics_df = pd.DataFrame({
            model_name: results['metrics'] 
            for model_name, results in self.results.items()
        }).T
        
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_df.plot(kind='bar', ax=ax, rot=0, alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_xlabel('Model')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/08_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: model_comparison.png")
    
    def _plot_confusion_matrices(self, y_test: pd.Series, output_dir: str):
        """Plot confusion matrices for all models."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_xticklabels(['No Crash', 'Crash'])
            axes[idx].set_yticklabels(['No Crash', 'Crash'])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/09_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: confusion_matrices.png")
    
    def _plot_roc_curves(self, y_test: pd.Series, output_dir: str):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
            auc = results['metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/10_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: roc_curves.png")
    
    def _plot_feature_importance(self, X_test: pd.DataFrame, output_dir: str):
        """Plot feature importance for tree-based models."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Random Forest
        if 'Random Forest' in self.models:
            rf_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': self.models['Random Forest'].feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            axes[0].barh(rf_importance['feature'], rf_importance['importance'], color='green', alpha=0.7)
            axes[0].set_xlabel('Importance')
            axes[0].set_title('Random Forest - Feature Importance', fontweight='bold')
            axes[0].invert_yaxis()
            axes[0].grid(axis='x', alpha=0.3)
        
        # XGBoost
        if 'XGBoost' in self.models:
            xgb_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': self.models['XGBoost'].feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            axes[1].barh(xgb_importance['feature'], xgb_importance['importance'], color='blue', alpha=0.7)
            axes[1].set_xlabel('Importance')
            axes[1].set_title('XGBoost - Feature Importance', fontweight='bold')
            axes[1].invert_yaxis()
            axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/11_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_importance.png")
    
    def save_models(self, output_dir: str = 'models'):
        """Save all trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = model_name.lower().replace(' ', '_')
            filepath = f'{output_dir}/{filename}.pkl'
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Saved: {filepath}")
        
        # Save scaler
        with open(f'{output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved: {output_dir}/scaler.pkl")
    
    def generate_report(self, output_dir: str = '.'):
        """Generate a comprehensive results report."""
        report_path = f'{output_dir}/model_performance_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("F1 CRASH PREDICTOR - MODEL PERFORMANCE REPORT\n")
            f.write("="*60 + "\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"\n{model_name}\n")
                f.write("-"*60 + "\n")
                metrics = results['metrics']
                f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall:    {metrics['recall']:.4f}\n")
                f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
                f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
                f.write("\nConfusion Matrix:\n")
                f.write(str(results['confusion_matrix']) + "\n")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"BEST MODEL: {self.best_model_name}\n")
            f.write(f"F1-Score: {self.results[self.best_model_name]['metrics']['f1_score']:.4f}\n")
            f.write(f"ROC-AUC: {self.results[self.best_model_name]['metrics']['roc_auc']:.4f}\n")
            f.write(f"{'='*60}\n")
        
        print(f"\nReport saved to {report_path}")


def main():
    """Main model training pipeline."""
    # Load processed data
    print("Loading processed data...")
    try:
        df = pd.read_csv('data/processed/f1_telemetry_features.csv')
    except FileNotFoundError:
        print("Processed data not found. Please run feature_engineering.py first.")
        return
    
    # Load feature names
    from src.feature_engineering import F1FeatureEngineer
    engineer = F1FeatureEngineer()
    
    # Get features
    feature_cols = [col for col in df.columns if col in engineer.feature_names or 
                   col.startswith(('lap_time', 'tire_', 'speed_', 'position_', 'sector_', 
                                  'weather_', 'driver_', 'track_', 'temp_', 'is_'))]
    
    # Ensure we have the basic features
    if not feature_cols:
        feature_cols = [col for col in df.columns if col not in 
                       ['crash', 'session_key', 'driver_number', 'circuit', 'country', 'session_name']]
    
    X = df[feature_cols]
    y = df['crash']
    
    # Initialize predictor
    predictor = F1CrashPredictor()
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(
        X, y, test_size=0.2, balance_method='smote'
    )
    
    # Train all models (with hyperparameter tuning)
    predictor.train_all_models(X_train, y_train, X_test, y_test, tune_hyperparameters=True)
    
    # Create visualizations
    predictor.plot_results(X_test, y_test)
    
    # Save models
    predictor.save_models()
    
    # Generate report
    predictor.generate_report()
    
    print("\n" + "="*60)
    print("Model Training Complete")
    print("="*60)


if __name__ == "__main__":
    main()

