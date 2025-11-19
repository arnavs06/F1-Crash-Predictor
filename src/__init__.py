"""
F1 Crash Predictor source package.
"""

__version__ = '1.0.0'

from .data_collection import OpenF1DataCollector
from .feature_engineering import F1FeatureEngineer
from .eda import F1EDA
from .model_training import F1CrashPredictor
from .utils import (
    create_directories,
    print_data_summary,
    balance_classes,
    identify_crash_laps
)

__all__ = [
    'OpenF1DataCollector',
    'F1FeatureEngineer',
    'F1EDA',
    'F1CrashPredictor',
    'create_directories',
    'print_data_summary',
    'balance_classes',
    'identify_crash_laps'
]

