#!/bin/bash

# F1 Crash Predictor Setup Script
echo "======================================================================"
echo "  F1 CRASH PREDICTOR - Setup Script"
echo "======================================================================"
echo ""

# Check Python version
echo "Checking Python installation..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install pandas==2.1.4
pip install numpy==1.26.2
pip install scikit-learn==1.3.2
pip install xgboost==2.0.3
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install requests==2.31.0
pip install imbalanced-learn==0.11.0
pip install scipy==1.11.4
pip install tqdm==4.66.1

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✓ All dependencies installed successfully"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p visualizations
mkdir -p models

echo "✓ Project directories created"
echo ""

echo "======================================================================"
echo "  Setup Complete!"
echo "======================================================================"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the pipeline: python main.py --quick"
echo "  3. View results in: visualizations/ and models/"
echo ""
echo "For more options: python main.py --help"
echo ""

