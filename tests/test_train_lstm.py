import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path to allow script imports
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from scripts.train_lstm_black_swan import (
    calculate_financial_metrics,
    find_optimal_threshold,
    _calculate_sharpe,
    _calculate_sortino,
    SymbolDataGenerator
)

# Mock CONFIG for testing
CONFIG = {
    "data": {
        "seq_len": 10,
    },
    "training": {
        "batch_size": 8,
    },
    "evaluation": {
        "trading_days_per_year": 252,
        "intervals_per_day": 288,
    }

}

@pytest.fixture
def sample_financial_data():
    """Creates a sample DataFrame for financial metric tests."""
    data = {
        'ret': np.random.randn(100) * 0.01,
        'side': np.random.choice([-1, 1], 100),
        'signal': np.random.choice([-1, 0, 1], 100)
    }
    return pd.DataFrame(data)

def test_calculate_financial_metrics(sample_financial_data):
    """Tests that financial metrics are calculated without error."""
    metrics = calculate_financial_metrics(sample_financial_data, CONFIG)
    assert 'annualized_sharpe_ratio' in metrics
    assert 'annualized_sortino_ratio' in metrics
    assert isinstance(metrics['annualized_sharpe_ratio'], float)
    assert isinstance(metrics['annualized_sortino_ratio'], float)

@pytest.fixture
def sample_threshold_data():
    """Creates sample data for threshold optimization tests."""
    y_true = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100)
    returns = np.random.randn(100) * 0.01
    sides = np.random.choice([-1, 1], 100)
    return y_true, y_pred_proba, returns, sides

def test_find_optimal_threshold_f1(sample_threshold_data):
    """Tests threshold optimization for F1-score."""
    y_true, y_pred_proba, returns, sides = sample_threshold_data
    threshold = find_optimal_threshold(y_true, y_pred_proba, returns, sides, CONFIG, metric='f1')
    assert 0.05 <= threshold <= 0.95

def test_find_optimal_threshold_sharpe(sample_threshold_data):
    """Tests threshold optimization for Sharpe ratio."""
    y_true, y_pred_proba, returns, sides = sample_threshold_data
    threshold = find_optimal_threshold(y_true, y_pred_proba, returns, sides, CONFIG, metric='sharpe')
    assert 0.05 <= threshold <= 0.95

def test_find_optimal_threshold_sortino(sample_threshold_data):
    """Tests threshold optimization for Sortino ratio."""
    y_true, y_pred_proba, returns, sides = sample_threshold_data
    threshold = find_optimal_threshold(y_true, y_pred_proba, returns, sides, CONFIG, metric='sortino')

@pytest.fixture
def sample_generator_data():
    """Creates sample data for the SymbolDataGenerator test."""
    num_features = 5
    num_samples = 30
    # Create a DataFrame for a single symbol
    symbol_df = pd.DataFrame({
        'label': np.random.randint(0, 2, num_samples),
        'trgt': np.random.rand(num_samples),
        'ret': np.random.randn(num_samples) * 0.01,
        'side': np.random.choice([-1, 1], num_samples)
    })
    # Create a corresponding processed features DataFrame
    processed_features = pd.DataFrame(np.random.rand(num_samples, num_features), index=symbol_df.index)
    return symbol_df, processed_features

def test_symbol_data_generator(sample_generator_data):
    """Tests the SymbolDataGenerator for batch shape and correct handling of the final batch."""
    symbol_df, processed_features = sample_generator_data
    generator = SymbolDataGenerator(symbol_df, processed_features, CONFIG)

    # Total samples = 30, seq_len = 10 -> 21 sequences
    # Batch size = 8 -> 3 batches (8, 8, 5)
    assert len(generator) == 3

    # Check first batch
    X1, y1, w1 = generator[0]
    assert X1.shape == (8, CONFIG["data"]["seq_len"], processed_features.shape[1])
    assert y1.shape == (8,)
    assert w1.shape == (8,)

    # Check last batch
    X3, y3, w3 = generator[2]
    assert X3.shape == (5, CONFIG["data"]["seq_len"], processed_features.shape[1])
    assert y3.shape == (5,)
    assert w3.shape == (5,)

