import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    RFMFeatureGenerator, 
    TargetVariableGenerator,
    process_data
)

@pytest.fixture
def sample_transactions():
    """Generate sample transaction data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'TransactionId': range(100),
        'AccountId': np.random.choice(['A1', 'A2', 'A3', 'A4'], size=100),
        'Value': np.random.uniform(10, 500, size=100),
        'TransactionStartTime': dates,
        'FraudResult': np.random.choice([0, 1], size=100, p=[0.95, 0.05])
    })

def test_rfm_generator(sample_transactions):
    """Test RFM feature generation"""
    generator = RFMFeatureGenerator()
    result = generator.fit_transform(sample_transactions)
    
    assert not result.empty
    assert all(col in result.columns for col in [
        'AccountId', 'recency', 'frequency', 'monetary_total',
        'monetary_mean', 'monetary_std'
    ])
    assert len(result['AccountId'].unique()) <= 4  # We have 4 unique accounts

def test_target_generator(sample_transactions):
    """Test target variable generation"""
    rfm_generator = RFMFeatureGenerator()
    rfm_data = rfm_generator.fit_transform(sample_transactions)
    
    fraud_counts = sample_transactions.groupby('AccountId')['FraudResult'].sum().reset_index()
    fraud_counts.columns = ['AccountId', 'fraud_count']
    rfm_data = rfm_data.merge(fraud_counts, on='AccountId', how='left')
    
    target_gen = TargetVariableGenerator()
    result = target_gen.fit_transform(rfm_data)
    
    assert 'default_proxy' in result.columns
    assert 'risk_score' in result.columns
    assert set(result['default_proxy'].unique()).issubset({0, 1})

def test_full_processing(tmp_path, sample_transactions):
    """Test end-to-end data processing"""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    
    sample_transactions.to_csv(input_path, index=False)
    result = process_data(input_path, output_path)
    
    assert output_path.exists()
    assert not result.empty
    assert 'default_proxy' in result.columns