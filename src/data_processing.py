import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

class RFMFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate Recency, Frequency, Monetary features for each customer"""
    
    def __init__(self, date_col='TransactionStartTime', amount_col='Value', 
                 customer_col='AccountId', time_window=90):
        self.date_col = date_col
        self.amount_col = amount_col
        self.customer_col = customer_col
        self.time_window = time_window
        self.reference_date = None
        
    def fit(self, X, y=None):
        # Set reference date as the most recent transaction in the data
        self.reference_date = pd.to_datetime(X[self.date_col]).max()
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        
        # Calculate RFM features
        rfm = X.groupby(self.customer_col).agg({
            self.date_col: lambda x: (self.reference_date - x.max()).days,
            self.amount_col: ['count', 'sum', 'mean', 'std']
        })
        
        # Flatten multi-index columns
        rfm.columns = ['_'.join(col).strip() for col in rfm.columns.values]
        rfm.rename(columns={
            f'{self.date_col}_<lambda>': 'recency',
            f'{self.amount_col}_count': 'frequency',
            f'{self.amount_col}_sum': 'monetary_total',
            f'{self.amount_col}_mean': 'monetary_mean',
            f'{self.amount_col}_std': 'monetary_std'
        }, inplace=True)
        
        # Calculate additional features
        rfm['frequency_recency_ratio'] = rfm['frequency'] / (rfm['recency'] + 1)
        rfm['monetary_consistency'] = 1 / (rfm['monetary_std'] + 1e-6)
        
        # Reset index to merge back
        rfm.reset_index(inplace=True)
        
        return rfm

class TargetVariableGenerator(BaseEstimator, TransformerMixin):
    """Create proxy target variable for credit risk"""
    
    def __init__(self, monetary_threshold=0.9, frequency_threshold=0.75, 
                 recency_threshold=30, fraud_weight=2.0):
        self.monetary_threshold = monetary_threshold
        self.frequency_threshold = frequency_threshold
        self.recency_threshold = recency_threshold
        self.fraud_weight = fraud_weight
        self.monetary_cutoff = None
        self.frequency_cutoff = None
        
    def fit(self, X, y=None):
        # Determine thresholds based on percentiles of the data
        self.monetary_cutoff = np.percentile(
            X['monetary_total'], self.monetary_threshold * 100)
        self.frequency_cutoff = np.percentile(
            X['frequency'], self.frequency_threshold * 100)
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Calculate risk score components
        monetary_risk = (X['monetary_total'] > self.monetary_cutoff).astype(int)
        frequency_risk = (X['frequency'] > self.frequency_cutoff).astype(int)
        recency_risk = (X['recency'] > self.recency_threshold).astype(int)
        
        # Combine with fraud data if available
        if 'fraud_count' in X.columns:
            fraud_risk = (X['fraud_count'] > 0).astype(int) * self.fraud_weight
        else:
            fraud_risk = 0
            
        # Calculate composite risk score (higher = more risky)
        X['risk_score'] = (monetary_risk + frequency_risk + recency_risk + fraud_risk)
        
        # Create binary target (1 = risky, 0 = not risky)
        X['default_proxy'] = (X['risk_score'] >= 2).astype(int)
        
        return X

def load_data(filepath):
    """Load raw transaction data"""
    df = pd.read_csv(filepath, parse_dates=['TransactionStartTime'])
    return df

def preprocess_transactions(df):
    """Initial data cleaning and feature extraction"""
    
    # Convert to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Extract time-based features
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_dow'] = df['TransactionStartTime'].dt.dayofweek
    
    # Create fraud indicator
    df['is_fraud'] = df['FraudResult'].fillna(0)
    
    return df

def build_feature_pipeline():
    """Create pipeline for feature processing"""
    
    # Numeric features
    numeric_features = [
        'recency', 'frequency', 'monetary_total', 
        'monetary_mean', 'monetary_std',
        'frequency_recency_ratio', 'monetary_consistency'
    ]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features (if we had any)
    categorical_features = []
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def process_data(input_path, output_path=None):
    """Full data processing pipeline"""
    
    # Load and preprocess raw data
    df = load_data(input_path)
    df = preprocess_transactions(df)
    
    # Generate RFM features
    rfm_generator = RFMFeatureGenerator()
    rfm_features = rfm_generator.fit_transform(df)
    
    # Add fraud count per customer
    fraud_counts = df.groupby('AccountId')['is_fraud'].sum().reset_index()
    fraud_counts.columns = ['AccountId', 'fraud_count']
    rfm_features = rfm_features.merge(fraud_counts, on='AccountId', how='left')
    
    # Generate target variable
    target_generator = TargetVariableGenerator()
    processed_data = target_generator.fit_transform(rfm_features)
    
    # Save processed data
    if output_path:
        processed_data.to_csv(output_path, index=False)
    
    return processed_data

if __name__ == "__main__":
    # Example usage
    raw_data_path = "./data/raw/transactions.csv"
    processed_data_path = "./data/processed/processed_data.csv"
    process_data(raw_data_path, processed_data_path)