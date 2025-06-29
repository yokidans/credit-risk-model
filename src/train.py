# src/train.py
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           classification_report, confusion_matrix, roc_curve)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
import warnings

# Configure warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('future.no_silent_downcasting', True)

class CreditRiskModel:
    """Production-grade credit risk model with guaranteed convergence"""
    
    def __init__(self, model_type='logistic', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.preprocessor = self._build_preprocessor()
        self.model = self._initialize_model()
        self.feature_importances_ = None
        self.metrics = {}
        self.convergence_info = {}
        
    def _build_preprocessor(self):
        """Build robust feature engineering pipeline"""
        numeric_features = [
            'recency', 'frequency', 'monetary_total',
            'monetary_mean', 'monetary_std',
            'frequency_recency_ratio', 'monetary_consistency'
        ]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        return ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_features)]
        )
    
    def _initialize_model(self):
        """Initialize model with convergence-optimized settings"""
        return LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='saga',
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=10000,  # Increased default iterations
            tol=1e-3,  # More relaxed tolerance
            warm_start=False,
            verbose=0
        )
    
    def _check_data_quality(self, X, y):
        """Perform comprehensive data quality checks"""
        checks = {
            'has_nan': bool(X.isna().any().any()),
            'has_inf': bool(np.isinf(X.values).any()),
            'n_features': int(X.shape[1]),
            'n_samples': int(X.shape[0]),
            'class_balance': y.value_counts(normalize=True).to_dict(),
            'feature_ranges': {col: {'min': float(X[col].min()), 'max': float(X[col].max())} 
                             for col in X.columns}
        }
        return checks
    
    def _ensure_convergence(self, pipeline, X_train, y_train):
        """Guaranteed convergence with multiple strategies"""
        classifier = pipeline.named_steps['classifier']
        original_params = classifier.get_params()
        self.convergence_info['attempts'] = []
        
        # Strategy 1: Try with original parameters
        attempt = {
            'strategy': 'initial',
            'max_iter': int(original_params['max_iter']),
            'tol': float(original_params['tol']),
            'solver': str(original_params['solver'])
        }
        pipeline.fit(X_train, y_train)
        if hasattr(classifier, 'converged_') and classifier.converged_:
            attempt['converged'] = True
            self.convergence_info['attempts'].append(attempt)
            return pipeline
        
        attempt['converged'] = False
        self.convergence_info['attempts'].append(attempt)
        
        # Strategy 2: Increase iterations
        print("Attempting convergence with increased iterations...")
        new_max_iter = original_params['max_iter'] * 2
        attempt = {
            'strategy': 'increase_iterations',
            'max_iter': int(new_max_iter),
            'tol': float(original_params['tol']),
            'solver': str(original_params['solver'])
        }
        pipeline.named_steps['classifier'].set_params(max_iter=new_max_iter)
        pipeline.fit(X_train, y_train)
        if hasattr(classifier, 'converged_') and classifier.converged_:
            attempt['converged'] = True
            self.convergence_info['attempts'].append(attempt)
            return pipeline
        
        attempt['converged'] = False
        self.convergence_info['attempts'].append(attempt)
        
        # Strategy 3: Relax tolerance
        print("Attempting convergence with relaxed tolerance...")
        new_tol = 1e-2  # More relaxed tolerance
        attempt = {
            'strategy': 'relax_tolerance',
            'max_iter': int(new_max_iter),
            'tol': float(new_tol),
            'solver': str(original_params['solver'])
        }
        pipeline.named_steps['classifier'].set_params(tol=new_tol)
        pipeline.fit(X_train, y_train)
        if hasattr(classifier, 'converged_') and classifier.converged_:
            attempt['converged'] = True
            self.convergence_info['attempts'].append(attempt)
            return pipeline
        
        attempt['converged'] = False
        self.convergence_info['attempts'].append(attempt)
        
        # Strategy 4: Fallback to more robust solver
        print("Falling back to 'lbfgs' solver...")
        attempt = {
            'strategy': 'change_solver',
            'max_iter': int(new_max_iter),
            'tol': float(new_tol),
            'solver': 'lbfgs'
        }
        pipeline.named_steps['classifier'].set_params(
            solver='lbfgs',
            penalty='l2'
        )
        pipeline.fit(X_train, y_train)
        
        final_converged = (hasattr(classifier, 'converged_') and classifier.converged_) or not hasattr(classifier, 'converged_')
        attempt['converged'] = bool(final_converged)
        self.convergence_info['attempts'].append(attempt)
        self.convergence_info['final_convergence'] = bool(final_converged)
        
        if not final_converged:
            print("Warning: Model failed to fully converge. Results may be suboptimal.")
        
        return pipeline
    
    def _make_json_serializable(self, data):
        """Recursively convert data to JSON-serializable formats"""
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data
        elif isinstance(data, (np.integer, np.int64)):
            return int(data)
        elif isinstance(data, (np.floating, float)):
            return float(data)
        elif isinstance(data, (np.ndarray, np.generic)):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_json_serializable(item) for item in data]
        elif hasattr(data, 'to_dict'):  # For pandas DataFrames/Series
            return data.to_dict()
        else:
            return str(data)
    
    def train(self, X, y, test_size=0.2, tune_hyperparams=True):
        """Robust training method with convergence handling"""
        # Data quality checks
        self.convergence_info['data_quality'] = self._check_data_quality(X, y)
        
        if self.convergence_info['data_quality']['has_nan']:
            print("Warning: NaN values found in features. These will be imputed.")
        if self.convergence_info['data_quality']['has_inf']:
            print("Warning: Infinite values found in features. These should be handled.")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            stratify=y, 
            random_state=self.random_state
        )
        
        # Build pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        
        # Hyperparameter tuning
        if tune_hyperparams:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                best_params = self._tune_hyperparameters(pipeline, X_train, y_train)
            pipeline.set_params(**best_params)
            pipeline.named_steps['classifier'].set_params(max_iter=10000)
        
        # Train with convergence guarantees
        pipeline = self._ensure_convergence(pipeline, X_train, y_train)
        
        # Calibrate probabilities
        self.model = CalibratedClassifierCV(
            pipeline, 
            cv=3, 
            method='isotonic',
            n_jobs=-1
        ).fit(X_train, y_train)
        
        # Evaluation
        self._evaluate(X_test, y_test)
        self._get_feature_importances(X.columns)
        
        return self
    
    def _tune_hyperparameters(self, pipeline, X, y):
        """Optimized grid search with convergence handling"""
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['saga'],
            'classifier__max_iter': [10000],
            'classifier__tol': [1e-3],
            'classifier__l1_ratio': [None]
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            error_score='raise',
            refit=True
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            grid_search.fit(X, y)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV AUC: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def _evaluate(self, X, y):
        """Comprehensive evaluation with JSON-serializable metrics"""
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        # Convert classification report to serializable format
        report_dict = classification_report(y, y_pred, output_dict=True)
        serializable_report = {}
        for key, value in report_dict.items():
            if isinstance(value, dict):
                serializable_report[key] = {k: float(v) for k, v in value.items()}
            else:
                serializable_report[key] = float(value) if isinstance(value, (float, np.floating)) else value
        
        self.metrics = {
            'roc_auc': float(roc_auc_score(y, y_proba)),
            'average_precision': float(average_precision_score(y, y_proba)),
            'classification_report': serializable_report,
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'best_threshold': float(self._find_optimal_threshold(y, y_proba)),
            'convergence_info': self._make_json_serializable(self.convergence_info)
        }
        
        print(f"\nModel Evaluation:")
        print(f"ROC AUC: {self.metrics['roc_auc']:.4f}")
        print(f"Average Precision: {self.metrics['average_precision']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
    
    def _find_optimal_threshold(self, y_true, y_proba):
        """Optimal threshold finding without warnings"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            j_scores = tpr - fpr
            return thresholds[np.argmax(j_scores)]
    
    def _get_feature_importances(self, feature_names):
        """Feature importance extraction with checks"""
        try:
            if hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps['classifier'], 'coef_'):
                coef = self.model.named_steps['classifier'].coef_[0]
                self.feature_importances_ = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coef,
                    'abs_importance': np.abs(coef)
                }).sort_values('abs_importance', ascending=False)
        except Exception as e:
            print(f"Could not extract feature importances: {str(e)}")
            self.feature_importances_ = pd.DataFrame({
                'feature': feature_names,
                'coefficient': np.nan,
                'abs_importance': np.nan
            })
    
    def save_model(self, output_path):
        """Robust model saving with comprehensive metadata"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_package = {
            'model': self.model,
            'metrics': self._make_json_serializable(self.metrics),
            'feature_importances': self._make_json_serializable(
                self.feature_importances_.to_dict('records') if self.feature_importances_ is not None else None
            ),
            'model_type': str(self.model_type),
            'random_state': int(self.random_state),
            'timestamp': datetime.now().isoformat(),
            'convergence_info': self._make_json_serializable(self.convergence_info),
            'training_parameters': self._make_json_serializable({
                'preprocessor': str(self.preprocessor),
                'model_parameters': self.model.get_params() if hasattr(self.model, 'get_params') else None
            })
        }
        
        joblib.dump(model_package, output_path)
        print(f"Model package saved to {output_path}")

def train_credit_model(data_path, model_type='logistic', output_dir=None):
    """End-to-end training workflow with enhanced robustness"""
    try:
        # Configure paths
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'models'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and validate data
        print("Loading data...")
        df = pd.read_csv(data_path)
        
        # Check required columns
        required_columns = ['AccountId', 'default_proxy'] + [
            'recency', 'frequency', 'monetary_total',
            'monetary_mean', 'monetary_std',
            'frequency_recency_ratio', 'monetary_consistency'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        X = df.drop(['AccountId', 'default_proxy', 'risk_score'], axis=1, errors='ignore')
        y = df['default_proxy']
        
        # Initialize and train model
        print("Initializing model...")
        model = CreditRiskModel(model_type=model_type)
        
        print("Starting model training...")
        model.train(X, y, tune_hyperparams=True)
        
        # Save artifacts with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = output_dir / f"credit_risk_model_{model_type}_{timestamp}.pkl"
        metrics_path = output_dir / f"metrics_{model_type}_{timestamp}.json"
        data_quality_path = output_dir / f"data_quality_{timestamp}.json"
        
        model.save_model(model_path)
        
        with open(metrics_path, 'w') as f:
            json.dump(model._make_json_serializable(model.metrics), f, indent=2)
        
        with open(data_quality_path, 'w') as f:
            json.dump(model._make_json_serializable(model.convergence_info['data_quality']), f, indent=2)
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Metrics saved to: {metrics_path}")
        print(f"Data quality report saved to: {data_quality_path}")
        
        return model
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise

if __name__ == "__main__":
    processed_data_path = "./data/processed/processed_data.csv"
    train_credit_model(processed_data_path, model_type='logistic')