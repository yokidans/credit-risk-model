import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class CreditApplication:
    """Class to represent a credit application"""
    customer_id: str
    features: Dict[str, Any]
    requested_amount: float = 0.0
    requested_duration: int = 0  # in days

class CreditScorer:
    """Handles credit risk scoring and loan optimization"""
    
    def __init__(self, model_path, scorecard_params=None):
        """
        Initialize with trained model
        
        Args:
            model_path: Path to saved model file
            scorecard_params: Parameters for scorecard scaling
                             {'odds': 1/50, 'pdo': 20, 'score_at_odds': 600}
        """
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.model_type = model_data.get('model_type', 'logistic')
        self.feature_importances = model_data.get('feature_importances')
        
        # Default scorecard parameters (Standard 50:1 odds at 600 points, 20 PDO)
        self.scorecard_params = scorecard_params or {
            'odds': 1/50,  # 50:1 good:bad odds at anchor score
            'pdo': 20,      # Points to double odds
            'score_at_odds': 600  # Anchor score
        }
        
    def predict_risk(self, application: CreditApplication) -> Dict[str, float]:
        """Predict risk probability and generate score"""
        
        # Convert features to DataFrame for prediction
        features_df = pd.DataFrame([application.features])
        
        # Get probability of default (our proxy is 1=risky, so this is P(default))
        prob_default = self.model.predict_proba(features_df)[0, 1]
        
        # Calculate credit score using log-odds scaling
        score = self._probability_to_score(prob_default)
        
        # Determine optimal loan terms
        optimal_terms = self._calculate_optimal_terms(prob_default, 
                                                    application.requested_amount,
                                                    application.requested_duration)
        
        return {
            'customer_id': application.customer_id,
            'probability_default': float(prob_default),
            'credit_score': int(score),
            'approved': optimal_terms['approved'],
            'approved_amount': float(optimal_terms['approved_amount']),
            'approved_duration': int(optimal_terms['approved_duration']),
            'interest_rate': float(optimal_terms['interest_rate'])
        }
    
    def _probability_to_score(self, prob_default: float) -> int:
        """Convert probability to credit score using scorecard scaling"""
        
        if prob_default < 1e-6:
            prob_default = 1e-6
        elif prob_default > 1 - 1e-6:
            prob_default = 1 - 1e-6
            
        # Calculate log-odds
        odds = (1 - prob_default) / prob_default
        log_odds = np.log(odds)
        
        # Scorecard scaling formula
        params = self.scorecard_params
        score = params['score_at_odds'] - (params['pdo'] / np.log(2)) * log_odds
        
        # Round to nearest integer and ensure within typical bounds
        score = int(np.round(score))
        score = max(300, min(850, score))  # Typical credit score range
        
        return score
    
    def _calculate_optimal_terms(self, prob_default: float, 
                               requested_amount: float, 
                               requested_duration: int) -> Dict[str, Any]:
        """Calculate optimal loan terms based on risk"""
        
        # Risk-based pricing parameters (these would be tuned based on business rules)
        BASE_RATE = 0.10  # 10% base interest rate
        RISK_PREMIUM = 0.20  # Additional 20% for high risk
        
        # Calculate risk-adjusted interest rate
        interest_rate = BASE_RATE + (RISK_PREMIUM * prob_default)
        
        # Determine maximum allowable amount and duration based on risk
        max_amount = requested_amount * (1 - prob_default)
        max_duration = min(requested_duration, int(365 * (1 - prob_default)))
        
        # Approval logic (example: approve if prob_default < 0.7)
        approved = prob_default < 0.7
        
        # If approved, return terms with possible adjustments
        if approved:
            return {
                'approved': True,
                'approved_amount': min(requested_amount, max_amount),
                'approved_duration': min(requested_duration, max_duration),
                'interest_rate': interest_rate
            }
        else:
            return {
                'approved': False,
                'approved_amount': 0.0,
                'approved_duration': 0,
                'interest_rate': 0.0
            }

def load_example_application():
    """Create an example credit application"""
    features = {
        'recency': 45,
        'frequency': 12,
        'monetary_total': 1500.50,
        'monetary_mean': 125.04,
        'monetary_std': 50.25,
        'frequency_recency_ratio': 0.27,
        'monetary_consistency': 0.02,
        'fraud_count': 0
    }
    return CreditApplication(
        customer_id="CUST12345",
        features=features,
        requested_amount=1000.0,
        requested_duration=90
    )

if __name__ == "__main__":
    # Example usage
    model_path = "./models/credit_risk_model_logistic.pkl"
    scorer = CreditScorer(model_path)
    
    application = load_example_application()
    result = scorer.predict_risk(application)
    
    print("Credit Decision:")
    for key, value in result.items():
        print(f"{key:>20}: {value}")