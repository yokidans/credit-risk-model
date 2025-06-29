from pydantic import BaseModel, Field, confloat, conint
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime, date  # Added datetime import

class CreditProductType(str, Enum):
    PERSONAL_LOAN = "personal_loan"
    MORTGAGE = "mortgage"
    CREDIT_CARD = "credit_card"
    BUSINESS_LOAN = "business_loan"
    AUTO_LOAN = "auto_loan"

class CustomerDemographics(BaseModel):
    age: int
    education_level: str
    employment_status: str
    years_at_current_job: int
    marital_status: str
    dependents: int

class FinancialHistory(BaseModel):
    credit_score: int
    outstanding_debt: float
    delinquencies_2y: int
    credit_utilization: confloat(ge=0, le=1)
    bankruptcies: bool
    credit_history_length: int  # in months

class CreditRequest(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    timestamp: Optional[datetime] = Field(None, description="Request timestamp")
    product_type: CreditProductType = Field(..., description="Type of credit product")
    requested_amount: float = Field(..., gt=0, description="Requested loan amount")
    requested_duration: int = Field(..., gt=0, description="Loan term in months")
    demographics: CustomerDemographics
    financials: FinancialHistory
    custom_features: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional model-specific features"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional request metadata"
    )

class CreditDecision(str, Enum):
    APPROVED = "approved"
    DECLINED = "declined"
    MANUAL_REVIEW = "manual_review"

class CreditResponse(BaseModel):
    customer_id: str
    request_id: str
    timestamp: datetime
    product_type: CreditProductType
    decision: CreditDecision
    probability_default: confloat(ge=0, le=1)
    credit_score: conint(ge=300, le=850)
    risk_band: str
    approved_amount: Optional[float] = None
    approved_duration: Optional[int] = None
    interest_rate: Optional[confloat(ge=0)] = None
    decision_reasons: List[str]
    alternative_offers: Optional[List[Dict[str, Any]]] = None
    model_version: str
    model_metadata: Optional[Dict[str, Any]] = None

class BatchCreditRequest(BaseModel):
    requests: List[CreditRequest]
    batch_id: Optional[str] = Field(
        None,
        description="Batch identifier for tracking"
    )

class FeatureImportance(BaseModel):
    feature_name: str
    importance_score: float
    direction: str  # positive/negative impact
    value: Any

class ExplainabilityResponse(BaseModel):
    customer_id: str
    request_id: str
    shap_values: List[FeatureImportance]
    decision_boundary: float
    decision_factors: List[str]
    global_feature_importance: Optional[List[FeatureImportance]] = None
    counterfactuals: Optional[List[Dict[str, Any]]] = None

class SimulationScenario(BaseModel):
    scenario_id: str
    feature_overrides: Dict[str, Any]
    amount_override: Optional[float] = None
    duration_override: Optional[int] = None
    description: Optional[str] = None

class SimulationRequest(BaseModel):
    base_request: CreditRequest
    scenarios: List[SimulationScenario]

class SimulationResult(BaseModel):
    scenario_id: str
    result: CreditResponse
    changed_features: List[str]
    delta_probability: Optional[float] = None
    delta_score: Optional[int] = None

class SimulationResponse(BaseModel):
    base_result: CreditResponse
    simulations: List[SimulationResult]
    optimal_scenario: Optional[SimulationResult] = None

class ModelMetrics(BaseModel):
    model_version: str
    training_date: date
    performance_metrics: Dict[str, float]
    fairness_metrics: Optional[Dict[str, float]] = None
    data_drift: Optional[Dict[str, float]] = None
    concept_drift: Optional[float] = None
    feature_stability: Optional[Dict[str, float]] = None