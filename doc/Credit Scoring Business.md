# Credit Scoring Business Understanding

## Basel II Accord and Model Interpretability
The Basel II Accord emphasizes rigorous risk measurement and capital adequacy requirements. This necessitates:
- **Interpretability**: Regulators and stakeholders must understand how decisions are made
- **Documentation**: Clear evidence of model development and validation processes
- **Auditability**: Ability to trace model outputs back to inputs and assumptions

## Proxy Variable Necessity and Risks
Since we lack direct default labels:
- **Necessity**: We must create a proxy using behavioral patterns (RFM metrics)
- **Risks**:
  - Proxy may not perfectly correlate with actual default risk
  - Potential for misclassification leading to incorrect risk assessments
  - Model may learn patterns specific to the proxy rather than true risk

## Model Selection Trade-offs
- **Simple Models (Logistic Regression with WoE)**:
  - Pros: Highly interpretable, easier to validate, regulatory compliance
  - Cons: May underfit complex patterns in the data
  
- **Complex Models (Gradient Boosting)**:
  - Pros: Higher predictive performance, captures non-linear relationships
  - Cons: Black-box nature makes regulatory approval challenging, harder to explain decisions