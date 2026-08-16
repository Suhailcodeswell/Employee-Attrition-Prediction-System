# Employee Attrition Prediction

Machine learning system that flags employees at risk of leaving. Tuned for recall so HR can act before attrition happens.

## Results

| Metric | Value |
| --- | --- |
| Recall | 83% |
| Attrition base rate | 16.1% |
| Projected annual savings | $315K+ |
| Models in ensemble | 4 |

## Approach

- Feature engineering on the IBM HR dataset (overtime, tenure ratios, income, career growth)
- Class-balanced logistic regression, random forest, and gradient boosting
- Soft-voting ensemble with GridSearchCV
- Decision threshold tuned for recall
- Ranked risk file for HR review

## Structure

```
notebooks/   Feature engineering, modeling, evaluation
models/      Saved model artifacts
outputs/     Charts and risk exports
Dataset/     Source data
```

## How to run

1. Open the notebooks in `notebooks/` in order.
2. Train and evaluate the ensemble.
3. Export the ranked risk assessment from the evaluation notebook.

## Stack

Python, scikit-learn, pandas, Matplotlib
