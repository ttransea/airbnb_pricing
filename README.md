#  Seattle Airbnb Pricing Analysis

## Target Audience
New Airbnb host - may use this model as a reference to determine the optimal price range for a new listing.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Preprocessing](#preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Modeling](#modeling)
- [Results](#results)
- [Interpretability](#interpretability)


---

## Overview
This project investigates what drives private, entire-house listing prices in Seattle using a dataset of active short-term rental listings. Four regression models are trained and compared ( OLS, Random Forest, XGBoost, and SVR ) with performance evaluated on both log-scale and original-price metrics. SHAP values are used to interpret feature importance across models.

---

## Dataset

- **Source:** [InsideAirbnb.com](https://insideairbnb.com/seattle/) — scraped from Airbnb in August 2025
- **Scope:** Seattle, WA short-term/vacation rentals
- **Target variable:** `price` (nightly rate in USD)

**Filtering applied:**
- Listings with `maximum_nights > 365` removed (extended stays excluded)
- Listings with `minimum_nights > 14` removed
- Top 1% price outliers removed
- Analysis restricted to entire home/apartment listings only

---

## Methodology

### Preprocessing

- **Duplicate removal** via `drop_duplicates()`
- **Missing value imputation:**
  - Numeric columns (`bathrooms`, `bedrooms`, `beds`, `host_lifetime`, `price`) : k-NN Imputer (k=5)
  - `host_response_rate` : mode imputation
  - `host_acceptance_rate` : median imputation
  - `host_response_speed` : ordinal encoding + k-NN imputation
- **Binary encoding** for boolean fields: `host_is_superhost`, `host_has_profile_pic`, `host_identity_verified`, `instant_bookable`, `has_availability`

### Feature Engineering

| Feature | Description |
|---|---|
| `host_lifetime` | Years since host joined (relative to scrape date) |
| `host_response_speed` | Encoded response time: 0 = within an hour, 1 = within a day, 2 = slow |
| `recent_reviewed` | Binary: listing reviewed in 2025 |
| `north_seattle` / `sw_other` | Neighborhood group dummies (baseline = Downtown Seattle) |
| Amenity dummies | 18 binary features: kitchen, stove, WiFi, hot tub, washer, workspace, etc. |
| `unavail_30` | Days booked in the next 30 days  |
| `entire_place` | Binary flag for entire home/apt listings |

### Modeling

All models use a log1p-transformed target (`np.log1p(price)`) to address right-skewed price distributions. A 70/30 train/test split is used (`random_state=108`).

| Model | Key Hyperparameters |
|---|---|
| **OLS** | HC3 heteroskedasticity-robust standard errors |
| **Random Forest** | `n_estimators=100`, `max_depth=15`, `min_samples_leaf=5` |
| **XGBoost** | `n_estimators=100`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.7` |
| **SVR** | `kernel='linear'`, `C=10`, `epsilon=0.3`, StandardScaler pipeline |

**Features used across models:**
`host_acceptance_rate`, `host_is_superhost`, `bathrooms`, `bedrooms`, `accommodates`, `availability_30`, `review_scores_rating`, `review_scores_cleanliness`, `estimated_occupancy_l365d`, `north_seattle`, `sw_other`, `recent_reviewed`, `stove`, `free_street_parking`, `workspace`, `hot_tub`, `wifi`, `tv`, `washer`

---

## Results

Models are evaluated on both log scale and original price scale using R², RMSE, MAE, and MAPE.

> See the notebook for full metric tables and residual/actual-vs-predicted plots for each model.

---

## Interpretability

SHAP (SHapley Additive exPlanations) values are computed for both the OLS and SVR models to explain global feature importance and individual predictions:

- **Bar plots** (OLS and SVR) : mean absolute SHAP values per feature
- **Beeswarm plots** (OLS) : direction and magnitude of each feature's effect on price

```python
explainer = shap.Explainer(model_predict, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

---

## Required libraries

```
pandas
numpy
scikit-learn
statsmodels
xgboost
shap
matplotlib
seaborn
```

---

