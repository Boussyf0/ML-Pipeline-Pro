# 📊 Data Documentation

Comprehensive guide to the customer churn dataset, features, and data processing pipeline.

## 📁 Dataset Overview

### Telco Customer Churn Dataset

**Source**: [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Business Problem**: Predict which customers are likely to cancel their subscription (churn) to enable proactive retention strategies.

**Dataset Size**: ~7,000 customers with 21 features

**Target Variable**: `churn` (Yes/No - whether customer left within the last month)

**Churn Rate**: ~26.5% (industry-typical for telecommunications)

## 🗂️ Data Structure

```
data/
├── raw/                    # Original downloaded data
│   ├── customer_data.csv   # Training data (80% of dataset)
│   └── validation_data.csv # Validation data (20% of dataset)
├── processed/              # Processed and feature-engineered data  
│   ├── feature_config.yaml # Feature configuration
│   └── train_processed.csv # Processed training data
└── external/               # External data sources
    └── industry_benchmarks.csv
```

## 🏷️ Feature Dictionary

### Customer Demographics

| Feature | Type | Description | Values | Business Impact |
|---------|------|-------------|--------|-----------------|
| `customer_id` | String | Unique customer identifier | CUST_XXXXXX | Tracking individual customers |
| `gender` | Categorical | Customer gender | Male, Female | Demographic segmentation |
| `senior_citizen` | Binary | Whether customer is senior (65+) | 0, 1 | Age-based service packages |
| `partner` | Categorical | Has a partner | Yes, No | Household stability indicator |
| `dependents` | Categorical | Has dependents | Yes, No | Family size impact on loyalty |

### Service Tenure & Usage

| Feature | Type | Description | Values | Business Impact |
|---------|------|-------------|--------|-----------------|
| `tenure` | Numerical | Months customer has stayed | 0-72 months | Key loyalty indicator |
| `phone_service` | Categorical | Has phone service | Yes, No | Basic service adoption |
| `multiple_lines` | Categorical | Has multiple phone lines | Yes, No, No phone service | Service expansion |

### Internet & Add-on Services

| Feature | Type | Description | Values | Business Impact |
|---------|------|-------------|--------|-----------------|
| `internet_service` | Categorical | Type of internet service | DSL, Fiber optic, No | Core service type |
| `online_security` | Categorical | Online security add-on | Yes, No, No internet service | Security service adoption |
| `online_backup` | Categorical | Online backup add-on | Yes, No, No internet service | Data protection service |
| `device_protection` | Categorical | Device protection plan | Yes, No, No internet service | Hardware protection |
| `tech_support` | Categorical | Technical support plan | Yes, No, No internet service | Support service adoption |
| `streaming_tv` | Categorical | Streaming TV service | Yes, No, No internet service | Entertainment service |
| `streaming_movies` | Categorical | Streaming movies service | Yes, No, No internet service | Content consumption |

### Contract & Billing

| Feature | Type | Description | Values | Business Impact |
|---------|------|-------------|--------|-----------------|
| `contract` | Categorical | Contract term length | Month-to-month, One year, Two year | Commitment level |
| `paperless_billing` | Categorical | Uses paperless billing | Yes, No | Digital adoption |
| `payment_method` | Categorical | Payment method | Electronic check, Mailed check, Bank transfer, Credit card | Payment reliability |
| `monthly_charges` | Numerical | Current monthly charges | $18.25 - $118.75 | Revenue per customer |
| `total_charges` | Numerical | Total charges to date | $18.80 - $8684.80 | Customer lifetime value |

### Target Variable

| Feature | Type | Description | Values | Business Impact |
|---------|------|-------------|--------|-----------------|
| `churn` | Binary | Customer churned | Yes, No | Primary prediction target |

## 📊 Data Quality Assessment

### Completeness

```python
# Missing value analysis
Missing Values:
- total_charges: 11 customers (0.16%) - New customers with null values
- Other features: 0% missing
```

### Data Quality Issues

1. **Total Charges**: Some values stored as strings with spaces, converted to numeric
2. **New Customers**: Zero tenure customers with null total charges (handled)
3. **Outliers**: Monthly charges and total charges have some extreme values (capped)

### Data Distribution

```python
# Churn distribution
Churn Rate: 26.5%
- No (Retained): 73.5% (5,174 customers)  
- Yes (Churned): 26.5% (1,869 customers)

# Feature distributions
Contract Types:
- Month-to-month: 55% (highest churn risk)
- One year: 21%
- Two year: 24% (lowest churn risk)

Internet Service:
- Fiber optic: 44% (highest churn rate: 42%)
- DSL: 34% (churn rate: 19%)
- No internet: 22% (churn rate: 7%)
```

## 🔧 Data Processing Pipeline

### 1. Data Ingestion

```python
# Download from Kaggle
python scripts/download_kaggle_data.py

# Validate data quality
python scripts/validate_data.py --data-path data/raw/customer_data.csv
```

### 2. Data Cleaning

```python
# Automatic cleaning in preprocessor
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
df_clean = preprocessor.clean_data(df_raw)

# Cleaning steps:
# - Convert total_charges to numeric
# - Fill missing values (median for numeric, mode for categorical)
# - Remove duplicates
# - Cap outliers using IQR method
```

### 3. Feature Engineering

```python
# Engineered features
df_engineered = preprocessor.feature_engineering(df_clean)

# New features created:
# - total_amount: tenure * monthly_charges
# - avg_monthly_charges: monthly_charges / (tenure + 1)
# - tenure_category: New, Medium, Long, Very_Long
# - charges_category: Low, Medium, High
```

### 4. Data Validation

```python
# Great Expectations validation
validation_results = preprocessor.validate_data(df)

# Validation checks:
# - No missing values in critical columns
# - Data types match expected schema
# - Numerical values within expected ranges
# - Categorical values in valid sets
# - Target distribution is reasonable
```

## 📈 Exploratory Data Analysis

### Key Insights

#### High-Risk Customer Segments

1. **Contract Type Impact**:
   - Month-to-month: 42.7% churn rate
   - One year: 11.3% churn rate  
   - Two year: 2.8% churn rate

2. **Service Usage Patterns**:
   - Fiber optic customers: Higher churn (41.9%)
   - Customers without add-on services: Higher churn
   - New customers (tenure < 6 months): 50%+ churn rate

3. **Payment Behavior**:
   - Electronic check users: 45.3% churn rate
   - Credit card/bank transfer users: <20% churn rate

#### Feature Correlations

```python
# Strong predictors of churn:
Correlation with Churn:
- Contract type: -0.40 (strong negative)
- Tenure: -0.35 (strong negative)  
- Monthly charges: +0.19 (moderate positive)
- Internet service type: +0.17 (moderate positive)

# Feature interactions:
- Monthly charges ↔ Total charges: 0.65
- Tenure ↔ Total charges: 0.83
- Internet service ↔ Add-on services: 0.40+
```

## 🎯 Business Context

### Customer Acquisition Cost (CAC)
- **Telecommunications Industry Average**: $315
- **Customer Lifetime Value**: $1,800-2,400
- **Retention Impact**: Reducing churn by 5% can increase profits by 25-95%

### Churn Prediction Value
```python
# Business impact calculation
Monthly Customers = 7,043
Monthly Churn Rate = 26.5%
Monthly Churned Customers = 1,866

# Value of 10% improvement in churn prediction:
Additional Retained Customers = 187 per month
Annual Revenue Recovery = 187 × 12 × $64.76 = $1.45M
Model ROI = 1,450% (assuming $100K implementation cost)
```

### Customer Segments

1. **High-Value, High-Risk**: 
   - Fiber optic, month-to-month, high charges
   - **Priority**: Immediate retention campaigns

2. **New Customer Risk**:
   - Tenure < 6 months
   - **Priority**: Onboarding optimization

3. **Stable Long-term**:
   - Long contracts, multiple services, tenure > 24 months
   - **Priority**: Expansion/upsell opportunities

## 🔄 Data Drift Monitoring

### Drift Detection Methods

```python
# Implemented drift detection
from src.monitoring.drift_detector import DriftDetector

detector = DriftDetector()
drift_results = detector.detect_drift(reference_data, current_data)

# Detection methods:
1. Kolmogorov-Smirnov Test (numerical features)
2. Population Stability Index - PSI (categorical features)  
3. Wasserstein Distance (distribution comparison)
4. Jensen-Shannon Divergence (probability distributions)
```

### Drift Thresholds

```yaml
drift_detection:
  thresholds:
    ks_test: 0.05      # p-value threshold
    psi_score: 0.1     # PSI threshold
    wasserstein: 0.2   # Distance threshold
    js_divergence: 0.1 # Divergence threshold
```

### Expected Drift Patterns

1. **Seasonal Trends**: 
   - Higher churn in Q1 (post-holiday cost cutting)
   - Lower churn in Q4 (holiday season)

2. **Market Changes**:
   - New competitor entry → increased churn
   - Service improvements → reduced churn
   - Price changes → payment method shifts

3. **Product Evolution**:
   - 5G rollout → internet service distribution changes
   - Streaming service partnerships → feature adoption changes

## 📋 Data Governance

### Data Privacy & Security

```python
# PII handling
Sensitive Fields:
- customer_id: Hashed in production
- Location data: Not included in dataset
- Payment info: Anonymized payment method only

# GDPR Compliance:
- Right to be forgotten: Customer deletion process
- Data portability: Export capabilities
- Consent tracking: Service agreement tracking
```

### Data Lineage

```
External Source (Kaggle) 
    ↓
Raw Data Ingestion
    ↓  
Data Validation (Great Expectations)
    ↓
Data Cleaning & Preprocessing
    ↓
Feature Engineering
    ↓
Model Training Data
    ↓
Production Predictions
    ↓
Performance Monitoring
    ↓
Drift Detection & Alerts
```

### Data Quality Monitoring

```python
# Automated monitoring
Daily Checks:
✅ Data freshness (last update < 24h)
✅ Row count within expected range
✅ No missing values in key columns
✅ Feature distributions within control limits
✅ Target variable balance maintained

Weekly Checks:
✅ Feature correlation stability  
✅ New categorical values detection
✅ Outlier detection and flagging
✅ Data drift assessment
✅ Model performance degradation
```

## 🚀 Next Steps

### Data Enhancement Opportunities

1. **External Data Integration**:
   - Economic indicators (GDP, unemployment)
   - Competitor pricing data
   - Market penetration rates

2. **Behavioral Data**:
   - Customer service interaction history
   - Usage patterns (peak times, data consumption)
   - Support ticket sentiment analysis

3. **Real-time Features**:
   - Recent payment delays
   - Service outage incidents
   - Competitor activity in customer area

### Advanced Analytics

1. **Customer Segmentation**:
   - RFM analysis (Recency, Frequency, Monetary)
   - Cohort analysis for retention patterns
   - Propensity scoring for cross-sell/upsell

2. **Time Series Analysis**:
   - Seasonal churn pattern detection
   - Trend analysis for early warning
   - Lifecycle stage prediction

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Customer Churn Analysis Best Practices](https://blog.hubspot.com/service/what-does-it-cost-to-acquire-a-customer)

---

**Need help with data?** Check the [data exploration notebook](../notebooks/data_exploration.ipynb) or contact the data team.