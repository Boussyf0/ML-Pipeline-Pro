# 🧪 A/B Testing Guide

Comprehensive A/B testing framework for comparing model versions in production with statistical rigor and automated analysis.

## 🏗️ A/B Testing Architecture

```
Traffic → Assignment Engine → Model Routing → Result Collection → Statistical Analysis
   ↓           ↓                 ↓              ↓                    ↓
User Hash   Redis Cache      Model A/B       Event Logging      Hypothesis Testing
Customer    Assignment       Predictions     Database Storage    Confidence Intervals
Request     Persistence      Responses       Metrics Collection  Significance Tests
```

## 🎯 Key Components

### Experiment Management
- **Experiment Configuration**: Define test parameters and success metrics
- **Traffic Allocation**: Smart traffic splitting algorithms
- **Statistical Analysis**: Rigorous hypothesis testing and confidence intervals
- **Automated Decision Making**: Statistical significance and practical significance

### Traffic Splitting Methods

#### User Hash Based (Recommended)
```python
# Consistent assignment based on user ID
def get_user_assignment(user_id: str, experiment_id: str) -> str:
    hash_input = f"{user_id}_{experiment_id}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()
    numeric_hash = int(hash_value[:8], 16)
    return "A" if (numeric_hash % 100) < 50 else "B"
```

#### Feature Hash Based
```python
# Assignment based on feature combinations
def get_feature_assignment(features: Dict, experiment_id: str) -> str:
    feature_string = "_".join([f"{k}:{v}" for k, v in sorted(features.items())])
    hash_input = f"{feature_string}_{experiment_id}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()
    numeric_hash = int(hash_value[:8], 16)
    return "A" if (numeric_hash % 100) < 50 else "B"
```

#### Random Assignment
```python
# Pure random assignment (less consistent)
def get_random_assignment() -> str:
    return "A" if random.random() < 0.5 else "B"
```

## 🧪 Running A/B Tests

### Creating an Experiment

```bash
# Create new A/B test experiment
python scripts/ab_test_cli.py create \
  --name "XGBoost vs LightGBM Comparison" \
  --description "Compare XGBoost v1.2.0 vs LightGBM v1.3.0 for churn prediction" \
  --model-a churn-predictor \
  --model-a-version 1.2.0 \
  --model-b churn-predictor-lightgbm \
  --model-b-version 1.3.0 \
  --traffic-split 0.5 \
  --allocation-method user_hash \
  --duration-days 14 \
  --min-sample-size 1000 \
  --significance-level 0.05 \
  --success-metrics "accuracy,precision,recall,auc_roc" \
  --created-by "data-scientist-john"
```

### Starting an Experiment

```bash
# Start the experiment
python scripts/ab_test_cli.py start exp_abc123

# Check experiment status
python scripts/ab_test_cli.py status exp_abc123

# List all experiments
python scripts/ab_test_cli.py list --status active
```

### Monitoring Experiment Progress

```bash
# Real-time status monitoring
python scripts/ab_test_cli.py status exp_abc123

# Example output:
📊 Experiment Status: exp_abc123
   Status: ACTIVE
   Duration: 5 days
   Total Assignments: 2,456

   Models:
     A: churn-predictor:1.2.0
     B: churn-predictor-lightgbm:1.3.0

   Traffic Split:
     Configured: A=50%, B=50%
     Actual: A=51%, B=49%

   Sample Sizes:
     A: 1,253
     B: 1,203
```

## 📊 Statistical Analysis

### Hypothesis Testing

```python
# src/ab_testing/statistical_analysis.py
from scipy import stats
import numpy as np

class StatisticalAnalyzer:
    def __init__(self, significance_level: float = 0.05, power: float = 0.8):
        self.alpha = significance_level
        self.power = power
    
    def two_proportion_test(self, successes_a: int, total_a: int,
                           successes_b: int, total_b: int):
        """Two-proportion z-test for conversion rates."""
        # Calculate proportions
        p1 = successes_a / total_a
        p2 = successes_b / total_b
        
        # Pooled proportion
        p_pool = (successes_a + successes_b) / (total_a + total_b)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/total_a + 1/total_b))
        
        # Z statistic and p-value
        z_stat = (p1 - p2) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Confidence interval
        se_diff = np.sqrt(p1*(1-p1)/total_a + p2*(1-p2)/total_b)
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        ci_lower = (p1 - p2) - z_critical * se_diff
        ci_upper = (p1 - p2) + z_critical * se_diff
        
        return {
            'statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': (p2 - p1) / p1 if p1 > 0 else 0
        }
```

### Bayesian Analysis

```python
def bayesian_ab_test(successes_a: int, total_a: int,
                    successes_b: int, total_b: int):
    """Bayesian A/B test using Beta-Binomial model."""
    from scipy.stats import beta
    
    # Posterior parameters (using uniform prior)
    alpha_a, beta_a = 1 + successes_a, 1 + total_a - successes_a
    alpha_b, beta_b = 1 + successes_b, 1 + total_b - successes_b
    
    # Monte Carlo simulation
    samples_a = beta.rvs(alpha_a, beta_a, size=100000)
    samples_b = beta.rvs(alpha_b, beta_b, size=100000)
    
    # Probability that B > A
    prob_b_better = np.mean(samples_b > samples_a)
    
    # Credible interval for difference
    differences = samples_b - samples_a
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)
    
    return {
        'probability_b_better': prob_b_better,
        'credible_interval': (ci_lower, ci_upper),
        'expected_improvement': np.mean(differences)
    }
```

### Sequential Testing

```python
def sequential_probability_ratio_test(successes_a: int, total_a: int,
                                    successes_b: int, total_b: int):
    """Sequential test for early stopping decision."""
    p_a = successes_a / total_a if total_a > 0 else 0
    p_b = successes_b / total_b if total_b > 0 else 0
    
    # Log likelihood ratio
    if p_a > 0 and p_b > 0:
        llr = (successes_a * np.log(p_a/p_b) + 
               (total_a - successes_a) * np.log((1-p_a)/(1-p_b)))
    else:
        llr = 0
    
    # Decision boundaries
    alpha, beta = 0.05, 0.2
    upper_boundary = np.log((1-beta)/alpha)
    lower_boundary = np.log(beta/(1-alpha))
    
    if llr >= upper_boundary:
        return {"decision": "stop", "winner": "A", "confidence": "high"}
    elif llr <= lower_boundary:
        return {"decision": "stop", "winner": "B", "confidence": "high"}
    else:
        return {"decision": "continue", "winner": None, "confidence": "low"}
```

## 📈 Sample Size Calculation

### Power Analysis

```bash
# Calculate required sample size
python scripts/ab_test_cli.py sample-size \
  --baseline-rate 0.15 \
  --mde 0.10 \
  --power 0.8 \
  --significance-level 0.05

# Example output:
📊 Sample Size Calculation
==============================
Baseline Rate: 15.0%
Minimum Detectable Effect: 10.0%
Statistical Power: 80.0%
Significance Level: 5.0%

Required Sample Size per Group: 2,863
Total Sample Size: 5,726

Duration Estimates (based on daily traffic):
  100 visitors/day: 57.3 days
  500 visitors/day: 11.5 days
  1,000 visitors/day: 5.7 days
  5,000 visitors/day: 1.1 days
  10,000 visitors/day: 0.6 days
```

### Sample Size Formula

```python
def calculate_sample_size(baseline_rate: float, mde: float, 
                         power: float = 0.8, alpha: float = 0.05):
    """Calculate required sample size per group."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)  # Relative change
    
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Pooled proportion
    p_pool = (p1 + p2) / 2
    
    # Sample size calculation
    numerator = (z_alpha * np.sqrt(2 * p_pool * (1 - p_pool)) + 
                z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
    denominator = (p1 - p2)**2
    
    return int(np.ceil(numerator / denominator))
```

## 🔍 Analyzing Results

### Running Analysis

```bash
# Analyze experiment results
python scripts/ab_test_cli.py analyze exp_abc123 --output results.json

# Example output:
📈 Analysis Results: exp_abc123
==================================================
Sample Sizes: A=1,253, B=1,203
Test Duration: 14 days

Model A Metrics:
  accuracy: 0.8420
  precision: 0.7890
  recall: 0.7650
  auc_roc: 0.8670

Model B Metrics:
  accuracy: 0.8580
  precision: 0.8120
  recall: 0.7890
  auc_roc: 0.8820

Statistical Tests:
  accuracy: p-value=0.0234 ✅
  precision: p-value=0.0456 ✅
  recall: p-value=0.1203 ❌
  auc_roc: p-value=0.0189 ✅

🏆 Winner: Model B
📋 Recommendation: Deploy Model B to production
```

### Analysis Results Structure

```python
@dataclass
class AnalysisResult:
    experiment_id: str
    model_a_metrics: Dict[str, float]
    model_b_metrics: Dict[str, float] 
    sample_size_a: int
    sample_size_b: int
    statistical_tests: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    effect_sizes: Dict[str, float]
    statistical_significance: bool
    practical_significance: bool
    winner: Optional[str]
    recommendation: str
    test_duration_days: int
    analyzed_at: datetime
```

## 🚦 Decision Framework

### Statistical Significance

```python
def assess_statistical_significance(p_values: Dict[str, float], 
                                  alpha: float = 0.05) -> bool:
    """Assess if results are statistically significant."""
    # Multiple testing correction (Bonferroni)
    adjusted_alpha = alpha / len(p_values)
    
    significant_tests = [p < adjusted_alpha for p in p_values.values()]
    
    # Require majority of metrics to be significant
    return sum(significant_tests) > len(significant_tests) / 2
```

### Practical Significance

```python
def assess_practical_significance(effect_sizes: Dict[str, float],
                                min_practical_effect: float = 0.02) -> bool:
    """Assess if results are practically significant."""
    # Check if effect sizes meet minimum practical threshold
    significant_effects = [
        abs(effect) >= min_practical_effect 
        for effect in effect_sizes.values()
    ]
    
    return any(significant_effects)
```

### Automated Decision Rules

```python
def make_recommendation(analysis_result: AnalysisResult) -> str:
    """Generate automated recommendation."""
    if not analysis_result.statistical_significance:
        return "No significant difference detected. Keep current model."
    
    if not analysis_result.practical_significance:
        return "Statistically significant but not practically significant. Consider cost of change."
    
    if analysis_result.winner == "B":
        # Check for negative effects in important metrics
        critical_metrics = ['accuracy', 'precision', 'auc_roc']
        b_worse_in_critical = any(
            analysis_result.model_b_metrics.get(metric, 0) < 
            analysis_result.model_a_metrics.get(metric, 0)
            for metric in critical_metrics
        )
        
        if b_worse_in_critical:
            return "Mixed results. Model B better overall but worse in critical metrics. Manual review recommended."
        else:
            return "Deploy Model B to production. Significant improvement detected."
    else:
        return "Keep current model (Model A). No improvement from Model B."
```

## 🎛️ Advanced Features

### Multi-Armed Bandit

```python
class EpsilonGreedyBandit:
    """Epsilon-greedy bandit for dynamic traffic allocation."""
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.model_rewards = defaultdict(list)
        
    def select_model(self, available_models: List[str]) -> str:
        """Select model using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            # Exploration: random selection
            return random.choice(available_models)
        else:
            # Exploitation: select best performing model
            if not self.model_rewards:
                return random.choice(available_models)
                
            avg_rewards = {
                model: np.mean(rewards) 
                for model, rewards in self.model_rewards.items()
                if model in available_models
            }
            
            return max(avg_rewards, key=avg_rewards.get)
    
    def update_reward(self, model: str, reward: float):
        """Update model reward based on feedback."""
        self.model_rewards[model].append(reward)
```

### Segmented Analysis

```python
def analyze_by_segments(experiment_id: str, 
                       segment_features: List[str]) -> Dict[str, AnalysisResult]:
    """Analyze experiment results by customer segments."""
    results = {}
    
    # Get experiment data
    experiment_data = get_experiment_data(experiment_id)
    
    # Create segments based on feature combinations
    segments = create_segments(experiment_data, segment_features)
    
    for segment_name, segment_data in segments.items():
        if len(segment_data) >= 100:  # Minimum sample size
            segment_result = analyze_segment_data(segment_data)
            results[segment_name] = segment_result
    
    return results

def create_segments(data: pd.DataFrame, 
                   features: List[str]) -> Dict[str, pd.DataFrame]:
    """Create customer segments based on features."""
    segments = {}
    
    # Example: Segment by contract type and tenure
    if 'contract' in features and 'tenure' in features:
        for contract in data['contract'].unique():
            for tenure_range in ['0-12', '12-24', '24+']:
                if tenure_range == '0-12':
                    condition = (data['contract'] == contract) & (data['tenure'] <= 12)
                elif tenure_range == '12-24':
                    condition = (data['contract'] == contract) & (data['tenure'] > 12) & (data['tenure'] <= 24)
                else:
                    condition = (data['contract'] == contract) & (data['tenure'] > 24)
                
                segment_data = data[condition]
                if len(segment_data) > 0:
                    segments[f"{contract}_{tenure_range}"] = segment_data
    
    return segments
```

## 🔧 Configuration

### Experiment Configuration

```yaml
# config/ab_testing.yaml
ab_testing:
  enabled: true
  
  default_settings:
    significance_level: 0.05
    power: 0.8
    min_sample_size_per_group: 1000
    max_duration_days: 30
    
  traffic_allocation:
    method: "user_hash"  # user_hash, feature_hash, random
    hash_seed: "mlops-2024"
    
  statistical_tests:
    multiple_testing_correction: "bonferroni"
    effect_size_threshold: 0.02
    
  automation:
    auto_stop_on_significance: false
    auto_promote_winner: false
    early_stopping_enabled: true
    
  monitoring:
    check_interval_hours: 6
    alert_on_significant_result: true
    alert_on_sample_size_reached: true
```

### Database Schema

```sql
-- A/B testing database tables

CREATE TABLE experiments (
    experiment_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    model_a_name VARCHAR(255) NOT NULL,
    model_a_version VARCHAR(255) NOT NULL,
    model_b_name VARCHAR(255) NOT NULL,
    model_b_version VARCHAR(255) NOT NULL,
    traffic_split FLOAT NOT NULL,
    allocation_method VARCHAR(50) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP,
    status VARCHAR(50) NOT NULL,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE experiment_assignments (
    assignment_id UUID PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(experiment_id),
    user_id VARCHAR(255) NOT NULL,
    variant VARCHAR(10) NOT NULL, -- 'A' or 'B'
    assigned_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(experiment_id, user_id)
);

CREATE TABLE experiment_events (
    event_id UUID PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(experiment_id),
    user_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL, -- 'prediction', 'conversion', etc.
    event_value FLOAT,
    variant VARCHAR(10) NOT NULL,
    occurred_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);
```

## 📊 Monitoring A/B Tests

### Grafana Dashboards

```json
{
  "title": "A/B Testing Dashboard",
  "panels": [
    {
      "title": "Active Experiments",
      "targets": ["count(experiments{status='ACTIVE'})"]
    },
    {
      "title": "Traffic Split Distribution", 
      "targets": [
        "experiment_assignments_total{variant='A'}",
        "experiment_assignments_total{variant='B'}"
      ]
    },
    {
      "title": "Conversion Rates by Variant",
      "targets": [
        "experiment_conversion_rate{variant='A'}",
        "experiment_conversion_rate{variant='B'}"
      ]
    }
  ]
}
```

### Automated Monitoring

```python
# scripts/monitor_ab_tests.py
def monitor_active_experiments():
    """Monitor active A/B tests and send alerts."""
    manager = ExperimentManager()
    active_experiments = manager.list_experiments(ExperimentStatus.ACTIVE)
    
    for experiment in active_experiments:
        # Check if sample size threshold reached
        status = get_experiment_status(experiment.experiment_id)
        
        if (status['allocation']['A'] >= experiment.minimum_sample_size and
            status['allocation']['B'] >= experiment.minimum_sample_size):
            
            # Run interim analysis
            result = manager.analyze_experiment(experiment.experiment_id)
            
            if result.statistical_significance:
                send_significance_alert(experiment, result)
                
                # Auto-stop if enabled
                if config.get('auto_stop_on_significance'):
                    manager.update_experiment_status(
                        experiment.experiment_id, 
                        ExperimentStatus.COMPLETED
                    )
```

## 🚨 Best Practices

### Experimental Design
- **Define Success Metrics**: Clear, measurable outcomes
- **Sample Size Planning**: Calculate required sample size before starting
- **Randomization**: Use consistent assignment methods
- **Segment Analysis**: Plan for segment-specific analysis

### Statistical Rigor
- **Multiple Testing Correction**: Account for multiple comparisons
- **Effect Size**: Consider practical significance, not just statistical
- **Power Analysis**: Ensure adequate power to detect meaningful effects
- **Early Stopping**: Use proper sequential testing methods

### Operational Excellence
- **Monitoring**: Continuous monitoring of experiment health
- **Documentation**: Document experiment design and results
- **Rollback Plan**: Have rollback procedures ready
- **Stakeholder Communication**: Regular updates to stakeholders

## 📚 Next Steps

After mastering A/B testing:

1. **[Deployment Guide](deployment.md)**: Deploy winning models to production
2. **[Monitoring Guide](monitoring.md)**: Set up comprehensive monitoring
3. **[API Documentation](api.md)**: Understand serving infrastructure
4. **[Training Pipeline](training.md)**: Automated model training

---

**Need help?** Review experiment logs with `python scripts/ab_test_cli.py status <experiment_id>` or contact the MLOps team.