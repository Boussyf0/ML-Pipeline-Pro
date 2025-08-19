"""Advanced statistical analysis for A/B testing."""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from scipy.stats import beta, norm
import math
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class TestType(Enum):
    """Statistical test types."""
    TWO_PROPORTION_Z_TEST = "two_proportion_z_test"
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    CHI_SQUARE = "chi_square"
    BOOTSTRAP = "bootstrap"


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_type: TestType
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Tuple[float, float]
    power: Optional[float]
    is_significant: bool
    interpretation: str


@dataclass
class BayesianTestResult:
    """Result of a Bayesian A/B test."""
    probability_b_better: float
    credible_interval_difference: Tuple[float, float]
    expected_loss_a: float
    expected_loss_b: float
    recommendation: str


class StatisticalAnalyzer:
    """Advanced statistical analysis for A/B testing."""
    
    def __init__(self, significance_level: float = 0.05, power: float = 0.8):
        """Initialize statistical analyzer."""
        self.significance_level = significance_level
        self.power = power
        
    def two_proportion_test(self, successes_a: int, total_a: int,
                           successes_b: int, total_b: int) -> StatisticalTestResult:
        """Two-proportion z-test for conversion rates."""
        try:
            # Calculate proportions
            p1 = successes_a / total_a if total_a > 0 else 0
            p2 = successes_b / total_b if total_b > 0 else 0
            
            # Pooled proportion
            p_pool = (successes_a + successes_b) / (total_a + total_b)
            
            # Standard error
            se = np.sqrt(p_pool * (1 - p_pool) * (1/total_a + 1/total_b))
            
            # Z statistic
            z_stat = (p1 - p2) / se if se > 0 else 0
            
            # P-value (two-tailed)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            # Effect size (Cohen's h)
            h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
            
            # Confidence interval for difference in proportions
            se_diff = np.sqrt(p1 * (1 - p1) / total_a + p2 * (1 - p2) / total_b)
            z_critical = stats.norm.ppf(1 - self.significance_level / 2)
            diff = p1 - p2
            ci_lower = diff - z_critical * se_diff
            ci_upper = diff + z_critical * se_diff
            
            # Power calculation
            power = self._calculate_power_two_proportion(total_a, total_b, p1, p2)
            
            # Interpretation
            is_significant = p_value < self.significance_level
            interpretation = self._interpret_two_proportion_test(
                p1, p2, p_value, h, is_significant
            )
            
            return StatisticalTestResult(
                test_type=TestType.TWO_PROPORTION_Z_TEST,
                statistic=z_stat,
                p_value=p_value,
                effect_size=h,
                confidence_interval=(ci_lower, ci_upper),
                power=power,
                is_significant=is_significant,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Two-proportion test failed: {e}")
            return self._create_error_result(TestType.TWO_PROPORTION_Z_TEST)
            
    def welch_t_test(self, group_a: List[float], group_b: List[float]) -> StatisticalTestResult:
        """Welch's t-test for continuous metrics with unequal variances."""
        try:
            # Convert to numpy arrays
            a = np.array(group_a)
            b = np.array(group_b)
            
            # Descriptive statistics
            mean_a, std_a, n_a = a.mean(), a.std(ddof=1), len(a)
            mean_b, std_b, n_b = b.mean(), b.std(ddof=1), len(b)
            
            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
            cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for difference in means
            se_diff = np.sqrt(std_a**2 / n_a + std_b**2 / n_b)
            df = (std_a**2 / n_a + std_b**2 / n_b)**2 / (
                (std_a**2 / n_a)**2 / (n_a - 1) + (std_b**2 / n_b)**2 / (n_b - 1)
            )
            t_critical = stats.t.ppf(1 - self.significance_level / 2, df)
            diff = mean_a - mean_b
            ci_lower = diff - t_critical * se_diff
            ci_upper = diff + t_critical * se_diff
            
            # Power calculation
            power = self._calculate_power_t_test(n_a, n_b, cohens_d)
            
            # Interpretation
            is_significant = p_value < self.significance_level
            interpretation = self._interpret_t_test(
                mean_a, mean_b, p_value, cohens_d, is_significant
            )
            
            return StatisticalTestResult(
                test_type=TestType.WELCH_T_TEST,
                statistic=t_stat,
                p_value=p_value,
                effect_size=cohens_d,
                confidence_interval=(ci_lower, ci_upper),
                power=power,
                is_significant=is_significant,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Welch's t-test failed: {e}")
            return self._create_error_result(TestType.WELCH_T_TEST)
            
    def mann_whitney_u_test(self, group_a: List[float], group_b: List[float]) -> StatisticalTestResult:
        """Mann-Whitney U test for non-parametric comparison."""
        try:
            # Convert to numpy arrays
            a = np.array(group_a)
            b = np.array(group_b)
            
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
            
            # Effect size (rank-biserial correlation)
            n_a, n_b = len(a), len(b)
            r = 1 - (2 * u_stat) / (n_a * n_b)  # rank-biserial correlation
            
            # Confidence interval (approximate)
            # For Mann-Whitney U, CI is complex, using median difference approximation
            combined = np.concatenate([a, b])
            median_a = np.median(a)
            median_b = np.median(b)
            
            # Bootstrap confidence interval for median difference
            ci_lower, ci_upper = self._bootstrap_median_difference_ci(a, b)
            
            # Interpretation
            is_significant = p_value < self.significance_level
            interpretation = self._interpret_mann_whitney(
                median_a, median_b, p_value, r, is_significant
            )
            
            return StatisticalTestResult(
                test_type=TestType.MANN_WHITNEY_U,
                statistic=u_stat,
                p_value=p_value,
                effect_size=r,
                confidence_interval=(ci_lower, ci_upper),
                power=None,  # Power calculation complex for non-parametric tests
                is_significant=is_significant,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Mann-Whitney U test failed: {e}")
            return self._create_error_result(TestType.MANN_WHITNEY_U)
            
    def bayesian_ab_test(self, successes_a: int, total_a: int,
                        successes_b: int, total_b: int,
                        prior_alpha: float = 1, prior_beta: float = 1) -> BayesianTestResult:
        """Bayesian A/B test using Beta-Binomial model."""
        try:
            # Posterior parameters
            alpha_a = prior_alpha + successes_a
            beta_a = prior_beta + total_a - successes_a
            
            alpha_b = prior_alpha + successes_b  
            beta_b = prior_beta + total_b - successes_b
            
            # Monte Carlo simulation for probability B > A
            n_samples = 100000
            samples_a = beta.rvs(alpha_a, beta_a, size=n_samples)
            samples_b = beta.rvs(alpha_b, beta_b, size=n_samples)
            
            prob_b_better = np.mean(samples_b > samples_a)
            
            # Credible interval for difference B - A
            differences = samples_b - samples_a
            ci_lower = np.percentile(differences, 2.5)
            ci_upper = np.percentile(differences, 97.5)
            
            # Expected loss calculations
            # Loss if we choose A when B is better
            loss_choose_a = np.mean(np.maximum(0, samples_b - samples_a))
            # Loss if we choose B when A is better  
            loss_choose_b = np.mean(np.maximum(0, samples_a - samples_b))
            
            # Recommendation based on expected loss
            if loss_choose_a < loss_choose_b:
                recommendation = "Choose A (lower expected loss)"
            elif loss_choose_b < loss_choose_a:
                recommendation = "Choose B (lower expected loss)"
            else:
                recommendation = "No clear winner (similar expected loss)"
                
            return BayesianTestResult(
                probability_b_better=prob_b_better,
                credible_interval_difference=(ci_lower, ci_upper),
                expected_loss_a=loss_choose_a,
                expected_loss_b=loss_choose_b,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Bayesian A/B test failed: {e}")
            return BayesianTestResult(
                probability_b_better=0.5,
                credible_interval_difference=(0.0, 0.0),
                expected_loss_a=0.0,
                expected_loss_b=0.0,
                recommendation="Analysis failed"
            )
            
    def sequential_test(self, successes_a: int, total_a: int,
                       successes_b: int, total_b: int,
                       alpha: float = 0.05, beta: float = 0.2) -> Dict[str, Any]:
        """Sequential probability ratio test for early stopping."""
        try:
            # Observed proportions
            p_a = successes_a / total_a if total_a > 0 else 0
            p_b = successes_b / total_b if total_b > 0 else 0
            
            # Log likelihood ratio
            if p_a > 0 and p_b > 0 and p_a < 1 and p_b < 1:
                llr = (successes_a * np.log(p_a / p_b) + 
                       (total_a - successes_a) * np.log((1 - p_a) / (1 - p_b)))
            else:
                llr = 0
                
            # Decision boundaries
            upper_boundary = np.log((1 - beta) / alpha)
            lower_boundary = np.log(beta / (1 - alpha))
            
            # Decision
            if llr >= upper_boundary:
                decision = "Stop - A is significantly better"
                can_stop = True
            elif llr <= lower_boundary:
                decision = "Stop - B is significantly better" 
                can_stop = True
            else:
                decision = "Continue - insufficient evidence"
                can_stop = False
                
            return {
                "log_likelihood_ratio": llr,
                "upper_boundary": upper_boundary,
                "lower_boundary": lower_boundary,
                "decision": decision,
                "can_stop": can_stop,
                "samples_a": total_a,
                "samples_b": total_b
            }
            
        except Exception as e:
            logger.error(f"Sequential test failed: {e}")
            return {
                "error": str(e),
                "can_stop": False,
                "decision": "Continue - analysis error"
            }
            
    def calculate_sample_size(self, baseline_rate: float, minimum_detectable_effect: float,
                             power: float = 0.8, significance_level: float = 0.05) -> int:
        """Calculate required sample size per group."""
        try:
            # Convert MDE to absolute difference
            p1 = baseline_rate
            p2 = baseline_rate * (1 + minimum_detectable_effect)
            
            # Ensure p2 is valid probability
            p2 = min(max(p2, 0), 1)
            
            # Pooled proportion under null hypothesis
            p_pool = (p1 + p2) / 2
            
            # Standard error under null and alternative
            se_null = np.sqrt(2 * p_pool * (1 - p_pool))
            se_alt = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
            
            # Critical values
            z_alpha = stats.norm.ppf(1 - significance_level / 2)
            z_beta = stats.norm.ppf(power)
            
            # Sample size calculation
            n = ((z_alpha * se_null + z_beta * se_alt) / abs(p1 - p2))**2
            
            return int(np.ceil(n))
            
        except Exception as e:
            logger.error(f"Sample size calculation failed: {e}")
            return 1000  # Default fallback
            
    def _calculate_power_two_proportion(self, n_a: int, n_b: int, p_a: float, p_b: float) -> float:
        """Calculate statistical power for two-proportion test."""
        try:
            # Pooled proportion under null
            p_pool = (p_a + p_b) / 2
            
            # Standard error under null and alternative
            se_null = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
            se_alt = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
            
            # Critical value
            z_critical = stats.norm.ppf(1 - self.significance_level / 2)
            
            # Non-centrality parameter
            delta = abs(p_a - p_b) / se_alt if se_alt > 0 else 0
            
            # Power calculation
            power = 1 - stats.norm.cdf(z_critical - delta) + stats.norm.cdf(-z_critical - delta)
            
            return min(max(power, 0), 1)
            
        except Exception as e:
            logger.error(f"Power calculation failed: {e}")
            return 0.5
            
    def _calculate_power_t_test(self, n_a: int, n_b: int, effect_size: float) -> float:
        """Calculate statistical power for t-test."""
        try:
            # Degrees of freedom (simplified)
            df = n_a + n_b - 2
            
            # Non-centrality parameter
            ncp = effect_size * np.sqrt(n_a * n_b / (n_a + n_b))
            
            # Critical t-value
            t_critical = stats.t.ppf(1 - self.significance_level / 2, df)
            
            # Power using non-central t-distribution
            power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
            
            return min(max(power, 0), 1)
            
        except Exception as e:
            logger.error(f"T-test power calculation failed: {e}")
            return 0.5
            
    def _bootstrap_median_difference_ci(self, group_a: List[float], group_b: List[float],
                                       n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Bootstrap confidence interval for median difference."""
        try:
            a = np.array(group_a)
            b = np.array(group_b)
            
            bootstrap_diffs = []
            
            for _ in range(n_bootstrap):
                # Bootstrap samples
                a_boot = np.random.choice(a, size=len(a), replace=True)
                b_boot = np.random.choice(b, size=len(b), replace=True)
                
                # Calculate median difference
                diff = np.median(a_boot) - np.median(b_boot)
                bootstrap_diffs.append(diff)
                
            # Confidence interval
            ci_lower = np.percentile(bootstrap_diffs, 100 * self.significance_level / 2)
            ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - self.significance_level / 2))
            
            return (ci_lower, ci_upper)
            
        except Exception as e:
            logger.error(f"Bootstrap CI calculation failed: {e}")
            return (0.0, 0.0)
            
    def _interpret_two_proportion_test(self, p1: float, p2: float, p_value: float,
                                     effect_size: float, is_significant: bool) -> str:
        """Interpret two-proportion test results."""
        if not is_significant:
            return f"No significant difference (p={p_value:.3f}). Conversion rates: A={p1:.3f}, B={p2:.3f}"
            
        better = "B" if p2 > p1 else "A"
        relative_change = abs((p2 - p1) / p1) * 100 if p1 > 0 else 0
        
        # Effect size interpretation
        if abs(effect_size) < 0.2:
            magnitude = "small"
        elif abs(effect_size) < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"
            
        return (f"Significant difference (p={p_value:.3f}). "
                f"Model {better} performs better by {relative_change:.1f}% "
                f"({magnitude} effect size: {effect_size:.3f})")
        
    def _interpret_t_test(self, mean_a: float, mean_b: float, p_value: float,
                         effect_size: float, is_significant: bool) -> str:
        """Interpret t-test results."""
        if not is_significant:
            return f"No significant difference (p={p_value:.3f}). Means: A={mean_a:.3f}, B={mean_b:.3f}"
            
        better = "B" if mean_b > mean_a else "A"
        relative_change = abs((mean_b - mean_a) / mean_a) * 100 if mean_a != 0 else 0
        
        # Effect size interpretation (Cohen's d)
        if abs(effect_size) < 0.2:
            magnitude = "small"
        elif abs(effect_size) < 0.5:
            magnitude = "medium"
        elif abs(effect_size) < 0.8:
            magnitude = "large"
        else:
            magnitude = "very large"
            
        return (f"Significant difference (p={p_value:.3f}). "
                f"Model {better} performs better by {relative_change:.1f}% "
                f"({magnitude} effect size: {effect_size:.3f})")
        
    def _interpret_mann_whitney(self, median_a: float, median_b: float, p_value: float,
                               effect_size: float, is_significant: bool) -> str:
        """Interpret Mann-Whitney U test results."""
        if not is_significant:
            return f"No significant difference (p={p_value:.3f}). Medians: A={median_a:.3f}, B={median_b:.3f}"
            
        better = "B" if median_b > median_a else "A"
        
        return (f"Significant difference (p={p_value:.3f}). "
                f"Model {better} has significantly different distribution "
                f"(rank-biserial correlation: {effect_size:.3f})")
        
    def _create_error_result(self, test_type: TestType) -> StatisticalTestResult:
        """Create error result for failed tests."""
        return StatisticalTestResult(
            test_type=test_type,
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            power=0.0,
            is_significant=False,
            interpretation="Test failed due to insufficient data or error"
        )