#!/usr/bin/env python3
"""Command-line interface for A/B testing management."""
import click
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ab_testing.experiment_manager import (
    ExperimentManager, ExperimentConfig, ExperimentStatus, TrafficAllocation
)
from ab_testing.traffic_splitter import TrafficSplitter
from ab_testing.statistical_analysis import StatisticalAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """A/B Testing CLI for MLOps Pipeline."""
    pass


@cli.command()
@click.option('--name', required=True, help='Experiment name')
@click.option('--description', help='Experiment description')
@click.option('--model-a', required=True, help='Model A name')
@click.option('--model-a-version', required=True, help='Model A version')
@click.option('--model-b', required=True, help='Model B name')
@click.option('--model-b-version', required=True, help='Model B version')
@click.option('--traffic-split', type=float, default=0.5, help='Traffic split for model A (0.0-1.0)')
@click.option('--allocation-method', type=click.Choice(['random', 'user_hash', 'feature_hash']), 
              default='user_hash', help='Traffic allocation method')
@click.option('--duration-days', type=int, default=14, help='Experiment duration in days')
@click.option('--min-sample-size', type=int, default=1000, help='Minimum sample size per group')
@click.option('--significance-level', type=float, default=0.05, help='Statistical significance level')
@click.option('--success-metrics', help='Comma-separated list of success metrics')
@click.option('--created-by', default='cli', help='Creator of the experiment')
def create(name: str, description: str, model_a: str, model_a_version: str,
           model_b: str, model_b_version: str, traffic_split: float,
           allocation_method: str, duration_days: int, min_sample_size: int,
           significance_level: float, success_metrics: str, created_by: str):
    """Create a new A/B testing experiment."""
    try:
        # Parse success metrics
        metrics = [m.strip() for m in success_metrics.split(',')] if success_metrics else ['conversion_rate']
        
        # Create experiment configuration
        config = ExperimentConfig(
            experiment_id="",  # Will be generated
            name=name,
            description=description or f"A/B test: {model_a} vs {model_b}",
            model_a_name=model_a,
            model_a_version=model_a_version,
            model_b_name=model_b,
            model_b_version=model_b_version,
            traffic_split=traffic_split,
            allocation_method=TrafficAllocation(allocation_method.upper()),
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=duration_days),
            minimum_sample_size=min_sample_size,
            significance_level=significance_level,
            success_metrics=metrics,
            status=ExperimentStatus.DRAFT,
            created_by=created_by,
            metadata={}
        )
        
        # Create experiment
        manager = ExperimentManager()
        experiment_id = manager.create_experiment(config)
        
        click.echo(f"✅ Created experiment: {experiment_id}")
        click.echo(f"   Name: {name}")
        click.echo(f"   Models: {model_a}:{model_a_version} vs {model_b}:{model_b_version}")
        click.echo(f"   Traffic Split: {traffic_split:.0%} / {1-traffic_split:.0%}")
        click.echo(f"   Duration: {duration_days} days")
        
    except Exception as e:
        click.echo(f"❌ Failed to create experiment: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('experiment_id')
def start(experiment_id: str):
    """Start an experiment."""
    try:
        manager = ExperimentManager()
        success = manager.update_experiment_status(experiment_id, ExperimentStatus.ACTIVE)
        
        if success:
            click.echo(f"✅ Started experiment: {experiment_id}")
            
            # Setup traffic splitting
            splitter = TrafficSplitter()
            experiment = manager.get_experiment(experiment_id)
            
            if experiment:
                splitter.setup_traffic_split(
                    experiment_id,
                    experiment.model_a_name,
                    experiment.model_a_version,
                    experiment.model_b_name,
                    experiment.model_b_version,
                    experiment.traffic_split
                )
                click.echo("   Traffic splitting configured")
        else:
            click.echo(f"❌ Failed to start experiment: {experiment_id}", err=True)
            
    except Exception as e:
        click.echo(f"❌ Error starting experiment: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('experiment_id')
def pause(experiment_id: str):
    """Pause an experiment."""
    try:
        splitter = TrafficSplitter()
        success = splitter.pause_experiment(experiment_id)
        
        if success:
            click.echo(f"⏸️  Paused experiment: {experiment_id}")
            click.echo("   All traffic routed to control group (Model A)")
        else:
            click.echo(f"❌ Failed to pause experiment: {experiment_id}", err=True)
            
    except Exception as e:
        click.echo(f"❌ Error pausing experiment: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('experiment_id')
def resume(experiment_id: str):
    """Resume a paused experiment."""
    try:
        splitter = TrafficSplitter()
        success = splitter.resume_experiment(experiment_id)
        
        if success:
            click.echo(f"▶️  Resumed experiment: {experiment_id}")
        else:
            click.echo(f"❌ Failed to resume experiment: {experiment_id}", err=True)
            
    except Exception as e:
        click.echo(f"❌ Error resuming experiment: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('experiment_id')
def stop(experiment_id: str):
    """Stop an experiment."""
    try:
        manager = ExperimentManager()
        success = manager.update_experiment_status(experiment_id, ExperimentStatus.COMPLETED)
        
        if success:
            click.echo(f"🛑 Stopped experiment: {experiment_id}")
        else:
            click.echo(f"❌ Failed to stop experiment: {experiment_id}", err=True)
            
    except Exception as e:
        click.echo(f"❌ Error stopping experiment: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--status', type=click.Choice(['draft', 'active', 'paused', 'completed', 'cancelled']),
              help='Filter by status')
def list(status: str):
    """List experiments."""
    try:
        manager = ExperimentManager()
        
        # Convert status to enum if provided
        status_filter = ExperimentStatus(status.upper()) if status else None
        experiments = manager.list_experiments(status_filter)
        
        if not experiments:
            click.echo("No experiments found.")
            return
            
        # Display experiments in table format
        click.echo(f"{'ID':<36} {'Name':<20} {'Status':<10} {'Models':<30} {'Split':<8} {'Created':<12}")
        click.echo("=" * 120)
        
        for exp in experiments:
            models_str = f"{exp.model_a_name} vs {exp.model_b_name}"
            split_str = f"{exp.traffic_split:.0%}/{1-exp.traffic_split:.0%}"
            created_str = exp.start_date.strftime("%Y-%m-%d")
            
            click.echo(f"{exp.experiment_id:<36} {exp.name[:20]:<20} {exp.status.value:<10} "
                      f"{models_str[:30]:<30} {split_str:<8} {created_str:<12}")
                      
    except Exception as e:
        click.echo(f"❌ Error listing experiments: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('experiment_id')
def status(experiment_id: str):
    """Get experiment status."""
    try:
        splitter = TrafficSplitter()
        status = splitter.get_experiment_status(experiment_id)
        
        if 'error' in status:
            click.echo(f"❌ {status['error']}", err=True)
            return
            
        click.echo(f"📊 Experiment Status: {experiment_id}")
        click.echo(f"   Status: {status['status']}")
        click.echo(f"   Duration: {status['duration_days']} days")
        click.echo(f"   Total Assignments: {status['total_assignments']}")
        click.echo()
        click.echo("   Models:")
        click.echo(f"     A: {status['models']['A']}")
        click.echo(f"     B: {status['models']['B']}")
        click.echo()
        click.echo("   Traffic Split:")
        click.echo(f"     Configured: A={status['configured_split']:.0%}, B={1-status['configured_split']:.0%}")
        click.echo(f"     Actual: A={status['actual_split']['A']:.0%}, B={status['actual_split']['B']:.0%}")
        click.echo()
        click.echo("   Sample Sizes:")
        click.echo(f"     A: {status['allocation']['A']}")
        click.echo(f"     B: {status['allocation']['B']}")
        
        if status['is_paused']:
            click.echo("\n   ⚠️  Experiment is currently PAUSED")
            
    except Exception as e:
        click.echo(f"❌ Error getting experiment status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('experiment_id')
@click.option('--output', '-o', help='Output file for results (JSON)')
def analyze(experiment_id: str, output: str):
    """Analyze experiment results."""
    try:
        manager = ExperimentManager()
        result = manager.analyze_experiment(experiment_id)
        
        click.echo(f"📈 Analysis Results: {experiment_id}")
        click.echo("=" * 50)
        
        # Basic metrics
        click.echo(f"Sample Sizes: A={result.sample_size_a}, B={result.sample_size_b}")
        click.echo(f"Test Duration: {result.test_duration_days} days")
        click.echo()
        
        # Model A metrics
        click.echo("Model A Metrics:")
        for metric, value in result.model_a_metrics.items():
            click.echo(f"  {metric}: {value:.4f}")
        click.echo()
        
        # Model B metrics  
        click.echo("Model B Metrics:")
        for metric, value in result.model_b_metrics.items():
            click.echo(f"  {metric}: {value:.4f}")
        click.echo()
        
        # Statistical tests
        click.echo("Statistical Tests:")
        for metric, test_result in result.statistical_tests.items():
            sig_symbol = "✅" if test_result.get('significant', False) else "❌"
            click.echo(f"  {metric}: p-value={test_result.get('p_value', 0):.4f} {sig_symbol}")
        click.echo()
        
        # Winner and recommendation
        if result.winner:
            click.echo(f"🏆 Winner: Model {result.winner}")
        else:
            click.echo("🤝 No clear winner")
            
        click.echo(f"📋 Recommendation: {result.recommendation}")
        
        # Save to file if requested
        if output:
            result_dict = {
                "experiment_id": result.experiment_id,
                "model_a_metrics": result.model_a_metrics,
                "model_b_metrics": result.model_b_metrics,
                "sample_size_a": result.sample_size_a,
                "sample_size_b": result.sample_size_b,
                "statistical_tests": result.statistical_tests,
                "confidence_intervals": result.confidence_intervals,
                "effect_sizes": result.effect_sizes,
                "statistical_significance": result.statistical_significance,
                "practical_significance": result.practical_significance,
                "winner": result.winner,
                "recommendation": result.recommendation,
                "test_duration_days": result.test_duration_days,
                "analyzed_at": result.analyzed_at.isoformat()
            }
            
            with open(output, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
                
            click.echo(f"\n💾 Results saved to: {output}")
            
    except Exception as e:
        click.echo(f"❌ Error analyzing experiment: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--baseline-rate', type=float, required=True, help='Baseline conversion rate')
@click.option('--mde', type=float, required=True, help='Minimum detectable effect (relative)')
@click.option('--power', type=float, default=0.8, help='Statistical power')
@click.option('--significance-level', type=float, default=0.05, help='Significance level (alpha)')
def sample_size(baseline_rate: float, mde: float, power: float, significance_level: float):
    """Calculate required sample size for an experiment."""
    try:
        analyzer = StatisticalAnalyzer(significance_level, power)
        sample_size_per_group = analyzer.calculate_sample_size(
            baseline_rate, mde, power, significance_level
        )
        
        total_sample_size = sample_size_per_group * 2
        
        click.echo("📊 Sample Size Calculation")
        click.echo("=" * 30)
        click.echo(f"Baseline Rate: {baseline_rate:.1%}")
        click.echo(f"Minimum Detectable Effect: {mde:.1%}")
        click.echo(f"Statistical Power: {power:.1%}")
        click.echo(f"Significance Level: {significance_level:.1%}")
        click.echo()
        click.echo(f"Required Sample Size per Group: {sample_size_per_group:,}")
        click.echo(f"Total Sample Size: {total_sample_size:,}")
        
        # Estimate duration based on daily traffic
        click.echo()
        click.echo("Duration Estimates (based on daily traffic):")
        for daily_traffic in [100, 500, 1000, 5000, 10000]:
            days = total_sample_size / daily_traffic
            click.echo(f"  {daily_traffic:,} visitors/day: {days:.1f} days")
            
    except Exception as e:
        click.echo(f"❌ Error calculating sample size: {e}", err=True)
        sys.exit(1)


@cli.command()
def cleanup():
    """Clean up expired experiment assignments."""
    try:
        splitter = TrafficSplitter()
        cleaned_count = splitter.cleanup_expired_assignments()
        
        click.echo(f"🧹 Cleaned up {cleaned_count} expired assignments")
        
    except Exception as e:
        click.echo(f"❌ Error during cleanup: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()