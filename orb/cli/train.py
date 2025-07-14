"""
CLI interface for ORB model training and evaluation.

Provides command-line interface for hyperparameter tuning,
walk-forward evaluation, and model training workflows.
"""

import typer
from typing import Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from ..data.feature_builder import FeatureBuilder
from ..data.polygon_loader import PolygonLoader
from ..tuning.optuna_tuner import OptunaTuner
from ..evaluation.walkforward import WalkForwardRunner
from ..models.lgbm_model import LGBMModel
from ..utils.logging import setup_logging, LoggingMixin

app = typer.Typer(help="ORB Trading System - Model Training CLI")


class TrainingPipeline(LoggingMixin):
    """Main training pipeline orchestrator."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.log_info(f"Initialized training pipeline with random_state={random_state}")
    
    def prepare_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        output_dir: str,
        force_download: bool = False
    ) -> str:
        """Prepare training data by downloading and building features."""
        
        self.log_info(f"Preparing data for {symbol} from {start_date} to {end_date}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download data
        loader = PolygonLoader()
        data_file = output_path / f"{symbol}_minute_data.parquet"
        
        if force_download or not data_file.exists():
            self.log_info(f"Downloading minute data for {symbol}")
            
            # Parse dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Download data month by month
            all_data = []
            current_date = start_dt
            
            while current_date <= end_dt:
                month_data = loader.download_month(
                    symbol=symbol,
                    year=current_date.year,
                    month=current_date.month
                )
                
                if month_data is not None and not month_data.empty:
                    all_data.append(month_data)
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                
                # Build parquet file
                loader.build_parquet(combined_data, str(data_file))
                self.log_info(f"Saved minute data to {data_file}")
            else:
                raise ValueError(f"No data found for {symbol} in specified date range")
        
        # Build features
        features_file = output_path / f"{symbol}_features.parquet"
        
        if force_download or not features_file.exists():
            self.log_info(f"Building features for {symbol}")
            
            builder = FeatureBuilder()
            features_df = builder.build_features(str(data_file))
            
            # Save features
            features_df.to_parquet(features_file)
            self.log_info(f"Saved features to {features_file}")
        
        return str(features_file)
    
    def tune_hyperparameters(
        self,
        features_file: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        n_trials: int = 100,
        cv_folds: int = 5,
        output_dir: str = "models",
        experiment_name: str = "orb-hyperparameter-tuning"
    ) -> dict:
        """Tune hyperparameters using Optuna."""
        
        self.log_info(f"Starting hyperparameter tuning with {n_trials} trials")
        
        # Load features
        features_df = pd.read_parquet(features_file)
        
        # Define feature columns and target
        feature_cols = [
            'or_high', 'or_low', 'or_range', 'or_vol',
            'atr14pct', 'ema20_slope', 'vwap_dev'
        ]
        target_col = 'forward_return_label'
        
        # Validate required columns
        required_cols = feature_cols + [target_col, 'date']
        missing_cols = [col for col in required_cols if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Split data chronologically
        n_samples = len(features_df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = features_df[feature_cols].iloc[:train_end]
        y_train = features_df[target_col].iloc[:train_end]
        X_val = features_df[feature_cols].iloc[train_end:val_end]
        y_val = features_df[target_col].iloc[train_end:val_end]
        
        self.log_info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        
        # Create tuner
        tuner = OptunaTuner(
            random_state=self.random_state,
            n_trials=n_trials,
            mlflow_experiment_name=experiment_name
        )
        
        # Run tuning
        best_params, best_model = tuner.tune_lgbm(
            X_train, y_train, X_val, y_val,
            cv_folds=cv_folds
        )
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        best_model_path = output_path / "best_model"
        best_model.save(best_model_path)
        
        # Save best parameters
        params_file = output_path / "best_params.json"
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save tuning results
        tuning_results = {
            'best_params': best_params,
            'best_score': tuner.study.best_value,
            'n_trials': len(tuner.study.trials),
            'experiment_name': experiment_name
        }
        
        results_file = output_path / "tuning_results.json"
        with open(results_file, 'w') as f:
            json.dump(tuning_results, f, indent=2)
        
        self.log_info(f"Tuning completed. Best AUC: {tuner.study.best_value:.4f}")
        
        return tuning_results
    
    def run_walkforward(
        self,
        features_file: str,
        model_params: Optional[dict] = None,
        train_days: int = 250,
        validation_days: int = 25,
        retrain_gap: int = 40,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: str = "walkforward_results"
    ) -> dict:
        """Run walk-forward evaluation."""
        
        self.log_info(f"Starting walk-forward evaluation")
        
        # Load features
        features_df = pd.read_parquet(features_file)
        
        # Define feature columns and target
        feature_cols = [
            'or_high', 'or_low', 'or_range', 'or_vol',
            'atr14pct', 'ema20_slope', 'vwap_dev'
        ]
        target_col = 'forward_return_label'
        
        # Create model factory
        if model_params:
            def model_factory():
                return LGBMModel(random_state=self.random_state, **model_params)
        else:
            def model_factory():
                return LGBMModel(random_state=self.random_state)
        
        # Create walk-forward runner
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        runner = WalkForwardRunner(
            train_days=train_days,
            validation_days=validation_days,
            retrain_gap=retrain_gap,
            model_save_dir=str(output_path / "models"),
            random_state=self.random_state
        )
        
        # Run evaluation
        results = runner.run_walkforward(
            data=features_df,
            features=feature_cols,
            target=target_col,
            date_col='date',
            model_factory=model_factory,
            start_date=start_date,
            end_date=end_date
        )
        
        # Save results
        runner.save_results(str(output_path / "walkforward_results.pkl"))
        
        # Save summary
        summary_file = output_path / "walkforward_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results['performance_metrics'], f, indent=2)
        
        self.log_info(f"Walk-forward evaluation completed")
        self.log_info(f"Overall AUC: {results['performance_metrics'].get('overall_roc_auc', 0):.4f}")
        
        return results


@app.command("prepare-data")
def prepare_data(
    symbol: str = typer.Argument(..., help="Stock symbol to download"),
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    output_dir: str = typer.Option("data", help="Output directory for data files"),
    force_download: bool = typer.Option(False, help="Force re-download of data"),
    random_state: int = typer.Option(42, help="Random state for reproducibility")
):
    """Prepare training data by downloading and building features."""
    
    setup_logging()
    
    try:
        pipeline = TrainingPipeline(random_state=random_state)
        features_file = pipeline.prepare_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            force_download=force_download
        )
        
        typer.echo(f"✅ Data preparation completed: {features_file}")
        
    except Exception as e:
        typer.echo(f"❌ Data preparation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("tune")
def tune_hyperparameters(
    features_file: str = typer.Argument(..., help="Path to features parquet file"),
    train_ratio: float = typer.Option(0.7, help="Training data ratio"),
    val_ratio: float = typer.Option(0.15, help="Validation data ratio"),
    n_trials: int = typer.Option(100, help="Number of Optuna trials"),
    cv_folds: int = typer.Option(5, help="Cross-validation folds"),
    output_dir: str = typer.Option("models", help="Output directory for models"),
    experiment_name: str = typer.Option("orb-hyperparameter-tuning", help="MLflow experiment name"),
    random_state: int = typer.Option(42, help="Random state for reproducibility")
):
    """Tune hyperparameters using Optuna."""
    
    setup_logging()
    
    try:
        pipeline = TrainingPipeline(random_state=random_state)
        results = pipeline.tune_hyperparameters(
            features_file=features_file,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            n_trials=n_trials,
            cv_folds=cv_folds,
            output_dir=output_dir,
            experiment_name=experiment_name
        )
        
        typer.echo(f"✅ Hyperparameter tuning completed")
        typer.echo(f"Best AUC: {results['best_score']:.4f}")
        typer.echo(f"Results saved to: {output_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Hyperparameter tuning failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("walkforward")
def run_walkforward(
    features_file: str = typer.Argument(..., help="Path to features parquet file"),
    model_params_file: Optional[str] = typer.Option(None, help="Path to model parameters JSON file"),
    train_days: int = typer.Option(250, help="Training window size in days"),
    validation_days: int = typer.Option(25, help="Validation window size in days"),
    retrain_gap: int = typer.Option(40, help="Days between retraining"),
    start_date: Optional[str] = typer.Option(None, help="Start date for evaluation (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, help="End date for evaluation (YYYY-MM-DD)"),
    output_dir: str = typer.Option("walkforward_results", help="Output directory for results"),
    random_state: int = typer.Option(42, help="Random state for reproducibility")
):
    """Run walk-forward evaluation."""
    
    setup_logging()
    
    try:
        # Load model parameters if provided
        model_params = None
        if model_params_file:
            with open(model_params_file, 'r') as f:
                model_params = json.load(f)
        
        pipeline = TrainingPipeline(random_state=random_state)
        results = pipeline.run_walkforward(
            features_file=features_file,
            model_params=model_params,
            train_days=train_days,
            validation_days=validation_days,
            retrain_gap=retrain_gap,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir
        )
        
        typer.echo(f"✅ Walk-forward evaluation completed")
        typer.echo(f"Overall AUC: {results['performance_metrics'].get('overall_roc_auc', 0):.4f}")
        typer.echo(f"Number of windows: {results['performance_metrics'].get('n_windows', 0)}")
        typer.echo(f"Results saved to: {output_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Walk-forward evaluation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("train")
def train_full_pipeline(
    symbol: str = typer.Argument(..., help="Stock symbol to train on"),
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    output_dir: str = typer.Option("output", help="Output directory for all results"),
    n_trials: int = typer.Option(50, help="Number of Optuna trials"),
    train_days: int = typer.Option(250, help="Training window size in days"),
    validation_days: int = typer.Option(25, help="Validation window size in days"),
    retrain_gap: int = typer.Option(40, help="Days between retraining"),
    force_download: bool = typer.Option(False, help="Force re-download of data"),
    random_state: int = typer.Option(42, help="Random state for reproducibility")
):
    """Run full training pipeline: data preparation, tuning, and walk-forward evaluation."""
    
    setup_logging()
    
    try:
        pipeline = TrainingPipeline(random_state=random_state)
        
        # Step 1: Prepare data
        typer.echo("🔄 Step 1: Preparing data...")
        features_file = pipeline.prepare_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            output_dir=f"{output_dir}/data",
            force_download=force_download
        )
        typer.echo(f"✅ Data preparation completed: {features_file}")
        
        # Step 2: Tune hyperparameters
        typer.echo("🔄 Step 2: Tuning hyperparameters...")
        tuning_results = pipeline.tune_hyperparameters(
            features_file=features_file,
            n_trials=n_trials,
            output_dir=f"{output_dir}/tuning",
            experiment_name=f"orb-{symbol.lower()}-tuning"
        )
        typer.echo(f"✅ Hyperparameter tuning completed. Best AUC: {tuning_results['best_score']:.4f}")
        
        # Step 3: Run walk-forward evaluation
        typer.echo("🔄 Step 3: Running walk-forward evaluation...")
        walkforward_results = pipeline.run_walkforward(
            features_file=features_file,
            model_params=tuning_results['best_params'],
            train_days=train_days,
            validation_days=validation_days,
            retrain_gap=retrain_gap,
            output_dir=f"{output_dir}/walkforward"
        )
        typer.echo(f"✅ Walk-forward evaluation completed. Overall AUC: {walkforward_results['performance_metrics'].get('overall_roc_auc', 0):.4f}")
        
        # Summary
        typer.echo("🎉 Full training pipeline completed!")
        typer.echo(f"Symbol: {symbol}")
        typer.echo(f"Date range: {start_date} to {end_date}")
        typer.echo(f"Tuning trials: {n_trials}")
        typer.echo(f"Best hyperparameters AUC: {tuning_results['best_score']:.4f}")
        typer.echo(f"Walk-forward AUC: {walkforward_results['performance_metrics'].get('overall_roc_auc', 0):.4f}")
        typer.echo(f"Walk-forward windows: {walkforward_results['performance_metrics'].get('n_windows', 0)}")
        typer.echo(f"All results saved to: {output_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Training pipeline failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("info")
def show_info():
    """Show information about the ORB training system."""
    
    typer.echo("🎯 ORB Trading System - Model Training")
    typer.echo("=====================================")
    typer.echo("")
    typer.echo("Commands:")
    typer.echo("  prepare-data    - Download and prepare training data")
    typer.echo("  tune           - Hyperparameter tuning with Optuna")
    typer.echo("  walkforward    - Walk-forward evaluation")
    typer.echo("  train          - Full training pipeline")
    typer.echo("")
    typer.echo("Features:")
    typer.echo("  • LightGBM binary classification")
    typer.echo("  • Optuna hyperparameter optimization")
    typer.echo("  • MLflow experiment tracking")
    typer.echo("  • Time-series walk-forward evaluation")
    typer.echo("  • Opening Range Breakout (ORB) features")
    typer.echo("")
    typer.echo("Example usage:")
    typer.echo("  orb train AAPL 2023-01-01 2023-12-31 --n-trials 100")


if __name__ == "__main__":
    app() 