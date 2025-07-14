"""
Optuna hyperparameter tuning for ORB trading models.

Provides automated hyperparameter optimization with MLflow experiment
tracking and time-series aware cross-validation.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import mlflow
import mlflow.lightgbm
from datetime import datetime
import warnings

from ..models.lgbm_model import LGBMModel
from ..utils.logging import LoggingMixin


class OptunaTuner(LoggingMixin):
    """
    Optuna-based hyperparameter tuner for ORB models.
    
    Features:
    - Bayesian optimization with TPE sampler
    - MLflow integration for experiment tracking
    - Time-series aware cross-validation
    - Early stopping and pruning
    """
    
    def __init__(
        self,
        random_state: int = 42,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        pruner_patience: int = 10,
        mlflow_experiment_name: str = "orb-hyperparameter-tuning",
        mlflow_tracking_uri: Optional[str] = None
    ) -> None:
        """
        Initialize Optuna tuner.
        
        Args:
            random_state: Random seed for reproducibility
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            pruner_patience: Patience for median pruner
            mlflow_experiment_name: MLflow experiment name
            mlflow_tracking_uri: MLflow tracking URI (None for local)
        """
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout = timeout
        self.pruner_patience = pruner_patience
        self.mlflow_experiment_name = mlflow_experiment_name
        
        # Setup MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        try:
            experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(mlflow_experiment_name)
                self.log_info(f"Created MLflow experiment: {mlflow_experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                self.log_info(f"Using existing MLflow experiment: {mlflow_experiment_name}")
            
            mlflow.set_experiment(experiment_id=experiment_id)
            
        except Exception as e:
            self.log_warning(f"MLflow setup failed: {e}")
        
        # Initialize study components
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_model: Optional[LGBMModel] = None
        
        self.log_info(f"Initialized OptunaTuner with {n_trials} trials")
    
    def tune_lgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cv_folds: int = 5,
        early_stopping_rounds: int = 50,
        param_space: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], LGBMModel]:
        """
        Tune LightGBM hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features  
            y_val: Validation targets
            cv_folds: Number of cross-validation folds
            early_stopping_rounds: Early stopping patience
            param_space: Custom parameter space (None for default)
            
        Returns:
            Tuple of (best_parameters, best_model)
        """
        self.log_info(f"Starting hyperparameter tuning with {self.n_trials} trials")
        
        # Store data for objective function
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.param_space = param_space or self._get_default_param_space()
        
        # Create study
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=self.pruner_patience)
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"lgbm_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Run optimization
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                self.study.optimize(
                    self._objective,
                    n_trials=self.n_trials,
                    timeout=self.timeout,
                    show_progress_bar=True
                )
            
            # Get best results
            self.best_params = self.study.best_params
            
            # Train final model with best parameters
            self.best_model = self._train_final_model()
            
            # Log final results to MLflow
            self._log_final_results()
            
            self.log_info(
                f"Optimization completed. Best score: {self.study.best_value:.4f}"
            )
            
            return self.best_params, self.best_model
            
        except Exception as e:
            self.log_error(f"Optimization failed: {e}")
            raise
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value (validation score)
        """
        # Sample hyperparameters
        params = self._sample_parameters(trial)
        
        # Start MLflow run for this trial
        with mlflow.start_run(nested=True):
            try:
                # Log trial parameters
                mlflow.log_params(params)
                mlflow.log_param("trial_number", trial.number)
                
                # Create and train model
                model = LGBMModel(random_state=self.random_state, **params)
                
                # Time-series cross-validation
                cv_scores = self._time_series_cv(model, trial)
                
                # Calculate mean score
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                # Log metrics
                mlflow.log_metric("cv_mean_score", mean_score)
                mlflow.log_metric("cv_std_score", std_score)
                mlflow.log_metric("cv_scores", cv_scores)
                
                # Final validation
                model.fit(
                    self.X_train, self.y_train,
                    X_val=self.X_val, y_val=self.y_val,
                    early_stopping_rounds=self.early_stopping_rounds
                )
                
                val_score = model.validation_metrics.get('roc_auc', 0.5)
                mlflow.log_metric("final_val_score", val_score)
                
                # Log model
                mlflow.lightgbm.log_model(
                    model.model, 
                    f"lgbm_trial_{trial.number}",
                    input_example=self.X_train.head()
                )
                
                self.log_info(
                    f"Trial {trial.number}: CV={mean_score:.4f}±{std_score:.4f}, "
                    f"Val={val_score:.4f}"
                )
                
                return val_score
                
            except Exception as e:
                self.log_error(f"Trial {trial.number} failed: {e}")
                return 0.0
    
    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters for the trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        
        for param_name, param_config in self.param_space.items():
            param_type = param_config['type']
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', 1)
                )
            elif param_type == 'float':
                if param_config.get('log', False):
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        step=param_config.get('step')
                    )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        return params
    
    def _time_series_cv(
        self, 
        model: LGBMModel, 
        trial: optuna.Trial
    ) -> List[float]:
        """
        Perform time-series aware cross-validation.
        
        Args:
            model: Model to evaluate
            trial: Optuna trial for pruning
            
        Returns:
            List of fold scores
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            # Split data
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]
            
            # Train model
            fold_model = LGBMModel(random_state=self.random_state, **model.get_model_params())
            fold_model.fit(
                X_fold_train, y_fold_train,
                X_val=X_fold_val, y_val=y_fold_val,
                early_stopping_rounds=self.early_stopping_rounds
            )
            
            # Predict and score
            y_pred_proba = fold_model.predict_proba(X_fold_val)
            
            if len(set(y_fold_val)) > 1:  # Need both classes for AUC
                score = roc_auc_score(y_fold_val, y_pred_proba[:, 1])
            else:
                score = 0.5  # Default score if only one class
            
            scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(score, fold)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return scores
    
    def _get_default_param_space(self) -> Dict[str, Any]:
        """
        Get default parameter space for LightGBM.
        
        Returns:
            Dictionary defining parameter search space
        """
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500, 'step': 50},
            'max_depth': {'type': 'int', 'low': 3, 'high': 12},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'num_leaves': {'type': 'int', 'low': 10, 'high': 200},
            'min_child_samples': {'type': 'int', 'low': 10, 'high': 100},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0, 'step': 0.1},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0, 'step': 0.1},
            'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 10.0},
            'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 10.0},
            'min_split_gain': {'type': 'float', 'low': 0.0, 'high': 1.0},
        }
    
    def _train_final_model(self) -> LGBMModel:
        """
        Train final model with best parameters.
        
        Returns:
            Trained model with best parameters
        """
        self.log_info("Training final model with best parameters")
        
        model = LGBMModel(random_state=self.random_state, **self.best_params)
        model.fit(
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            early_stopping_rounds=self.early_stopping_rounds
        )
        
        return model
    
    def _log_final_results(self) -> None:
        """Log final optimization results to MLflow."""
        with mlflow.start_run():
            # Log best parameters
            mlflow.log_params(self.best_params)
            
            # Log study statistics
            mlflow.log_metric("best_value", self.study.best_value)
            mlflow.log_metric("n_trials", len(self.study.trials))
            mlflow.log_metric("n_complete_trials", len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
            mlflow.log_metric("n_pruned_trials", len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]))
            
            # Log final model metrics
            if self.best_model:
                for metric_name, metric_value in self.best_model.training_metrics.items():
                    mlflow.log_metric(f"final_train_{metric_name}", metric_value)
                
                for metric_name, metric_value in self.best_model.validation_metrics.items():
                    mlflow.log_metric(f"final_val_{metric_name}", metric_value)
                
                # Log final model
                mlflow.lightgbm.log_model(
                    self.best_model.model,
                    "best_model",
                    input_example=self.X_train.head()
                )
                
                # Log feature importance
                importance = self.best_model.get_feature_importance()
                if importance is not None:
                    importance_dict = importance.to_dict()
                    mlflow.log_dict(importance_dict, "feature_importance.json")
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.
        
        Returns:
            DataFrame with trial history
        """
        if self.study is None:
            return pd.DataFrame()
        
        trials_data = []
        for trial in self.study.trials:
            trial_data = {
                'trial_number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete,
            }
            
            # Add parameters
            if trial.params:
                trial_data.update(trial.params)
            
            trials_data.append(trial_data)
        
        return pd.DataFrame(trials_data)
    
    def plot_optimization_history(self) -> None:
        """Plot optimization history."""
        if self.study is None:
            self.log_warning("No study available for plotting")
            return
        
        try:
            import optuna.visualization as vis
            import plotly.graph_objects as go
            
            # Plot optimization history
            fig = vis.plot_optimization_history(self.study)
            fig.show()
            
            # Plot parameter importances
            fig = vis.plot_param_importances(self.study)
            fig.show()
            
        except ImportError:
            self.log_warning("Plotly not available for visualization")
        except Exception as e:
            self.log_error(f"Plotting failed: {e}")
    
    def save_study(self, filepath: str) -> None:
        """
        Save study to disk.
        
        Args:
            filepath: Path to save study
        """
        if self.study is None:
            self.log_warning("No study to save")
            return
        
        try:
            import joblib
            
            study_data = {
                'study': self.study,
                'best_params': self.best_params,
                'optimization_config': {
                    'random_state': self.random_state,
                    'n_trials': self.n_trials,
                    'timeout': self.timeout,
                    'pruner_patience': self.pruner_patience,
                    'mlflow_experiment_name': self.mlflow_experiment_name
                }
            }
            
            joblib.dump(study_data, filepath)
            self.log_info(f"Study saved to {filepath}")
            
        except Exception as e:
            self.log_error(f"Failed to save study: {e}")
    
    def load_study(self, filepath: str) -> None:
        """
        Load study from disk.
        
        Args:
            filepath: Path to load study from
        """
        try:
            import joblib
            
            study_data = joblib.load(filepath)
            self.study = study_data['study']
            self.best_params = study_data['best_params']
            
            self.log_info(f"Study loaded from {filepath}")
            
        except Exception as e:
            self.log_error(f"Failed to load study: {e}")


def tune_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    cv_folds: int = 5,
    random_state: int = 42,
    mlflow_experiment_name: str = "orb-lgbm-tuning"
) -> Tuple[Dict[str, Any], LGBMModel]:
    """
    Convenience function for LightGBM hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_trials: Number of optimization trials
        cv_folds: Number of CV folds
        random_state: Random seed
        mlflow_experiment_name: MLflow experiment name
        
    Returns:
        Tuple of (best_parameters, best_model)
    """
    tuner = OptunaTuner(
        random_state=random_state,
        n_trials=n_trials,
        mlflow_experiment_name=mlflow_experiment_name
    )
    
    return tuner.tune_lgbm(
        X_train, y_train, X_val, y_val,
        cv_folds=cv_folds
    ) 