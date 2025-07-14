"""
Walk-forward evaluation for ORB trading models.

Implements time-series aware backtesting with proper train/validation splits,
retraining intervals, and comprehensive performance metrics.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings

from ..models.base_model import BaseModel
from ..models.lgbm_model import LGBMModel
from ..utils.logging import LoggingMixin


class WalkForwardRunner(LoggingMixin):
    """
    Walk-forward backtesting runner for ORB trading models.
    
    Features:
    - Time-series aware train/validation splits
    - Configurable retraining intervals
    - Comprehensive performance metrics
    - Model persistence between retrains
    - Out-of-sample prediction tracking
    """
    
    def __init__(
        self,
        train_days: int = 250,
        validation_days: int = 25,
        retrain_gap: int = 40,
        min_train_samples: int = 1000,
        model_save_dir: Optional[str] = None,
        random_state: int = 42
    ) -> None:
        """
        Initialize walk-forward runner.
        
        Args:
            train_days: Number of trading days for training window
            validation_days: Number of trading days for validation window
            retrain_gap: Number of days between retraining
            min_train_samples: Minimum samples required for training
            model_save_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.train_days = train_days
        self.validation_days = validation_days
        self.retrain_gap = retrain_gap
        self.min_train_samples = min_train_samples
        self.model_save_dir = Path(model_save_dir) if model_save_dir else None
        self.random_state = random_state
        
        # Results storage
        self.results: List[Dict[str, Any]] = []
        self.predictions: List[Dict[str, Any]] = []
        self.models: List[BaseModel] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, float] = {}
        self.validation_metrics: Dict[str, List[float]] = {}
        
        if self.model_save_dir:
            self.model_save_dir.mkdir(parents=True, exist_ok=True)
            
        self.log_info(
            f"Initialized WalkForwardRunner: train={train_days}d, "
            f"val={validation_days}d, retrain_gap={retrain_gap}d"
        )
    
    def run_walkforward(
        self,
        data: pd.DataFrame,
        features: List[str],
        target: str,
        date_col: str = 'date',
        model_factory: Optional[Callable[[], BaseModel]] = None,
        fit_params: Optional[Dict[str, Any]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward evaluation.
        
        Args:
            data: Time series data with features and target
            features: List of feature column names
            target: Target column name
            date_col: Date column name
            model_factory: Function to create new model instances
            fit_params: Parameters for model fitting
            start_date: Start date for evaluation (YYYY-MM-DD)
            end_date: End date for evaluation (YYYY-MM-DD)
            
        Returns:
            Dictionary with evaluation results
        """
        self.log_info("Starting walk-forward evaluation")
        
        # Validate inputs
        self._validate_inputs(data, features, target, date_col)
        
        # Prepare data
        prepared_data = self._prepare_data(data, features, target, date_col)
        
        # Filter date range if specified
        if start_date or end_date:
            prepared_data = self._filter_date_range(prepared_data, start_date, end_date)
        
        # Set default model factory if not provided
        if model_factory is None:
            model_factory = lambda: LGBMModel(random_state=self.random_state)
        
        # Set default fit parameters
        if fit_params is None:
            fit_params = {}
        
        # Generate walk-forward windows
        windows = self._generate_windows(prepared_data)
        
        self.log_info(f"Generated {len(windows)} walk-forward windows")
        
        # Reset results
        self.results = []
        self.predictions = []
        self.models = []
        
        # Run evaluation for each window
        for i, window in enumerate(windows):
            self.log_info(f"Processing window {i+1}/{len(windows)}")
            
            try:
                result = self._evaluate_window(
                    window, prepared_data, features, target, 
                    model_factory, fit_params, i
                )
                
                if result is not None:
                    self.results.append(result)
                    
            except Exception as e:
                self.log_error(f"Window {i+1} failed: {e}")
                continue
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics()
        
        self.log_info(f"Walk-forward evaluation completed. Final metrics: {final_metrics}")
        
        return {
            'results': self.results,
            'predictions': self.predictions,
            'models': self.models,
            'performance_metrics': final_metrics,
            'validation_metrics': self.validation_metrics,
            'config': {
                'train_days': self.train_days,
                'validation_days': self.validation_days,
                'retrain_gap': self.retrain_gap,
                'min_train_samples': self.min_train_samples,
                'n_windows': len(windows),
                'n_features': len(features),
                'feature_names': features
            }
        }
    
    def _validate_inputs(
        self, 
        data: pd.DataFrame, 
        features: List[str], 
        target: str, 
        date_col: str
    ) -> None:
        """Validate input parameters."""
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        if not features:
            raise ValueError("Features list cannot be empty")
        
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        if date_col not in data.columns:
            raise ValueError(f"Date column '{date_col}' not found in data")
        
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        # Check for missing values
        if data[features + [target]].isnull().any().any():
            n_missing = data[features + [target]].isnull().sum().sum()
            self.log_warning(f"Found {n_missing} missing values in features/target")
    
    def _prepare_data(
        self, 
        data: pd.DataFrame, 
        features: List[str], 
        target: str, 
        date_col: str
    ) -> pd.DataFrame:
        """Prepare data for walk-forward evaluation."""
        # Copy data and ensure date column is datetime
        prepared = data.copy()
        prepared[date_col] = pd.to_datetime(prepared[date_col])
        
        # Sort by date
        prepared = prepared.sort_values(date_col).reset_index(drop=True)
        
        # Validate target values
        unique_targets = set(prepared[target].unique())
        if not unique_targets.issubset({0, 1}):
            raise ValueError(f"Target must be binary (0/1), got: {unique_targets}")
        
        return prepared
    
    def _filter_date_range(
        self, 
        data: pd.DataFrame, 
        start_date: Optional[str], 
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Filter data by date range."""
        filtered = data.copy()
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            filtered = filtered[filtered.iloc[:, 0] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            filtered = filtered[filtered.iloc[:, 0] <= end_dt]
        
        return filtered.reset_index(drop=True)
    
    def _generate_windows(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate walk-forward windows."""
        windows = []
        dates = data.iloc[:, 0].unique()  # Assuming first column is date
        
        # Calculate required window sizes
        min_window_size = self.train_days + self.validation_days
        
        if len(dates) < min_window_size:
            raise ValueError(
                f"Not enough data for walk-forward. Need at least {min_window_size} days, "
                f"got {len(dates)} days"
            )
        
        # Generate windows
        current_start = 0
        
        while current_start + min_window_size <= len(dates):
            # Training window
            train_end = current_start + self.train_days
            train_start_date = dates[current_start]
            train_end_date = dates[min(train_end - 1, len(dates) - 1)]
            
            # Validation window
            val_start = train_end
            val_end = min(val_start + self.validation_days, len(dates))
            val_start_date = dates[val_start]
            val_end_date = dates[val_end - 1]
            
            # Check if we have enough validation data
            if val_end - val_start < self.validation_days:
                break
            
            window = {
                'window_id': len(windows),
                'train_start_date': train_start_date,
                'train_end_date': train_end_date,
                'val_start_date': val_start_date,
                'val_end_date': val_end_date,
                'train_start_idx': current_start,
                'train_end_idx': train_end,
                'val_start_idx': val_start,
                'val_end_idx': val_end
            }
            
            windows.append(window)
            
            # Move to next window
            current_start += self.retrain_gap
        
        return windows
    
    def _evaluate_window(
        self,
        window: Dict[str, Any],
        data: pd.DataFrame,
        features: List[str],
        target: str,
        model_factory: Callable[[], BaseModel],
        fit_params: Dict[str, Any],
        window_idx: int
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single walk-forward window."""
        
        # Extract training data
        train_data = data.iloc[window['train_start_idx']:window['train_end_idx']]
        X_train = train_data[features]
        y_train = train_data[target]
        
        # Extract validation data
        val_data = data.iloc[window['val_start_idx']:window['val_end_idx']]
        X_val = val_data[features]
        y_val = val_data[target]
        
        # Check minimum samples requirement
        if len(X_train) < self.min_train_samples:
            self.log_warning(f"Skipping window {window_idx}: insufficient training samples")
            return None
        
        # Check if we have both classes in training data
        if len(y_train.unique()) < 2:
            self.log_warning(f"Skipping window {window_idx}: training data has only one class")
            return None
        
        try:
            # Create and train model
            model = model_factory()
            
            # Fit the model
            model.fit(X_train, y_train, **fit_params)
            
            # Make predictions
            y_pred_proba = model.predict_proba(X_val)
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            metrics = self._calculate_window_metrics(y_val, y_pred, y_pred_proba)
            
            # Store predictions
            for idx, (true_val, pred_val, pred_proba) in enumerate(zip(y_val, y_pred, y_pred_proba[:, 1])):
                self.predictions.append({
                    'window_id': window['window_id'],
                    'date': val_data.iloc[idx].iloc[0],  # First column is date
                    'y_true': true_val,
                    'y_pred': pred_val,
                    'y_pred_proba': pred_proba
                })
            
            # Save model if directory provided
            if self.model_save_dir:
                model_path = self.model_save_dir / f"model_window_{window_idx}"
                model.save(model_path)
                self.log_info(f"Saved model for window {window_idx}")
            
            # Store model
            self.models.append(model)
            
            # Update validation metrics tracking
            for metric_name, metric_value in metrics.items():
                if metric_name not in self.validation_metrics:
                    self.validation_metrics[metric_name] = []
                self.validation_metrics[metric_name].append(metric_value)
            
            result = {
                'window_id': window['window_id'],
                'train_start_date': window['train_start_date'],
                'train_end_date': window['train_end_date'],
                'val_start_date': window['val_start_date'],
                'val_end_date': window['val_end_date'],
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'train_class_dist': y_train.value_counts().to_dict(),
                'val_class_dist': y_val.value_counts().to_dict(),
                'metrics': metrics,
                'model_info': model.get_training_summary()
            }
            
            self.log_info(
                f"Window {window_idx}: "
                f"Train: {len(X_train)} samples, "
                f"Val: {len(X_val)} samples, "
                f"AUC: {metrics.get('roc_auc', 0):.3f}"
            )
            
            return result
            
        except Exception as e:
            self.log_error(f"Error evaluating window {window_idx}: {e}")
            return None
    
    def _calculate_window_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics for a single window."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, log_loss, confusion_matrix
        )
        
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
            }
            
            # Calculate AUC if we have both classes
            if len(set(y_true)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            else:
                metrics['roc_auc'] = 0.5
                metrics['log_loss'] = 1.0
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['true_negatives'] = float(tn)
                metrics['false_positives'] = float(fp)
                metrics['false_negatives'] = float(fn)
                metrics['true_positives'] = float(tp)
                
                # Additional metrics
                if tp + fp > 0:
                    metrics['precision_positive'] = tp / (tp + fp)
                if tp + fn > 0:
                    metrics['recall_positive'] = tp / (tp + fn)
                if tn + fp > 0:
                    metrics['specificity'] = tn / (tn + fp)
            
            return metrics
            
        except Exception as e:
            self.log_error(f"Error calculating metrics: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'roc_auc': 0.5}
    
    def _calculate_final_metrics(self) -> Dict[str, float]:
        """Calculate final aggregated metrics."""
        if not self.results:
            return {}
        
        # Aggregate predictions for overall metrics
        all_predictions = pd.DataFrame(self.predictions)
        
        if all_predictions.empty:
            return {}
        
        # Calculate overall metrics
        overall_metrics = self._calculate_window_metrics(
            all_predictions['y_true'],
            all_predictions['y_pred'].values,
            np.column_stack([
                1 - all_predictions['y_pred_proba'].values,
                all_predictions['y_pred_proba'].values
            ])
        )
        
        # Calculate mean and std of window metrics
        window_metrics = {}
        for metric_name, values in self.validation_metrics.items():
            if values:
                window_metrics[f'mean_{metric_name}'] = np.mean(values)
                window_metrics[f'std_{metric_name}'] = np.std(values)
                window_metrics[f'min_{metric_name}'] = np.min(values)
                window_metrics[f'max_{metric_name}'] = np.max(values)
        
        # Combine metrics
        final_metrics = {
            'overall_accuracy': overall_metrics.get('accuracy', 0.0),
            'overall_precision': overall_metrics.get('precision', 0.0),
            'overall_recall': overall_metrics.get('recall', 0.0),
            'overall_f1_score': overall_metrics.get('f1_score', 0.0),
            'overall_roc_auc': overall_metrics.get('roc_auc', 0.5),
            'overall_log_loss': overall_metrics.get('log_loss', 1.0),
            'n_windows': len(self.results),
            'n_predictions': len(all_predictions),
            'total_train_samples': sum(r['train_samples'] for r in self.results),
            'total_val_samples': sum(r['val_samples'] for r in self.results),
            **window_metrics
        }
        
        self.performance_metrics = final_metrics
        return final_metrics
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        # Flatten results
        flattened = []
        for result in self.results:
            row = {
                'window_id': result['window_id'],
                'train_start_date': result['train_start_date'],
                'train_end_date': result['train_end_date'],
                'val_start_date': result['val_start_date'],
                'val_end_date': result['val_end_date'],
                'train_samples': result['train_samples'],
                'val_samples': result['val_samples'],
            }
            
            # Add metrics
            for metric_name, metric_value in result['metrics'].items():
                row[f'metric_{metric_name}'] = metric_value
            
            flattened.append(row)
        
        return pd.DataFrame(flattened)
    
    def get_predictions_dataframe(self) -> pd.DataFrame:
        """Get predictions as DataFrame."""
        if not self.predictions:
            return pd.DataFrame()
        
        return pd.DataFrame(self.predictions)
    
    def plot_performance(self) -> None:
        """Plot walk-forward performance metrics."""
        if not self.results:
            self.log_warning("No results to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            results_df = self.get_results_dataframe()
            
            # Plot key metrics over time
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy
            axes[0, 0].plot(results_df['val_start_date'], results_df['metric_accuracy'], 'o-')
            axes[0, 0].set_title('Accuracy Over Time')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # AUC
            axes[0, 1].plot(results_df['val_start_date'], results_df['metric_roc_auc'], 'o-')
            axes[0, 1].set_title('ROC AUC Over Time')
            axes[0, 1].set_ylabel('ROC AUC')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Precision/Recall
            axes[1, 0].plot(results_df['val_start_date'], results_df['metric_precision'], 'o-', label='Precision')
            axes[1, 0].plot(results_df['val_start_date'], results_df['metric_recall'], 'o-', label='Recall')
            axes[1, 0].set_title('Precision/Recall Over Time')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # F1 Score
            axes[1, 1].plot(results_df['val_start_date'], results_df['metric_f1_score'], 'o-')
            axes[1, 1].set_title('F1 Score Over Time')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.log_warning("matplotlib not available for plotting")
        except Exception as e:
            self.log_error(f"Plotting failed: {e}")
    
    def save_results(self, filepath: str) -> None:
        """Save walk-forward results to disk."""
        import joblib
        
        results_data = {
            'results': self.results,
            'predictions': self.predictions,
            'performance_metrics': self.performance_metrics,
            'validation_metrics': self.validation_metrics,
            'config': {
                'train_days': self.train_days,
                'validation_days': self.validation_days,
                'retrain_gap': self.retrain_gap,
                'min_train_samples': self.min_train_samples,
                'random_state': self.random_state
            }
        }
        
        joblib.dump(results_data, filepath)
        self.log_info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """Load walk-forward results from disk."""
        import joblib
        
        results_data = joblib.load(filepath)
        self.results = results_data['results']
        self.predictions = results_data['predictions']
        self.performance_metrics = results_data['performance_metrics']
        self.validation_metrics = results_data['validation_metrics']
        
        self.log_info(f"Results loaded from {filepath}") 