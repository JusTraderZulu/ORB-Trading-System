"""
Abstract base model for ORB trading system.

Defines the interface that all models must implement for consistent
training, prediction, and persistence operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

from ..utils.logging import LoggingMixin


class BaseModel(ABC, LoggingMixin):
    """
    Abstract base class for all ORB trading models.
    
    Provides consistent interface for model training, prediction,
    and persistence operations across different ML algorithms.
    """
    
    def __init__(self, random_state: int = 42, **kwargs: Any) -> None:
        """
        Initialize base model.
        
        Args:
            random_state: Random seed for reproducibility
            **kwargs: Additional model-specific parameters
        """
        self.random_state = random_state
        self.model_params = kwargs
        self.is_fitted = False
        self.feature_names: Optional[list] = None
        self.training_info: Dict[str, Any] = {}
        
        # Training metrics
        self.training_metrics: Dict[str, float] = {}
        self.validation_metrics: Dict[str, float] = {}
        
        self.log_info(f"Initialized {self.__class__.__name__} with random_state={random_state}")
    
    @abstractmethod
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of shape (n_samples, n_classes) with predicted probabilities
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels (default implementation using predict_proba).
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, filepath: Union[str, Path]) -> 'BaseModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        pass
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores (if supported by the model).
        
        Returns:
            Series with feature names as index and importance scores as values,
            or None if not supported
        """
        return None
    
    def validate_inputs(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Validate input data format and consistency.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("X cannot be empty")
        
        if y is not None:
            if not isinstance(y, pd.Series):
                raise ValueError("y must be a pandas Series")
            
            if len(X) != len(y):
                raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
            
            # Check for valid binary classification targets
            unique_targets = set(y.unique())
            if not unique_targets.issubset({0, 1}):
                raise ValueError(f"y must contain only 0 and 1, got: {unique_targets}")
        
        # Check for missing values
        if X.isnull().any().any():
            n_missing = X.isnull().sum().sum()
            self.log_warning(f"Found {n_missing} missing values in features")
        
        # Store feature names for consistency checking
        if self.feature_names is None:
            self.feature_names = list(X.columns)
        elif list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature names mismatch. Expected: {self.feature_names}, "
                f"got: {list(X.columns)}"
            )
    
    def _save_metadata(self, filepath: Path) -> None:
        """
        Save model metadata to accompany the main model file.
        
        Args:
            filepath: Base filepath for saving metadata
        """
        metadata = {
            'model_class': self.__class__.__name__,
            'random_state': self.random_state,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'training_info': self.training_info,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'is_fitted': self.is_fitted,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = filepath.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.log_info(f"Saved metadata to {metadata_path}")
    
    def _load_metadata(self, filepath: Path) -> Dict[str, Any]:
        """
        Load model metadata from disk.
        
        Args:
            filepath: Base filepath for loading metadata
            
        Returns:
            Dictionary with model metadata
        """
        metadata_path = filepath.with_suffix('.metadata.json')
        
        if not metadata_path.exists():
            self.log_warning(f"Metadata file not found: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Restore attributes from metadata
        self.random_state = metadata.get('random_state', 42)
        self.model_params = metadata.get('model_params', {})
        self.feature_names = metadata.get('feature_names')
        self.training_info = metadata.get('training_info', {})
        self.training_metrics = metadata.get('training_metrics', {})
        self.validation_metrics = metadata.get('validation_metrics', {})
        self.is_fitted = metadata.get('is_fitted', False)
        
        self.log_info(f"Loaded metadata from {metadata_path}")
        return metadata
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model's training information.
        
        Returns:
            Dictionary with training summary
        """
        return {
            'model_class': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'training_info': self.training_info,
            'random_state': self.random_state
        }
    
    def __str__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"fitted={self.is_fitted}, "
            f"features={len(self.feature_names) if self.feature_names else 0}, "
            f"random_state={self.random_state})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of the model."""
        return self.__str__()


class ModelEvaluator:
    """
    Utility class for evaluating model performance.
    """
    
    @staticmethod
    def evaluate_binary_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate binary classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            try:
                # Handle different probability formats
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    prob_pos = y_prob[:, 1]
                else:
                    prob_pos = y_prob.ravel()
                    
                metrics['auc'] = roc_auc_score(y_true, prob_pos)
            except ValueError:
                metrics['auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            })
        
        return metrics
    
    @staticmethod
    def calculate_trading_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate trading-specific performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            returns: Forward returns
            
        Returns:
            Dictionary with trading metrics
        """
        # Strategy returns (only take positions when predicted positive)
        strategy_returns = returns * y_pred
        
        # Basic statistics
        total_trades = np.sum(y_pred)
        winning_trades = np.sum((y_pred == 1) & (returns > 0))
        losing_trades = np.sum((y_pred == 1) & (returns < 0))
        
        metrics = {
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        }
        
        # Return statistics
        if len(strategy_returns) > 0:
            metrics.update({
                'total_return': np.sum(strategy_returns),
                'mean_return': np.mean(strategy_returns),
                'std_return': np.std(strategy_returns),
                'sharpe_ratio': np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0,
                'max_drawdown': ModelEvaluator._calculate_max_drawdown(strategy_returns)
            })
        
        return metrics
    
    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return float(np.min(drawdown)) 