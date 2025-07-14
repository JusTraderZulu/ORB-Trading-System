"""
LightGBM model implementation for ORB trading system.

Wraps lightgbm.LGBMClassifier with the BaseModel interface for
binary classification of opening range breakout patterns.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("lightgbm is required but not installed. Run: pip install lightgbm")

from .base_model import BaseModel


class LGBMModel(BaseModel):
    """
    LightGBM implementation for ORB binary classification.
    
    Wraps lightgbm.LGBMClassifier with consistent ORB model interface
    including training, prediction, and persistence operations.
    """
    
    def __init__(
        self, 
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        class_weight: Optional[Union[str, Dict]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize LightGBM model.
        
        Args:
            random_state: Random seed for reproducibility
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Boosting learning rate
            num_leaves: Maximum number of leaves in one tree
            min_split_gain: Minimum loss reduction to make further partition
            min_child_weight: Minimum sum of instance weight needed in a child
            min_child_samples: Minimum number of data points in a leaf
            subsample: Subsample ratio of training instances
            subsample_freq: Frequency of subsample (0 = disabled)
            colsample_bytree: Subsample ratio of columns when building each tree
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            class_weight: Weights for classes ('balanced' or dict)
            **kwargs: Additional LightGBM parameters
        """
        # Store parameters for model creation
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_split_gain': min_split_gain,
            'min_child_weight': min_child_weight,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'subsample_freq': subsample_freq,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'class_weight': class_weight,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,  # Suppress output
            'force_row_wise': True,  # Avoid warning
            **kwargs
        }
        
        super().__init__(random_state=random_state, **model_params)
        
        # Initialize the LightGBM model
        self.model: Optional[lgb.LGBMClassifier] = None
        self._create_model()
        
    def _create_model(self) -> None:
        """Create a new LightGBM classifier with current parameters."""
        # Add random_state back for LightGBM initialization
        lgb_params = self.model_params.copy()
        lgb_params['random_state'] = self.random_state
        self.model = lgb.LGBMClassifier(**lgb_params)
        self.log_info("Created new LightGBM classifier")
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs: Any
    ) -> 'LGBMModel':
        """
        Train the LightGBM model on the provided data.
        
        Args:
            X: Training features
            y: Training targets (0/1 for ORB signals)
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets
            early_stopping_rounds: Stop if no improvement for this many rounds
            **kwargs: Additional LightGBM fit parameters
            
        Returns:
            Self for method chaining
        """
        self.log_info(f"Training LightGBM model on {len(X)} samples")
        
        # Validate inputs
        self.validate_inputs(X, y)
        if X_val is not None:
            self.validate_inputs(X_val, y_val)
        
        # Prepare training arguments
        fit_params = kwargs.copy()
        
        # Set up validation set for early stopping if provided
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['eval_names'] = ['validation']
            
            if early_stopping_rounds is not None:
                fit_params['callbacks'] = [
                    lgb.early_stopping(early_stopping_rounds, verbose=False)
                ]
        
        # Train the model
        try:
            self.model.fit(X, y, **fit_params)
            self.is_fitted = True
            
            # Store training information
            self.training_info = {
                'n_samples': len(X),
                'n_features': len(X.columns),
                'feature_names': list(X.columns),
                'class_distribution': y.value_counts().to_dict(),
                'best_iteration': getattr(self.model, 'best_iteration_', self.model.n_estimators),
                'early_stopping_used': early_stopping_rounds is not None
            }
            
            # Calculate training metrics
            y_pred_proba = self.predict_proba(X)
            y_pred = self.predict(X)
            
            self.training_metrics = self._calculate_metrics(y, y_pred, y_pred_proba)
            
            # Calculate validation metrics if validation set provided
            if X_val is not None and y_val is not None:
                y_val_pred_proba = self.predict_proba(X_val)
                y_val_pred = self.predict(X_val)
                self.validation_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
            
            self.log_info(
                f"Training completed. "
                f"Training accuracy: {self.training_metrics.get('accuracy', 0):.3f}"
            )
            
            if self.validation_metrics:
                self.log_info(
                    f"Validation accuracy: {self.validation_metrics.get('accuracy', 0):.3f}"
                )
            
        except Exception as e:
            self.log_error(f"Training failed: {e}")
            raise
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for ORB signals.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for [no_signal, signal]
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_inputs(X)
        
        try:
            probabilities = self.model.predict_proba(X)
            return probabilities
            
        except Exception as e:
            self.log_error(f"Prediction failed: {e}")
            raise
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained LightGBM model to disk.
        
        Args:
            filepath: Path to save the model (without extension)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save the LightGBM model
            model_path = filepath.with_suffix('.pkl')
            joblib.dump(self.model, model_path)
            
            # Save metadata
            self._save_metadata(filepath)
            
            self.log_info(f"Model saved to {model_path}")
            
        except Exception as e:
            self.log_error(f"Failed to save model: {e}")
            raise
    
    def load(self, filepath: Union[str, Path]) -> 'LGBMModel':
        """
        Load a trained LightGBM model from disk.
        
        Args:
            filepath: Path to load the model from (without extension)
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        model_path = filepath.with_suffix('.pkl')
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load the LightGBM model
            self.model = joblib.load(model_path)
            
            # Load metadata
            self._load_metadata(filepath)
            
            self.log_info(f"Model loaded from {model_path}")
            
        except Exception as e:
            self.log_error(f"Failed to load model: {e}")
            raise
        
        return self
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores from the trained LightGBM model.
        
        Returns:
            Series with feature names as index and importance scores as values
        """
        if not self.is_fitted or self.model is None:
            self.log_warning("Model not fitted, cannot get feature importance")
            return None
        
        if self.feature_names is None:
            self.log_warning("Feature names not available")
            return None
        
        try:
            importances = self.model.feature_importances_
            # Use LightGBM's feature_name() method to get the correct feature order
            feature_names = self.model.booster_.feature_name()
            return pd.Series(
                data=importances,
                index=feature_names,
                name='importance'
            ).sort_values(ascending=False)
            
        except Exception as e:
            self.log_error(f"Failed to get feature importance: {e}")
            return None
    
    def _calculate_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with metric names and values
        """
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score, 
                roc_auc_score, log_loss
            )
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
            }
            
            # Add AUC if we have both classes
            if len(set(y_true)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            
            return metrics
            
        except Exception as e:
            self.log_warning(f"Failed to calculate some metrics: {e}")
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.5,
                'log_loss': 1.0
            }
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dictionary with model parameters
        """
        return self.model_params.copy()
    
    def set_model_params(self, **params: Any) -> 'LGBMModel':
        """
        Set model parameters and recreate the model.
        
        Args:
            **params: Parameters to update
            
        Returns:
            Self for method chaining
        """
        # Update parameters
        self.model_params.update(params)
        
        # Recreate model with new parameters
        self._create_model()
        
        # Reset fitted state
        self.is_fitted = False
        self.training_metrics.clear()
        self.validation_metrics.clear()
        
        self.log_info(f"Updated model parameters: {params}")
        return self 