"""
Tests for the ORB model layer components.

Tests cover LightGBM model, Optuna tuning, and walk-forward evaluation
with smoke tests designed to run in under 15 seconds.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from orb.models.base_model import BaseModel
from orb.models.lgbm_model import LGBMModel
from orb.tuning.optuna_tuner import OptunaTuner
from orb.evaluation.walkforward import WalkForwardRunner


class TestBaseModel:
    """Test BaseModel abstract class."""
    
    def test_base_model_cannot_be_instantiated(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()
    
    def test_base_model_validation(self):
        """Test input validation methods."""
        model = LGBMModel(random_state=42)
        
        # Test valid input
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        
        model.validate_inputs(X, y)  # Should not raise
        
        # Test invalid inputs
        with pytest.raises(ValueError):
            model.validate_inputs("not_a_dataframe", y)
        
        with pytest.raises(ValueError):
            model.validate_inputs(X, "not_a_series")
        
        with pytest.raises(ValueError):
            model.validate_inputs(X, pd.Series([0, 1]))  # Wrong length
        
        with pytest.raises(ValueError):
            model.validate_inputs(X, pd.Series([0, 1, 2]))  # Invalid target values


class TestLGBMModel:
    """Test LightGBM model implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create features
        X = pd.DataFrame({
            'or_high': np.random.normal(100, 10, n_samples),
            'or_low': np.random.normal(95, 10, n_samples),
            'or_range': np.random.normal(5, 2, n_samples),
            'or_vol': np.random.normal(1000000, 200000, n_samples),
            'atr14pct': np.random.normal(0.02, 0.005, n_samples),
            'ema20_slope': np.random.normal(0.001, 0.001, n_samples),
            'vwap_dev': np.random.normal(0.0, 0.01, n_samples)
        })
        
        # Create target with some correlation to features
        y = ((X['or_range'] > X['or_range'].median()) & 
             (X['or_vol'] > X['or_vol'].median())).astype(int)
        
        return X, y
    
    def test_lgbm_model_initialization(self):
        """Test LightGBM model initialization."""
        model = LGBMModel(random_state=42, n_estimators=10)
        
        assert model.random_state == 42
        assert model.model_params['n_estimators'] == 10
        assert not model.is_fitted
        assert model.feature_names is None
    
    def test_lgbm_model_fit_predict(self, sample_data):
        """Test model training and prediction."""
        X, y = sample_data
        
        model = LGBMModel(random_state=42, n_estimators=10)
        
        # Test fitting
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.feature_names == list(X.columns)
        assert len(model.training_metrics) > 0
        
        # Test prediction
        y_pred_proba = model.predict_proba(X)
        y_pred = model.predict(X)
        
        assert y_pred_proba.shape == (len(X), 2)
        assert y_pred.shape == (len(X),)
        assert set(y_pred) <= {0, 1}
    
    def test_lgbm_model_with_validation(self, sample_data):
        """Test model training with validation set."""
        X, y = sample_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = LGBMModel(random_state=42, n_estimators=10)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        assert model.is_fitted
        assert len(model.training_metrics) > 0
        assert len(model.validation_metrics) > 0
    
    def test_lgbm_model_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        
        model = LGBMModel(random_state=42, n_estimators=10)
        model.fit(X, y)
        
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert isinstance(importance, pd.Series)
        assert len(importance) == len(X.columns)
        # Check that all original features are present (order may differ due to LightGBM internal processing)
        assert set(importance.index) == set(X.columns)
    
    def test_lgbm_model_save_load(self, sample_data):
        """Test model persistence."""
        X, y = sample_data
        
        model = LGBMModel(random_state=42, n_estimators=10)
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_model"
            
            # Save model
            model.save(model_path)
            
            # Load model
            loaded_model = LGBMModel()
            loaded_model.load(model_path)
            
            assert loaded_model.is_fitted
            assert loaded_model.feature_names == model.feature_names
            assert loaded_model.random_state == model.random_state
            
            # Test predictions match
            pred_orig = model.predict_proba(X)
            pred_loaded = loaded_model.predict_proba(X)
            
            np.testing.assert_array_almost_equal(pred_orig, pred_loaded, decimal=5)
    
    def test_lgbm_model_parameter_updates(self):
        """Test model parameter updates."""
        model = LGBMModel(random_state=42, n_estimators=10)
        
        # Update parameters
        model.set_model_params(n_estimators=20, learning_rate=0.05)
        
        assert model.model_params['n_estimators'] == 20
        assert model.model_params['learning_rate'] == 0.05
        assert not model.is_fitted  # Should reset fitted state


class TestOptunaTuner:
    """Test Optuna hyperparameter tuning."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 500  # Smaller for faster tests
        
        X = pd.DataFrame({
            'or_high': np.random.normal(100, 10, n_samples),
            'or_low': np.random.normal(95, 10, n_samples),
            'or_range': np.random.normal(5, 2, n_samples),
            'or_vol': np.random.normal(1000000, 200000, n_samples),
            'atr14pct': np.random.normal(0.02, 0.005, n_samples),
            'ema20_slope': np.random.normal(0.001, 0.001, n_samples),
            'vwap_dev': np.random.normal(0.0, 0.01, n_samples)
        })
        
        y = ((X['or_range'] > X['or_range'].median()) & 
             (X['or_vol'] > X['or_vol'].median())).astype(int)
        
        return X, y
    
    @patch('mlflow.create_experiment')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    def test_optuna_tuner_initialization(self, mock_start_run, mock_set_exp, mock_create_exp):
        """Test Optuna tuner initialization."""
        mock_create_exp.return_value = "test_experiment_id"
        
        tuner = OptunaTuner(
            random_state=42,
            n_trials=2,
            mlflow_experiment_name="test_experiment"
        )
        
        assert tuner.random_state == 42
        assert tuner.n_trials == 2
        assert tuner.mlflow_experiment_name == "test_experiment"
    
    @patch('mlflow.create_experiment')
    @patch('mlflow.set_experiment') 
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    @patch('mlflow.lightgbm.log_model')
    def test_optuna_tuner_smoke_test(self, mock_log_model, mock_log_metric, 
                                   mock_log_params, mock_start_run, 
                                   mock_set_exp, mock_create_exp, sample_data):
        """Smoke test for Optuna tuning with minimal trials."""
        X, y = sample_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Mock MLflow
        mock_create_exp.return_value = "test_experiment_id"
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        tuner = OptunaTuner(
            random_state=42,
            n_trials=2,  # Minimal trials for speed
            mlflow_experiment_name="test_experiment"
        )
        
        # Run tuning
        best_params, best_model = tuner.tune_lgbm(
            X_train, y_train, X_val, y_val,
            cv_folds=2,  # Minimal folds for speed
            early_stopping_rounds=5
        )
        
        # Verify results
        assert best_params is not None
        assert isinstance(best_params, dict)
        assert best_model is not None
        assert isinstance(best_model, LGBMModel)
        assert best_model.is_fitted
        
        # Verify study was created
        assert tuner.study is not None
        assert len(tuner.study.trials) == 2
    
    def test_optuna_default_param_space(self):
        """Test default parameter space generation."""
        tuner = OptunaTuner(random_state=42)
        param_space = tuner._get_default_param_space()
        
        assert isinstance(param_space, dict)
        assert 'n_estimators' in param_space
        assert 'learning_rate' in param_space
        assert 'max_depth' in param_space
        
        # Check parameter types
        assert param_space['n_estimators']['type'] == 'int'
        assert param_space['learning_rate']['type'] == 'float'
        assert param_space['learning_rate']['log'] == True


class TestWalkForwardRunner:
    """Test walk-forward evaluation."""
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Create sample time series data for walk-forward testing."""
        np.random.seed(42)
        
        # Create 30 days of data (enough for 20 sessions as requested)
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        n_samples = len(dates)
        
        data = pd.DataFrame({
            'date': dates,
            'or_high': np.random.normal(100, 10, n_samples),
            'or_low': np.random.normal(95, 10, n_samples),
            'or_range': np.random.normal(5, 2, n_samples),
            'or_vol': np.random.normal(1000000, 200000, n_samples),
            'atr14pct': np.random.normal(0.02, 0.005, n_samples),
            'ema20_slope': np.random.normal(0.001, 0.001, n_samples),
            'vwap_dev': np.random.normal(0.0, 0.01, n_samples)
        })
        
        # Create target with some trend
        data['forward_return_label'] = (
            (data['or_range'] > data['or_range'].rolling(5).mean()) &
            (data['or_vol'] > data['or_vol'].rolling(5).mean())
        ).astype(int)
        
        return data
    
    def test_walkforward_initialization(self):
        """Test walk-forward runner initialization."""
        runner = WalkForwardRunner(
            train_days=10,
            validation_days=5,
            retrain_gap=3,
            random_state=42
        )
        
        assert runner.train_days == 10
        assert runner.validation_days == 5
        assert runner.retrain_gap == 3
        assert runner.random_state == 42
        assert len(runner.results) == 0
    
    def test_walkforward_input_validation(self, sample_time_series_data):
        """Test input validation for walk-forward evaluation."""
        runner = WalkForwardRunner(train_days=5, validation_days=3, retrain_gap=2)
        
        data = sample_time_series_data
        features = ['or_high', 'or_low', 'or_range']
        target = 'forward_return_label'
        
        # Test valid inputs
        runner._validate_inputs(data, features, target, 'date')
        
        # Test invalid inputs
        with pytest.raises(ValueError):
            runner._validate_inputs(pd.DataFrame(), features, target, 'date')
        
        with pytest.raises(ValueError):
            runner._validate_inputs(data, [], target, 'date')
        
        with pytest.raises(ValueError):
            runner._validate_inputs(data, features, 'nonexistent_target', 'date')
    
    def test_walkforward_window_generation(self, sample_time_series_data):
        """Test walk-forward window generation."""
        runner = WalkForwardRunner(
            train_days=5,
            validation_days=3,
            retrain_gap=2
        )
        
        data = sample_time_series_data
        prepared_data = runner._prepare_data(data, ['or_high'], 'forward_return_label', 'date')
        
        windows = runner._generate_windows(prepared_data)
        
        assert len(windows) > 0
        assert all('window_id' in w for w in windows)
        assert all('train_start_date' in w for w in windows)
        assert all('val_end_date' in w for w in windows)
    
    def test_walkforward_smoke_test(self, sample_time_series_data):
        """Smoke test for walk-forward evaluation with 20 sessions."""
        data = sample_time_series_data
        
        # Use minimal windows for speed
        runner = WalkForwardRunner(
            train_days=5,
            validation_days=2,
            retrain_gap=1,
            min_train_samples=3,  # Lower threshold for small dataset
            random_state=42
        )
        
        features = ['or_high', 'or_low', 'or_range']
        target = 'forward_return_label'
        
        # Create simple model factory
        def model_factory():
            return LGBMModel(random_state=42, n_estimators=5)
        
        # Run evaluation
        results = runner.run_walkforward(
            data=data,
            features=features,
            target=target,
            model_factory=model_factory
        )
        
        # Verify results structure
        assert 'results' in results
        assert 'predictions' in results
        assert 'performance_metrics' in results
        assert 'config' in results
        
        # Verify we have results (might be fewer than 20 due to small dataset)
        assert len(results['results']) > 0
        assert len(results['predictions']) > 0
        
        # Check that we have valid metrics
        metrics = results['performance_metrics']
        assert 'overall_accuracy' in metrics
        assert 'overall_roc_auc' in metrics
        assert 'n_windows' in metrics
    
    def test_walkforward_results_dataframe(self, sample_time_series_data):
        """Test conversion of results to DataFrame."""
        runner = WalkForwardRunner(
            train_days=5,
            validation_days=2,
            retrain_gap=2,
            min_train_samples=3
        )
        
        data = sample_time_series_data
        features = ['or_high', 'or_low']
        target = 'forward_return_label'
        
        def model_factory():
            return LGBMModel(random_state=42, n_estimators=5)
        
        runner.run_walkforward(
            data=data,
            features=features,
            target=target,
            model_factory=model_factory
        )
        
        # Test results DataFrame
        results_df = runner.get_results_dataframe()
        assert isinstance(results_df, pd.DataFrame)
        
        if not results_df.empty:
            assert 'window_id' in results_df.columns
            assert 'train_samples' in results_df.columns
            assert 'val_samples' in results_df.columns
        
        # Test predictions DataFrame
        predictions_df = runner.get_predictions_dataframe()
        assert isinstance(predictions_df, pd.DataFrame)
        
        if not predictions_df.empty:
            assert 'window_id' in predictions_df.columns
            assert 'y_true' in predictions_df.columns
            assert 'y_pred' in predictions_df.columns
    
    def test_walkforward_save_load(self, sample_time_series_data):
        """Test saving and loading walk-forward results."""
        runner = WalkForwardRunner(
            train_days=5,
            validation_days=2,
            retrain_gap=2,
            min_train_samples=3
        )
        
        data = sample_time_series_data
        features = ['or_high', 'or_low']
        target = 'forward_return_label'
        
        def model_factory():
            return LGBMModel(random_state=42, n_estimators=5)
        
        runner.run_walkforward(
            data=data,
            features=features,
            target=target,
            model_factory=model_factory
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir) / "walkforward_results.pkl"
            
            # Save results
            runner.save_results(str(results_path))
            assert results_path.exists()
            
            # Load results
            new_runner = WalkForwardRunner()
            new_runner.load_results(str(results_path))
            
            assert len(new_runner.results) == len(runner.results)
            assert len(new_runner.predictions) == len(runner.predictions)


class TestModelIntegration:
    """Integration tests for model layer components."""
    
    @pytest.fixture
    def sample_features_data(self):
        """Create sample features data matching real ORB features."""
        np.random.seed(42)
        
        # Create 50 days of data for faster testing
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        n_samples = len(dates)
        
        data = pd.DataFrame({
            'date': dates,
            'or_high': np.random.normal(100, 10, n_samples),
            'or_low': np.random.normal(95, 10, n_samples),
            'or_range': np.random.normal(5, 2, n_samples),
            'or_vol': np.random.normal(1000000, 200000, n_samples),
            'atr14pct': np.random.normal(0.02, 0.005, n_samples),
            'ema20_slope': np.random.normal(0.001, 0.001, n_samples),
            'vwap_dev': np.random.normal(0.0, 0.01, n_samples)
        })
        
        # Create realistic target
        data['forward_return_label'] = (
            (data['or_range'] > data['or_range'].rolling(3).mean()) &
            (data['vwap_dev'] > 0) &
            (data['atr14pct'] > data['atr14pct'].median())
        ).astype(int)
        
        return data
    
    def test_full_pipeline_smoke_test(self, sample_features_data):
        """Smoke test for full model pipeline."""
        data = sample_features_data
        
        # Define ORB features
        feature_cols = [
            'or_high', 'or_low', 'or_range', 'or_vol',
            'atr14pct', 'ema20_slope', 'vwap_dev'
        ]
        target_col = 'forward_return_label'
        
        # Step 1: Train a basic model
        split_idx = int(len(data) * 0.6)
        X_train = data[feature_cols].iloc[:split_idx]
        y_train = data[target_col].iloc[:split_idx]
        
        model = LGBMModel(random_state=42, n_estimators=10)
        model.fit(X_train, y_train)
        
        assert model.is_fitted
        
        # Step 2: Get feature importance
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == len(feature_cols)
        
        # Step 3: Test walk-forward with trained model
        runner = WalkForwardRunner(
            train_days=10,
            validation_days=3,
            retrain_gap=2,
            min_train_samples=5
        )
        
        def model_factory():
            return LGBMModel(random_state=42, n_estimators=10)
        
        results = runner.run_walkforward(
            data=data,
            features=feature_cols,
            target=target_col,
            model_factory=model_factory
        )
        
        assert len(results['results']) > 0
        assert 'overall_roc_auc' in results['performance_metrics']
        
        # Verify we can get reasonable performance
        auc = results['performance_metrics']['overall_roc_auc']
        assert 0.0 <= auc <= 1.0
    
    def test_model_consistency(self, sample_features_data):
        """Test that models produce consistent results."""
        data = sample_features_data
        
        feature_cols = ['or_high', 'or_low', 'or_range']
        target_col = 'forward_return_label'
        
        X = data[feature_cols]
        y = data[target_col]
        
        # Train two models with same parameters
        model1 = LGBMModel(random_state=42, n_estimators=10)
        model2 = LGBMModel(random_state=42, n_estimators=10)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Predictions should be identical
        pred1 = model1.predict_proba(X)
        pred2 = model2.predict_proba(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)


def test_model_layer_performance():
    """Test that model layer components complete within time constraints."""
    import time
    
    # Create small dataset for speed
    np.random.seed(42)
    X = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'f3': np.random.randn(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    
    start_time = time.time()
    
    # Test model training
    model = LGBMModel(random_state=42, n_estimators=5)
    model.fit(X, y)
    
    # Test prediction
    model.predict_proba(X)
    
    elapsed = time.time() - start_time
    
    # Should complete very quickly for small dataset
    assert elapsed < 5.0  # 5 seconds max for this small test


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 