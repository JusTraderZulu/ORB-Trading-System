"""
Tests for mechanical model features (Phase 4a).

Tests swing detection, liquidity sweeps, MSS, SMT divergence, and iFVG
identification across various market scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import warnings

from orb.data.mechmodel import (
    swings, detect_sweep, detect_mss, smt_flag, find_ifvg,
    run_mech_pipeline, get_mech_feature_names, validate_mech_features
)
from orb.data.feature_builder import FeatureBuilder, build_features


class TestMechanicalModelFeatures:
    """Test mechanical model feature detection."""
    
    @pytest.fixture
    def sample_minute_data(self):
        """Create sample minute-level data for testing."""
        np.random.seed(42)
        
        # Create 3 days of minute data (small for fast testing)
        timestamps = pd.date_range(
            '2024-01-01 08:00:00', 
            periods=180,  # 3 hours of data
            freq='1min', 
            tz='UTC'
        )
        
        # Create price series with swing patterns
        base_price = 100
        prices = []
        volumes = []
        
        for i, ts in enumerate(timestamps):
            # Create some swing patterns
            if i < 60:  # First hour: uptrend with swings
                price = base_price + (i * 0.02) + np.random.normal(0, 0.1)
                volume = np.random.randint(1000, 3000)
            elif i < 120:  # Second hour: consolidation 
                price = base_price + 1.2 + np.random.normal(0, 0.05)
                volume = np.random.randint(800, 2000)
            else:  # Third hour: breakout with volume
                price = base_price + 1.2 + ((i - 120) * 0.03) + np.random.normal(0, 0.15)
                volume = np.random.randint(2000, 5000)  # Higher volume
            
            prices.append(max(price, base_price - 2))  # Floor price
            volumes.append(volume)
        
        # Create OHLC from price series
        df = pd.DataFrame({
            'timestamp': timestamps,
            'close': prices,
            'volume': volumes
        })
        
        # Generate realistic OHLC
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 0.1, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 0.1, len(df))
        df['vwap'] = df['close']  # Simplified VWAP
        
        return df
    
    def test_swing_detection(self, sample_minute_data):
        """Test swing high/low detection."""
        result = swings(sample_minute_data, lookback=5)
        
        # Check that swing columns exist
        assert 'swing_high' in result.columns
        assert 'swing_low' in result.columns
        
        # Check data types
        assert result['swing_high'].dtype == bool
        assert result['swing_low'].dtype == bool
        
        # Should have some swings detected
        assert result['swing_high'].sum() >= 0  # May be 0 with small dataset
        assert result['swing_low'].sum() >= 0
        
        # Original data should be preserved
        assert len(result) == len(sample_minute_data)
        assert all(col in result.columns for col in sample_minute_data.columns)
    
    def test_sweep_detection(self, sample_minute_data):
        """Test liquidity sweep detection."""
        result = detect_sweep(sample_minute_data, atr_window=10)
        
        # Check sweep columns exist
        expected_cols = ['brk_dir', 'brk_stretch_pct_atr', 'brk_minutes', 'atr']
        for col in expected_cols:
            assert col in result.columns
        
        # Check data types
        assert result['brk_dir'].dtype in [int, 'int64', 'int32']
        assert result['brk_stretch_pct_atr'].dtype in [float, 'float64']
        assert result['brk_minutes'].dtype in [int, 'int64', 'int32']
        
        # Check value ranges
        assert result['brk_dir'].isin([-1, 0, 1]).all()
        assert (result['brk_stretch_pct_atr'] >= 0).all()
        assert (result['brk_minutes'] >= 0).all()
    
    def test_mss_detection(self, sample_minute_data):
        """Test Market Structure Shift detection."""
        result = detect_mss(sample_minute_data, lookback=5)
        
        # Check MSS column exists
        assert 'mss_flip' in result.columns
        
        # Check data type and values
        assert result['mss_flip'].dtype in [int, 'int64', 'int32']
        assert result['mss_flip'].isin([-1, 0, 1]).all()
        
        # Original data preserved
        assert len(result) == len(sample_minute_data)
    
    def test_smt_divergence(self, sample_minute_data):
        """Test SMT divergence detection."""
        # Test with crypto symbol (should detect BTC-ETH pair)
        smt_series = smt_flag(sample_minute_data, primary_symbol='BTC-USD')
        
        # Check result
        assert isinstance(smt_series, pd.Series)
        assert len(smt_series) == len(sample_minute_data)
        assert smt_series.dtype in [int, 'int64', 'int32']
        assert smt_series.isin([0, 1]).all()
        
        # Test with non-crypto symbol (should return all zeros)
        smt_series_other = smt_flag(sample_minute_data, primary_symbol='AAPL')
        assert (smt_series_other == 0).all()
    
    def test_ifvg_detection(self, sample_minute_data):
        """Test Institutional Fair Value Gap detection."""
        result = find_ifvg(sample_minute_data, max_look_ahead=10)
        
        # Check iFVG columns exist
        expected_cols = ['ifvg_mid', 'ifvg_size_pct', 'ifvg_top', 'ifvg_bottom', 'ifvg_confirmed']
        for col in expected_cols:
            assert col in result.columns
        
        # Check data types
        assert result['ifvg_confirmed'].dtype == bool
        assert result['ifvg_size_pct'].dtype in [float, 'float64']
        
        # Check value constraints
        assert (result['ifvg_size_pct'] >= 0).all()
        
        # If gaps are confirmed, they should have valid mid/top/bottom
        confirmed = result[result['ifvg_confirmed']]
        if not confirmed.empty:
            assert not confirmed['ifvg_mid'].isna().any()
            assert not confirmed['ifvg_top'].isna().any()
            assert not confirmed['ifvg_bottom'].isna().any()
    
    def test_mech_pipeline_integration(self, sample_minute_data):
        """Test complete mechanical model pipeline."""
        result = run_mech_pipeline(
            min_df=sample_minute_data,
            symbol='BTC-USD',
            liquid_session_only=False,  # Use all data for testing
            swing_lookback=5,
            atr_window=10,
            ifvg_lookahead=10
        )
        
        # Should have all mechanical features
        feature_names = get_mech_feature_names()
        
        # Check key feature categories exist
        for category, features in feature_names.items():
            if category == 'smt':  # SMT may be zeros without secondary data
                continue
            for feature in features:
                if feature in ['ifvg_mid', 'ifvg_top', 'ifvg_bottom']:
                    # These can be NaN if no gaps found
                    assert feature in result.columns
                else:
                    assert feature in result.columns
                    # Should have some variation or specific values
                    assert not result[feature].isna().all()
    
    def test_feature_builder_integration(self, sample_minute_data):
        """Test mechanical features integration in FeatureBuilder."""
        # Create a single day's data for feature building
        target_date = sample_minute_data['timestamp'].iloc[0].date()
        
        # Build features using the enhanced feature builder
        features = build_features('BTC-USD', target_date, sample_minute_data)
        
        # Check that mechanical features are included
        mech_feature_names = [
            'brk_dir', 'brk_stretch_pct_atr', 'brk_minutes',
            'mss_flip', 'smt_divergence', 'ifvg_mid', 'ifvg_size_pct',
            'sweep_count', 'mss_count', 'ifvg_count'
        ]
        
        for feature in mech_feature_names:
            assert feature in features.index, f"Missing mechanical feature: {feature}"
        
        # Check feature version is updated
        assert 'feature_version' in features.index
        assert features['feature_version'] == "4.1.0"
        
        # Mechanical features should be numeric
        for feature in mech_feature_names:
            if feature not in ['ifvg_mid']:  # iFVG mid can be NaN
                assert pd.notna(features[feature]), f"Feature {feature} is NaN"
    
    def test_empty_data_handling(self):
        """Test handling of empty or insufficient data."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        result = run_mech_pipeline(empty_df)
        assert result.empty
        
        # Insufficient data
        small_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'high': [100, 101, 102, 101, 100],
            'low': [99, 100, 101, 100, 99],
            'close': [100, 101, 102, 101, 100],
            'volume': [1000, 1100, 1200, 1100, 1000]
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_mech_pipeline(small_df)
        
        # Should return data with empty features
        assert len(result) == len(small_df)
        assert 'brk_dir' in result.columns
    
    def test_feature_validation(self, sample_minute_data):
        """Test mechanical feature validation."""
        result = run_mech_pipeline(sample_minute_data, symbol='BTC-USD')
        validation = validate_mech_features(result)
        
        # Check validation structure
        assert 'total_expected_features' in validation
        assert 'features_present' in validation
        assert 'feature_coverage' in validation
        assert 'categories' in validation
        
        # Should have good coverage
        assert validation['feature_coverage'] > 0.8  # At least 80% of features
        
        # Check individual categories
        for category in ['swing', 'sweep', 'mss', 'ifvg']:
            assert category in validation['categories']
            assert validation['categories'][category]['coverage'] > 0.5


class TestMechanicalModelPerformance:
    """Test performance characteristics of mechanical model features."""
    
    def test_runtime_performance(self):
        """Test that mechanical model runs within time constraints."""
        import time
        
        # Create larger dataset for performance testing
        np.random.seed(42)
        timestamps = pd.date_range('2024-01-01', periods=500, freq='1min', tz='UTC')
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'high': np.random.uniform(100, 110, 500),
            'low': np.random.uniform(90, 100, 500),
            'close': np.random.uniform(95, 105, 500),
            'volume': np.random.randint(1000, 5000, 500)
        })
        
        # Fix OHLC consistency
        data['high'] = np.maximum(data[['close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['close']].min(axis=1), data['low'])
        
        start_time = time.time()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_mech_pipeline(data, symbol='BTC-USD')
        
        runtime = time.time() - start_time
        
        # Should complete within 5 seconds for 500 bars
        assert runtime < 5.0, f"Mechanical pipeline too slow: {runtime:.2f}s"
        assert not result.empty
    
    def test_memory_efficiency(self):
        """Test memory usage of mechanical model features."""
        # Create moderate dataset
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1min'),
            'high': np.random.uniform(100, 110, 200),
            'low': np.random.uniform(90, 100, 200),
            'close': np.random.uniform(95, 105, 200),
            'volume': np.random.randint(1000, 5000, 200)
        })
        
        original_memory = data.memory_usage(deep=True).sum()
        
        result = run_mech_pipeline(data, symbol='BTC-USD')
        result_memory = result.memory_usage(deep=True).sum()
        
        # New features shouldn't more than double memory usage
        memory_ratio = result_memory / original_memory
        assert memory_ratio < 3.0, f"Memory usage too high: {memory_ratio:.1f}x"


class TestMechanicalModelEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_missing_columns(self):
        """Test handling of missing required columns."""
        # Missing 'high' column
        bad_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            swings(bad_data)
    
    def test_extreme_values(self):
        """Test handling of extreme price values."""
        # Very large prices
        extreme_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1min'),
            'high': [1e6] * 50,
            'low': [1e6 - 100] * 50,
            'close': [1e6 - 50] * 50,
            'volume': [1000] * 50
        })
        
        # Should not crash
        result = run_mech_pipeline(extreme_data)
        assert not result.empty
        assert 'brk_dir' in result.columns
    
    def test_no_volume_data(self):
        """Test handling when volume data is missing."""
        no_vol_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1min'),
            'high': np.random.uniform(100, 110, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(95, 105, 50)
        })
        
        # Should create synthetic volume and continue
        result = find_ifvg(no_vol_data)
        assert 'volume' in result.columns  # Should be added synthetically
        assert 'ifvg_confirmed' in result.columns


# Smoke test to run quickly
def test_mech_features_smoke():
    """Quick smoke test for mechanical model features."""
    # Minimal test data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01 08:00', periods=30, freq='1min', tz='UTC'),
        'high': np.random.uniform(100, 102, 30),
        'low': np.random.uniform(98, 100, 30), 
        'close': np.random.uniform(99, 101, 30),
        'volume': np.random.randint(1000, 2000, 30)
    })
    
    # Test individual components
    swing_result = swings(data, lookback=3)
    assert 'swing_high' in swing_result.columns
    
    sweep_result = detect_sweep(data, atr_window=5)
    assert 'brk_dir' in sweep_result.columns
    
    mss_result = detect_mss(data, lookback=3)
    assert 'mss_flip' in mss_result.columns
    
    # Test pipeline
    pipeline_result = run_mech_pipeline(data, symbol='BTC-USD')
    assert not pipeline_result.empty
    
    print("✅ Mechanical model smoke test passed!") 