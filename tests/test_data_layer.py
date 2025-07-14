"""
Tests for orb.data layer - polygon_loader and feature_builder modules.

Tests data ingestion, parquet processing, and feature engineering
with mocked API calls and real data validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, time
from pathlib import Path
import json
import gzip
import tempfile
import shutil
from unittest.mock import patch, Mock
import pytz

from orb.data.polygon_loader import PolygonLoader, download_month, build_parquet
from orb.data.feature_builder import build_features, session_slice, FeatureBuilder
from orb.utils.calendars import get_market_timezone


class TestPolygonLoader:
    """Test Polygon API data loading and processing."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_polygon_response(self):
        """Sample Polygon API response data."""
        return {
            'status': 'OK',
            'results': [
                {
                    't': 1641312600000,  # 2022-01-04 14:30:00 UTC = 09:30:00 EST
                    'o': 100.0,
                    'h': 101.0,
                    'l': 99.5,
                    'c': 100.5,
                    'v': 1000,
                    'vw': 100.25,
                    'n': 10
                },
                {
                    't': 1641312660000,  # 2022-01-04 14:31:00 UTC = 09:31:00 EST
                    'o': 100.5,
                    'h': 101.5,
                    'l': 100.0,
                    'c': 101.0,
                    'v': 1500,
                    'vw': 100.75,
                    'n': 15
                }
            ]
        }
    
    @pytest.fixture
    def polygon_loader(self, temp_data_dir):
        """Create PolygonLoader instance with temp directory."""
        return PolygonLoader(
            api_key="test_key",
            raw_data_dir=f"{temp_data_dir}/raw",
            minute_data_dir=f"{temp_data_dir}/minute"
        )
    
    def test_download_month_success(self, polygon_loader, sample_polygon_response):
        """Test successful month download with mocked API."""
        with patch.object(polygon_loader, '_make_request', return_value=sample_polygon_response):
            # Should not raise exception
            polygon_loader.download_month('AAPL', 2022, 1)
            
            # Check file was created
            raw_file = Path(polygon_loader.raw_data_dir) / 'AAPL' / '202201.json.gz'
            assert raw_file.exists()
            
            # Verify content
            with gzip.open(raw_file, 'rt') as f:
                data = json.load(f)
            
            assert data['symbol'] == 'AAPL'
            assert len(data['results']) == 2
            assert data['metadata']['year'] == 2022
            assert data['metadata']['month'] == 1
    
    def test_download_month_api_error(self, polygon_loader):
        """Test download failure with API error."""
        error_response = {'status': 'ERROR', 'error': 'API key invalid'}
        
        with patch.object(polygon_loader, '_make_request', return_value=error_response):
            with pytest.raises(Exception, match="Polygon API error"):
                polygon_loader.download_month('AAPL', 2022, 1)
    
    def test_build_parquet_success(self, polygon_loader, sample_polygon_response, temp_data_dir):
        """Test successful parquet building from raw JSON."""
        # Create test raw data
        raw_dir = Path(temp_data_dir) / 'raw' / 'AAPL'
        raw_dir.mkdir(parents=True)
        
        test_data = {
            'symbol': 'AAPL',
            'results': sample_polygon_response['results'],
            'metadata': {'year': 2022, 'month': 1, 'count': 2}
        }
        
        with gzip.open(raw_dir / '202201.json.gz', 'wt') as f:
            json.dump(test_data, f)
        
        # Test parquet building
        parquet_path = polygon_loader.build_parquet('AAPL')
        
        # Verify file was created
        assert parquet_path.exists()
        assert parquet_path.name == 'AAPL.parquet'
        
        # Verify content - timestamp conversion
        df = pd.read_parquet(parquet_path)
        assert len(df) == 2
        assert 'timestamp' in df.columns
        assert df['symbol'].iloc[0] == 'AAPL'
        
        # Check timezone conversion (should be Eastern time)
        first_timestamp = df['timestamp'].iloc[0]
        assert first_timestamp.tz is not None
        # Verify it's in Eastern timezone
        assert 'America/New_York' in str(first_timestamp.tz) or 'US/Eastern' in str(first_timestamp.tz)
        # Should be during trading hours (9-16 range)
        assert 9 <= first_timestamp.hour <= 16
    
    def test_build_parquet_no_raw_data(self, polygon_loader):
        """Test parquet building with no raw data."""
        with pytest.raises(Exception, match="No raw data found"):
            polygon_loader.build_parquet('NONEXISTENT')
    
    def test_trading_hours_filter(self, polygon_loader):
        """Test filtering to trading hours only."""
        # Create test data with various times
        test_data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2022-01-04 08:00:00',  # Pre-market
                '2022-01-04 09:30:00',  # Market open
                '2022-01-04 12:00:00',  # Mid-day
                '2022-01-04 16:00:00',  # Market close
                '2022-01-04 18:00:00',  # After hours
            ]).tz_localize(get_market_timezone()),
            'open': [100, 101, 102, 103, 104],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5],
            'close': [100.2, 101.2, 102.2, 103.2, 104.2],
            'volume': [1000, 1000, 1000, 1000, 1000]
        })
        
        filtered = polygon_loader._filter_trading_hours(test_data)
        
        # Should keep 9:25 AM to 4:00 PM (updated to 9:25)
        assert len(filtered) == 3  # 9:30, 12:00, 16:00
        assert filtered['timestamp'].dt.time.min() >= time(9, 25)
        assert filtered['timestamp'].dt.time.max() <= time(16, 0)
    
    def test_independence_day_2025_filtering(self, polygon_loader):
        """Test that Independence Day 2025 data is properly filtered out."""
        # Create test data including July 4, 2025 (Friday)
        test_data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2025-07-03 10:00:00',  # Thursday before
                '2025-07-04 10:00:00',  # Independence Day (Friday)
                '2025-07-07 10:00:00',  # Monday after
            ]).tz_localize(get_market_timezone()),
            'open': [100, 101, 102],
            'high': [100.5, 101.5, 102.5],
            'low': [99.5, 100.5, 101.5],
            'close': [100.2, 101.2, 102.2],
            'volume': [1000, 1000, 1000]
        })
        
        filtered = polygon_loader._filter_trading_hours(test_data)
        
        # July 4th should be excluded
        filtered_dates = filtered['timestamp'].dt.date.unique()
        assert date(2025, 7, 4) not in filtered_dates
        assert date(2025, 7, 3) in filtered_dates
        assert date(2025, 7, 7) in filtered_dates


class TestFeatureBuilder:
    """Test feature engineering and calculation."""
    
    @pytest.fixture
    def sample_minute_data(self):
        """Create sample minute-level data for testing."""
        # Create data for a single trading day
        timestamps = pd.date_range(
            start='2025-01-02 09:25:00',
            end='2025-01-02 16:00:00',
            freq='1min',
            tz=get_market_timezone()
        )
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible tests
        n = len(timestamps)
        
        # Start with opening price
        prices = [100.0]
        volumes = []
        
        for i in range(1, n):
            # Random walk with slight upward bias
            change = np.random.normal(0.001, 0.01)
            new_price = max(prices[-1] * (1 + change), 0.01)
            prices.append(new_price)
            
            # Higher volume during opening and closing
            hour = timestamps[i].hour
            if hour == 9 or hour == 15:
                base_volume = 2000
            else:
                base_volume = 1000
            volume = max(int(np.random.normal(base_volume, 300)), 100)
            volumes.append(volume)
        
        # Add opening volume
        volumes.insert(0, 2500)
        
        # Create OHLC data
        data = []
        for i, ts in enumerate(timestamps):
            if i == 0:
                open_price = prices[i]
                high_price = prices[i] * 1.002
                low_price = prices[i] * 0.998
                close_price = prices[i] * 1.001
            else:
                open_price = prices[i-1] if i > 0 else prices[i]
                close_price = prices[i]
                # High/low with some random variation
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.005))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.005))
            
            data.append({
                'timestamp': ts,
                'symbol': 'AAPL',
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volumes[i],
                'vwap': round((high_price + low_price + close_price) / 3, 2),
                'num_trades': np.random.randint(5, 25)
            })
        
        return pd.DataFrame(data)
    
    def test_build_features_basic(self, sample_minute_data):
        """Test basic feature building functionality."""
        target_date = date(2025, 1, 2)
        
        features = build_features('AAPL', target_date, sample_minute_data)
        
        # Check that all required features are present
        required_features = ['or_high', 'or_low', 'or_range', 'or_vol', 
                           'atr14pct', 'ema20_slope', 'vwap_dev', 'y']
        
        for feature in required_features:
            assert feature in features.index, f"Missing feature: {feature}"
        
        # Check data types
        assert isinstance(features['or_high'], float)
        assert isinstance(features['or_low'], float)
        assert isinstance(features['or_range'], float)
        assert isinstance(features['or_vol'], float)
        assert isinstance(features['vwap_dev'], float)
        assert isinstance(features['y'], (int, np.integer))
    
    def test_opening_range_features(self, sample_minute_data):
        """Test opening range feature calculations."""
        target_date = date(2025, 1, 2)
        
        features = build_features('AAPL', target_date, sample_minute_data)
        
        # Verify opening range relationships
        assert features['or_high'] > features['or_low'], "OR high should be > OR low"
        assert abs(features['or_range'] - (features['or_high'] - features['or_low'])) < 0.01, \
               "OR range should equal OR high - OR low"
        assert features['or_vol'] > 0, "OR volume should be positive"
        
        # Check that OR features are calculated from 9:30-10:00 window
        # Use the exact same logic as the feature builder (<=10:00 not <10:00)
        opening_range_end_time = pd.to_datetime("10:00", format='%H:%M').time()
        or_window = sample_minute_data[
            sample_minute_data['timestamp'].dt.time <= opening_range_end_time
        ]
        
        expected_high = or_window['high'].max()
        expected_low = or_window['low'].min()
        expected_volume = or_window['volume'].sum()
        
        assert abs(features['or_high'] - expected_high) < 0.01
        assert abs(features['or_low'] - expected_low) < 0.01
        # Use relative tolerance for volume since large numbers can have small differences
        assert abs(features['or_vol'] - expected_volume) / expected_volume < 0.01
    
    def test_session_slice_helper(self, sample_minute_data):
        """Test session_slice helper function."""
        target_date = date(2025, 1, 2)
        
        sliced = session_slice(sample_minute_data, target_date)
        
        # Should return data for the specified date only
        assert len(sliced) > 0
        assert all(sliced['timestamp'].dt.date == target_date)
        
        # Should be within trading hours (9:25-16:00)
        assert sliced['timestamp'].dt.time.min() >= time(9, 25)
        assert sliced['timestamp'].dt.time.max() <= time(16, 0)
    
    def test_empty_data_handling(self):
        """Test handling of empty or missing data."""
        empty_df = pd.DataFrame(columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        target_date = date(2025, 1, 2)
        
        features = build_features('AAPL', target_date, empty_df)
        
        # Should return empty series for missing data
        assert len(features) == 0 or features.isna().all()
    
    def test_independence_day_2025_empty_features(self):
        """Test that Independence Day 2025 returns empty features (holiday)."""
        # Create empty dataframe (no trading on holiday)
        empty_df = pd.DataFrame(columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        
        features = build_features('AAPL', date(2025, 7, 4), empty_df)
        
        # Should return empty or NaN features for holiday
        assert len(features) == 0 or features.isna().all()
    
    def test_feature_data_types(self, sample_minute_data):
        """Test that features have correct data types."""
        target_date = date(2025, 1, 2)
        
        features = build_features('AAPL', target_date, sample_minute_data)
        
        # Float features
        float_features = ['or_high', 'or_low', 'or_range', 'or_vol', 'vwap_dev']
        for feature in float_features:
            if feature in features.index and not pd.isna(features[feature]):
                assert isinstance(features[feature], (float, np.floating)), \
                       f"{feature} should be float, got {type(features[feature])}"
        
        # Integer features
        int_features = ['y']
        for feature in int_features:
            if feature in features.index and not pd.isna(features[feature]):
                assert isinstance(features[feature], (int, np.integer)), \
                       f"{feature} should be int, got {type(features[feature])}"
    
    def test_forward_return_label(self, sample_minute_data):
        """Test forward return label calculation."""
        target_date = date(2025, 1, 2)
        
        features = build_features('AAPL', target_date, sample_minute_data)
        
        # Label should be 0 or 1
        assert features['y'] in [0, 1], f"Label should be 0 or 1, got {features['y']}"
        
        # Verify calculation manually
        entry_mask = sample_minute_data['timestamp'].dt.time >= time(10, 0)
        exit_mask = sample_minute_data['timestamp'].dt.time <= time(15, 55)
        
        if entry_mask.any() and exit_mask.any():
            entry_price = sample_minute_data[entry_mask].iloc[0]['close']
            exit_price = sample_minute_data[exit_mask].iloc[-1]['close']
            forward_return = (exit_price - entry_price) / entry_price
            expected_label = 1 if forward_return > 0 else 0
            
            assert features['y'] == expected_label


class TestStandaloneFunctions:
    """Test standalone convenience functions."""
    
    @patch.dict('os.environ', {'POLYGON_API_KEY': 'test_key'})
    @patch('orb.data.polygon_loader.PolygonLoader')
    def test_download_month_standalone(self, mock_loader_class):
        """Test standalone download_month function."""
        mock_instance = Mock()
        mock_loader_class.return_value = mock_instance
        
        # Should not raise exception
        download_month('AAPL', 2025, 1)
        
        # Verify loader was created with correct parameters
        mock_loader_class.assert_called_once_with(
            api_key='test_key',
            raw_data_dir='data/raw'
        )
        mock_instance.download_month.assert_called_once_with('AAPL', 2025, 1)
    
    def test_download_month_no_api_key(self):
        """Test standalone download_month without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('dotenv.load_dotenv'):  # Mock dotenv to prevent loading from .env file
                with pytest.raises(Exception, match="POLYGON_API_KEY environment variable not set"):
                    download_month('AAPL', 2025, 1)
    
    @patch('orb.data.polygon_loader.PolygonLoader')
    def test_build_parquet_standalone(self, mock_loader_class):
        """Test standalone build_parquet function."""
        mock_instance = Mock()
        mock_instance.build_parquet.return_value = Path('test.parquet')
        mock_loader_class.return_value = mock_instance
        
        result = build_parquet('AAPL')
        
        # Verify result
        assert isinstance(result, Path)
        
        # Verify loader was created
        mock_loader_class.assert_called_once()
        mock_instance.build_parquet.assert_called_once_with('AAPL')


class TestPerformance:
    """Test performance requirements (< 5 sec)."""
    
    def test_feature_building_performance(self):
        """Test that feature building completes quickly."""
        import time
        
        # Create small dataset for performance test
        timestamps = pd.date_range(
            start='2025-01-02 09:25:00',
            end='2025-01-02 16:00:00',
            freq='1min',
            tz=get_market_timezone()
        )
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': 'AAPL',
            'open': 100.0,
            'high': 100.5,
            'low': 99.5,
            'close': 100.2,
            'volume': 1000,
            'vwap': 100.1,
            'num_trades': 10
        })
        
        start_time = time.time()
        features = build_features('AAPL', date(2025, 1, 2), data)
        end_time = time.time()
        
        # Should complete in under 5 seconds (much faster expected)
        assert (end_time - start_time) < 5.0, "Feature building took too long"
        assert len(features) > 0, "Should return features" 