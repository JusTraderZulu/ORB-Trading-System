"""
Feature engineering module for ORB trading system.

Creates opening range features, technical indicators, and forward return labels
from minute-level market data.

Implements build_features() and session_slice() functions per ORB spec §4.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, date
from typing import Optional, Union, Dict, Any
import warnings
import pytz

from ..utils.logging import LoggingMixin
from ..utils.calendars import get_market_timezone, trading_days


class FeatureBuilder(LoggingMixin):
    """
    Feature builder for ORB trading system.
    
    Creates features including:
    - Opening range features (high, low, range, volume)
    - Technical indicators (ATR, EMA, VWAP)
    - Forward return labels
    """
    
    def __init__(
        self,
        opening_range_end: str = "10:00",
        market_start: str = "09:30",
        market_end: str = "16:00",
        exit_time: str = "15:55"
    ):
        """
        Initialize feature builder.
        
        Args:
            opening_range_end: End time for opening range calculation
            market_start: Market open time
            market_end: Market close time
            exit_time: Exit time for forward return calculation
        """
        self.opening_range_end = opening_range_end
        self.market_start = market_start
        self.market_end = market_end
        self.exit_time = exit_time
        
        # Convert times to time objects
        self.opening_range_end_time = pd.to_datetime(opening_range_end, format='%H:%M').time()
        self.market_start_time = pd.to_datetime(market_start, format='%H:%M').time()
        self.market_end_time = pd.to_datetime(market_end, format='%H:%M').time()
        self.exit_time_obj = pd.to_datetime(exit_time, format='%H:%M').time()
        
    def build_features(
        self,
        symbol: str,
        date: Union[str, datetime, pd.Timestamp],
        minute_df: pd.DataFrame
    ) -> pd.Series:
        """
        Build features for a specific symbol and date.
        
        Args:
            symbol: Stock symbol
            date: Target date for feature calculation
            minute_df: DataFrame with minute-level data
            
        Returns:
            Series with calculated features
        """
        target_date = pd.to_datetime(date).date()
        
        # Filter data for the target date
        if 'timestamp' in minute_df.columns and len(minute_df) > 0:
            date_mask = minute_df['timestamp'].dt.date == target_date
            day_data = minute_df[date_mask].copy()
        else:
            if len(minute_df) == 0:
                self.log_warning(f"Empty DataFrame provided for {symbol} on {target_date}")
            else:
                self.log_error("DataFrame must have 'timestamp' column")
            return pd.Series(dtype=float)
        
        if day_data.empty:
            self.log_warning(f"No data found for {symbol} on {target_date}")
            return pd.Series(dtype=float)
        
        # Sort by timestamp
        day_data = day_data.sort_values('timestamp').reset_index(drop=True)
        
        try:
            features = {}
            
            # 1. Opening Range Features
            or_features = self._calculate_opening_range_features(day_data)
            features.update(or_features)
            
            # 2. Technical Indicators
            tech_features = self._calculate_technical_features(symbol, date, minute_df)
            features.update(tech_features)
            
            # 3. VWAP Deviation
            vwap_features = self._calculate_vwap_features(day_data)
            features.update(vwap_features)
            
            # 4. Forward Return Label
            label_features = self._calculate_forward_return_label(day_data)
            features.update(label_features)
            
            # 5. Additional Context Features
            context_features = self._calculate_context_features(day_data)
            features.update(context_features)
            
            # Add metadata
            features['symbol'] = symbol
            features['date'] = target_date
            features['timestamp'] = pd.to_datetime(f"{target_date} {self.opening_range_end}")
            
            return pd.Series(features)
            
        except Exception as e:
            self.log_error(f"Error building features for {symbol} on {date}: {e}")
            return pd.Series(dtype=float)
    
    def _calculate_opening_range_features(self, day_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate opening range features (9:30-10:00)."""
        features = {}
        
        # Filter to opening range period
        or_mask = day_data['timestamp'].dt.time <= self.opening_range_end_time
        or_data = day_data[or_mask]
        
        if or_data.empty:
            self.log_warning("No opening range data found")
            return {
                'or_high': np.nan,
                'or_low': np.nan,
                'or_range': np.nan,
                'or_vol': np.nan
            }
        
        # Calculate opening range features
        features['or_high'] = float(or_data['high'].max())
        features['or_low'] = float(or_data['low'].min())
        features['or_range'] = float(features['or_high'] - features['or_low'])
        features['or_vol'] = float(or_data['volume'].sum())
            
        return features
    
    def _calculate_technical_features(
        self,
        symbol: str,
        date: Union[str, datetime, pd.Timestamp],
        minute_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate technical indicators (ATR, EMA)."""
        features = {}
        target_date = pd.to_datetime(date).date()
        
        # Get historical data (up to target date)
        hist_mask = minute_df['timestamp'].dt.date <= target_date
        hist_data = minute_df[hist_mask].copy()
        
        if hist_data.empty:
            return {
                'atr14pct': np.nan,
                'ema20_slope': np.nan
            }
        
        # Create daily data for ATR calculation
        daily_data = self._create_daily_data(hist_data)
        
        # Calculate ATR (14-day) - forward-fill from previous close
        atr14 = self._calculate_atr(daily_data, window=14)
        if len(atr14) > 0 and len(daily_data) > 0:
            last_close = daily_data.iloc[-1]['close']
            features['atr14pct'] = float(atr14.iloc[-1] / last_close if last_close > 0 else np.nan)
        else:
            features['atr14pct'] = np.nan
        
        # Calculate EMA (20-day) slope - forward-fill from previous close
        ema20 = daily_data['close'].ewm(span=20, adjust=False).mean()
        if len(ema20) >= 2:
            features['ema20_slope'] = float((ema20.iloc[-1] - ema20.iloc[-2]) / ema20.iloc[-2])
        else:
            features['ema20_slope'] = np.nan
            
        return features
    
    def _calculate_vwap_features(self, day_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate VWAP deviation features."""
        features = {}
        
        if day_data.empty:
            return {'vwap_dev': np.nan}
        
        # Calculate session VWAP
        if 'vwap' in day_data.columns:
            # Use pre-calculated VWAP if available
            session_vwap = day_data['vwap'].iloc[-1]
        else:
            # Calculate VWAP manually
            typical_price = (day_data['high'] + day_data['low'] + day_data['close']) / 3
            vwap_num = (typical_price * day_data['volume']).cumsum()
            vwap_den = day_data['volume'].cumsum()
            session_vwap = (vwap_num / vwap_den).iloc[-1]
        
        # VWAP deviation
        last_close = day_data['close'].iloc[-1]
        features['vwap_dev'] = float(last_close - session_vwap)
        
        return features
    
    def _calculate_forward_return_label(self, day_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate forward return label (3-hour forward return > 0)."""
        features = {}
        
        if day_data.empty:
            return {'y': np.nan, 'forward_return': np.nan}
        
        # Find the 10:00 price (entry point)
        entry_time = pd.to_datetime(self.opening_range_end, format='%H:%M').time()
        entry_mask = day_data['timestamp'].dt.time >= entry_time
        entry_data = day_data[entry_mask]
        
        if entry_data.empty:
            return {'y': np.nan, 'forward_return': np.nan}
        
        entry_price = entry_data.iloc[0]['close']
        
        # Find the exit price (15:55 or 3 hours later)
        exit_mask = day_data['timestamp'].dt.time <= self.exit_time_obj
        exit_data = day_data[exit_mask]
        
        if exit_data.empty:
            return {'y': np.nan, 'forward_return': np.nan}
        
        exit_price = exit_data.iloc[-1]['close']
        
        # Calculate forward return
        forward_return = (exit_price - entry_price) / entry_price
        features['forward_return'] = forward_return
        
        # Binary label (1 if positive return, 0 otherwise)
        features['y'] = int(1 if forward_return > 0 else 0)
        
        return features
    
    def _calculate_context_features(self, day_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate additional context features."""
        features = {}
        
        if day_data.empty:
            return {}
        
        # Volume profile features
        total_volume = day_data['volume'].sum()
        or_mask = day_data['timestamp'].dt.time <= self.opening_range_end_time
        or_volume = day_data[or_mask]['volume'].sum()
        
        features['or_volume_pct'] = or_volume / total_volume if total_volume > 0 else 0
        
        # Price momentum features
        if len(day_data) >= 2:
            first_price = day_data.iloc[0]['open']
            or_end_price = day_data[or_mask].iloc[-1]['close'] if or_mask.any() else first_price
            
            features['or_momentum'] = (or_end_price - first_price) / first_price if first_price > 0 else 0
        else:
            features['or_momentum'] = 0
        
        # Volatility features
        if len(day_data) > 1:
            returns = day_data['close'].pct_change().dropna()
            features['intraday_volatility'] = returns.std()
        else:
            features['intraday_volatility'] = 0
            
        return features
    
    def _create_daily_data(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """Create daily OHLCV data from minute data."""
        if minute_df.empty:
            return pd.DataFrame()
        
        # Group by date
        daily_data = minute_df.groupby(minute_df['timestamp'].dt.date).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        daily_data = daily_data.rename(columns={'timestamp': 'date'})
        return daily_data
    
    def _calculate_atr(self, daily_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        if len(daily_data) < 2:
            return pd.Series(dtype=float)
        
        # Calculate True Range
        high_low = daily_data['high'] - daily_data['low']
        high_close = np.abs(daily_data['high'] - daily_data['close'].shift(1))
        low_close = np.abs(daily_data['low'] - daily_data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR (exponential moving average of True Range)
        atr = true_range.ewm(span=window, adjust=False).mean()
        
        return atr


def build_features(sym: str, date: date, minute_df: pd.DataFrame) -> pd.Series:
    """
    Build features for a specific symbol and date.
    
    Computes all columns listed in spec §4:
    or_high, or_low, or_range, or_vol,
    atr14pct, ema20_slope, vwap_dev, label y
    
    Uses 09:30–10:00 window for OR fields
    Forward-fills higher-TF columns (ATR, EMA) from previous closes
    
    Args:
        sym: Stock symbol
        date: Target date for feature calculation
        minute_df: DataFrame with minute-level data
        
    Returns:
        Series with calculated features (dtype: float or int as appropriate)
    """
    builder = FeatureBuilder(opening_range_end="10:00")
    return builder.build_features(sym, date, minute_df)


def session_slice(df: pd.DataFrame, date: date, tz: Optional[pytz.BaseTzInfo] = None) -> pd.DataFrame:
    """
    Helper function to slice DataFrame to a specific trading session.
    
    Simplifies tests by extracting data for a single trading day.
    
    Args:
        df: DataFrame with 'timestamp' column
        date: Target trading date
        tz: Timezone (defaults to America/New_York)
        
    Returns:
        DataFrame filtered to the specified trading session
    """
    if tz is None:
        tz = get_market_timezone()
    
    # Convert date to timezone-aware bounds
    start_time = pd.Timestamp.combine(date, time(9, 25)).tz_localize(tz)
    end_time = pd.Timestamp.combine(date, time(16, 0)).tz_localize(tz)
    
    # Filter DataFrame
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    return df[mask].copy()


def build_features_batch(
    symbols: list,
    dates: list,
    minute_data: Dict[str, pd.DataFrame],
    opening_range_end: str = "10:00"
) -> pd.DataFrame:
    """
    Build features for multiple symbols and dates.
    
    Args:
        symbols: List of stock symbols
        dates: List of dates
        minute_data: Dictionary mapping symbols to minute DataFrames
        opening_range_end: End time for opening range
        
    Returns:
        DataFrame with features for all symbol-date combinations
    """
    builder = FeatureBuilder(opening_range_end=opening_range_end)
    
    features_list = []
    
    for symbol in symbols:
        if symbol not in minute_data:
            builder.log_warning(f"No minute data found for {symbol}")
            continue
            
        symbol_df = minute_data[symbol]
        
        for date in dates:
            features = builder.build_features(symbol, date, symbol_df)
            
            if not features.empty:
                features_list.append(features)
    
    if features_list:
        return pd.DataFrame(features_list).reset_index(drop=True)
    else:
        return pd.DataFrame() 