"""
Feature engineering module for ORB trading system.

Creates opening range features, technical indicators, forward return labels,
and advanced mechanical model features from minute-level market data.

Implements build_features() and session_slice() functions per ORB spec §4.
Enhanced with Phase 4a mechanical model features (swing, sweep, MSS, SMT, iFVG).
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, date
from typing import Optional, Union, Dict, Any
import warnings
import pytz

from ..utils.logging import LoggingMixin
from ..utils.calendars import get_market_timezone, trading_days
from .mechmodel import run_mech_pipeline

# Feature version for tracking mechanical model updates
FEATURE_VERSION = "4.1.0"  # Updated for Phase 4a mechanical model features


class FeatureBuilder(LoggingMixin):
    """
    Feature builder for ORB trading system.
    
    Creates features including:
    - Opening range features (high, low, range, volume)
    - ORB signal detection (breakout, direction, timing, strength)
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
            
            # 2. ORB Signal Detection
            orb_signal_features = self._calculate_orb_signals(day_data, or_features)
            features.update(orb_signal_features)
            
            # 3. Technical Indicators
            tech_features = self._calculate_technical_features(symbol, date, minute_df)
            features.update(tech_features)
            
            # 4. VWAP Deviation
            vwap_features = self._calculate_vwap_features(day_data)
            features.update(vwap_features)
            
            # 5. Forward Return Label
            label_features = self._calculate_forward_return_label(day_data)
            features.update(label_features)
            
            # 6. Additional Context Features
            context_features = self._calculate_context_features(day_data)
            features.update(context_features)
            
            # 7. Mechanical Model Features (Phase 4a)
            mech_features = self._calculate_mechanical_features(symbol, date, minute_df)
            features.update(mech_features)
            
            # Add metadata
            features['symbol'] = symbol
            features['date'] = target_date
            features['timestamp'] = pd.to_datetime(f"{target_date} {self.opening_range_end}")
            features['feature_version'] = FEATURE_VERSION
            
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
    
    def _calculate_orb_signals(self, day_data: pd.DataFrame, or_features: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate ORB breakout signals and validation.
        
        Detects actual ORB signals after the opening range period ends:
        - Breakout detection (above or_high, below or_low)
        - Breakout direction and timing
        - Breakout strength and volume confirmation
        
        Args:
            day_data: Minute-level data for the trading day
            or_features: Previously calculated opening range features
            
        Returns:
            Dictionary with ORB signal features
        """
        features = {}
        
        # Extract OR levels from calculated features
        or_high = or_features.get('or_high', np.nan)
        or_low = or_features.get('or_low', np.nan)
        or_vol = or_features.get('or_vol', 0)
        
        if pd.isna(or_high) or pd.isna(or_low) or day_data.empty:
            return {
                'orb_breakout': 0,
                'orb_direction': 0,
                'orb_timing_minutes': np.nan,
                'orb_strength_pct': 0.0,
                'orb_volume_confirm': 0,
                'orb_failed_breakout': 0,
                'orb_max_extension_pct': 0.0
            }
        
        # Get post-OR data (after opening range ends)
        post_or_mask = day_data['timestamp'].dt.time > self.opening_range_end_time
        post_or_data = day_data[post_or_mask].copy()
        
        if post_or_data.empty:
            return {
                'orb_breakout': 0,
                'orb_direction': 0,
                'orb_timing_minutes': np.nan,
                'orb_strength_pct': 0.0,
                'orb_volume_confirm': 0,
                'orb_failed_breakout': 0,
                'orb_max_extension_pct': 0.0
            }
        
        # Initialize features
        features['orb_breakout'] = 0
        features['orb_direction'] = 0  # 1 = bullish (above OR), -1 = bearish (below OR)
        features['orb_timing_minutes'] = np.nan
        features['orb_strength_pct'] = 0.0
        features['orb_volume_confirm'] = 0
        features['orb_failed_breakout'] = 0
        features['orb_max_extension_pct'] = 0.0
        
        # Calculate average OR volume per minute for comparison
        or_mask = day_data['timestamp'].dt.time <= self.opening_range_end_time
        or_data = day_data[or_mask]
        avg_or_volume = or_vol / len(or_data) if len(or_data) > 0 else 1
        
        # Detect breakouts
        bullish_breakout = None  # First bar breaking above or_high
        bearish_breakout = None  # First bar breaking below or_low
        
        for idx, row in post_or_data.iterrows():
            # Check for bullish breakout (high > or_high)
            if bullish_breakout is None and row['high'] > or_high:
                bullish_breakout = row
                
            # Check for bearish breakout (low < or_low)  
            if bearish_breakout is None and row['low'] < or_low:
                bearish_breakout = row
                
            # Stop after finding first breakout in either direction
            if bullish_breakout is not None or bearish_breakout is not None:
                break
        
        # Determine which breakout occurred first (if any)
        first_breakout = None
        if bullish_breakout is not None and bearish_breakout is not None:
            # Both occurred, take the earlier one
            if bullish_breakout['timestamp'] <= bearish_breakout['timestamp']:
                first_breakout = ('bullish', bullish_breakout)
            else:
                first_breakout = ('bearish', bearish_breakout)
        elif bullish_breakout is not None:
            first_breakout = ('bullish', bullish_breakout)
        elif bearish_breakout is not None:
            first_breakout = ('bearish', bearish_breakout)
        
        # Calculate breakout features if breakout occurred
        if first_breakout is not None:
            direction, breakout_bar = first_breakout
            
            features['orb_breakout'] = 1
            features['orb_direction'] = 1 if direction == 'bullish' else -1
            
            # Calculate timing (minutes after OR end)
            or_end_time = pd.to_datetime(f"{breakout_bar['timestamp'].date()} {self.opening_range_end}")
            or_end_time = or_end_time.tz_localize(day_data['timestamp'].iloc[0].tz)
            timing_delta = breakout_bar['timestamp'] - or_end_time
            features['orb_timing_minutes'] = float(timing_delta.total_seconds() / 60)
            
            # Calculate breakout strength (how far beyond OR level)
            if direction == 'bullish':
                breakout_price = breakout_bar['high']
                breakout_strength = (breakout_price - or_high) / or_high
            else:
                breakout_price = breakout_bar['low'] 
                breakout_strength = (or_low - breakout_price) / or_low
                
            features['orb_strength_pct'] = float(breakout_strength * 100)
            
            # Volume confirmation (breakout bar volume vs average OR volume)
            breakout_volume = breakout_bar['volume']
            if avg_or_volume > 0:
                volume_ratio = breakout_volume / avg_or_volume
                features['orb_volume_confirm'] = 1 if volume_ratio >= 1.5 else 0
            
            # Calculate maximum extension from OR levels
            remaining_data = post_or_data[post_or_data['timestamp'] >= breakout_bar['timestamp']]
            
            if not remaining_data.empty:
                if direction == 'bullish':
                    max_high = remaining_data['high'].max()
                    max_extension = (max_high - or_high) / or_high
                else:
                    min_low = remaining_data['low'].min()
                    max_extension = (or_low - min_low) / or_low
                    
                features['orb_max_extension_pct'] = float(max_extension * 100)
            
            # Failed breakout detection
            # A breakout is considered "failed" if price reverses back into the OR within 30 minutes
            thirty_min_later = breakout_bar['timestamp'] + timedelta(minutes=30)
            reversal_data = remaining_data[remaining_data['timestamp'] <= thirty_min_later]
            
            if not reversal_data.empty:
                if direction == 'bullish':
                    # Failed if any bar's low goes back below or_high
                    failed = (reversal_data['low'] <= or_high).any()
                else:
                    # Failed if any bar's high goes back above or_low
                    failed = (reversal_data['high'] >= or_low).any()
                    
                features['orb_failed_breakout'] = 1 if failed else 0
        
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
    
    def _calculate_mechanical_features(
        self,
        symbol: str,
        date: Union[str, datetime, pd.Timestamp],
        minute_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate mechanical model features (Phase 4a).
        
        Runs the complete mechanical model pipeline on the day's minute data
        and extracts the last row's features for daily aggregation.
        
        Args:
            symbol: Stock/crypto symbol
            date: Target date
            minute_df: Full minute-level data
            
        Returns:
            Dictionary with mechanical model features
        """
        features = {}
        target_date = pd.to_datetime(date).date()
        
        try:
            # Filter minute data to target date
            if 'timestamp' in minute_df.columns and len(minute_df) > 0:
                date_mask = minute_df['timestamp'].dt.date == target_date
                day_minute_data = minute_df[date_mask].copy()
            else:
                self.log_warning(f"No minute data available for mechanical features on {target_date}")
                return self._get_empty_mechanical_features()
            
            if day_minute_data.empty:
                self.log_warning(f"No minute data found for {symbol} on {target_date}")
                return self._get_empty_mechanical_features()
            
            # Run mechanical model pipeline
            mech_result = run_mech_pipeline(
                min_df=day_minute_data,
                smt_pair=None,  # Would need secondary data in production
                symbol=symbol,
                liquid_session_only=True,  # Focus on 08:00-17:00 UTC
                swing_lookback=10,  # Smaller lookback for intraday
                atr_window=14,
                ifvg_lookahead=20
            )
            
            if mech_result.empty:
                return self._get_empty_mechanical_features()
            
            # Extract features from the last row (end-of-day summary)
            last_row = mech_result.iloc[-1]
            
            # Core mechanical features
            features['brk_dir'] = int(last_row.get('brk_dir', 0))
            features['brk_stretch_pct_atr'] = float(last_row.get('brk_stretch_pct_atr', 0.0))
            features['brk_minutes'] = int(last_row.get('brk_minutes', 0))
            features['mss_flip'] = int(last_row.get('mss_flip', 0))
            features['smt_divergence'] = int(last_row.get('smt_divergence', 0))
            features['ifvg_mid'] = float(last_row.get('ifvg_mid', 0.0)) if not pd.isna(last_row.get('ifvg_mid')) else 0.0
            features['ifvg_size_pct'] = float(last_row.get('ifvg_size_pct', 0.0))
            
            # Summary features (daily aggregations)
            features['sweep_count'] = int(last_row.get('sweep_count', 0))
            features['bullish_sweep_count'] = int(last_row.get('bullish_sweep_count', 0))
            features['bearish_sweep_count'] = int(last_row.get('bearish_sweep_count', 0))
            features['max_sweep_strength'] = float(last_row.get('max_sweep_strength', 0.0))
            features['mss_count'] = int(last_row.get('mss_count', 0))
            features['last_mss_direction'] = int(last_row.get('last_mss_direction', 0))
            features['ifvg_count'] = int(last_row.get('ifvg_count', 0))
            
            self.log_debug(f"Calculated {len(features)} mechanical features for {symbol} on {target_date}")
            
        except Exception as e:
            self.log_error(f"Error calculating mechanical features for {symbol} on {date}: {e}")
            features = self._get_empty_mechanical_features()
        
        return features
    
    def _get_empty_mechanical_features(self) -> Dict[str, Any]:
        """Return empty mechanical model features when calculation fails."""
        return {
            'brk_dir': 0,
            'brk_stretch_pct_atr': 0.0,
            'brk_minutes': 0,
            'mss_flip': 0,
            'smt_divergence': 0,
            'ifvg_mid': 0.0,
            'ifvg_size_pct': 0.0,
            'sweep_count': 0,
            'bullish_sweep_count': 0,
            'bearish_sweep_count': 0,
            'max_sweep_strength': 0.0,
            'mss_count': 0,
            'last_mss_direction': 0,
            'ifvg_count': 0
        }
    
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
    
    Computes all features for ORB trading including:
    - Opening Range: or_high, or_low, or_range, or_vol
    - ORB Signals: orb_breakout, orb_direction, orb_timing_minutes, orb_strength_pct,
                   orb_volume_confirm, orb_failed_breakout, orb_max_extension_pct
    - Technical: atr14pct, ema20_slope, vwap_dev
    - Labels: y (forward return > 0)
    
    Uses 09:30–10:00 window for OR fields
    Detects breakouts after 10:00 AM
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