"""
Liquidity sweep detection for market structure analysis.

Detects when price sweeps beyond swing levels and measures the strength
of the move relative to Average True Range (ATR).
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings

try:
    import ta
except ImportError:
    warnings.warn("ta library not installed. Install with: pip install ta")
    ta = None

from .swing_utils import swings, get_recent_swings


def detect_sweep(df: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """
    Detect liquidity sweeps beyond swing levels with ATR validation.
    
    A liquidity sweep occurs when price moves beyond a recent swing high/low,
    potentially hunting stop losses before reversing. The strength is measured
    relative to ATR to normalize across different instruments and volatility regimes.
    
    Args:
        df: DataFrame with OHLC data and timestamp
        atr_window: Window for ATR calculation (default 14)
        
    Returns:
        DataFrame copy with added columns:
        - brk_dir: Breakout direction (+1 bullish, -1 bearish, 0 none)
        - brk_stretch_pct_atr: Breakout strength as percentage of ATR
        - brk_minutes: Minutes since breakout occurred
        - atr: Average True Range values
        
    Example:
        >>> data = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        ...     'high': np.random.uniform(100, 110, 100),
        ...     'low': np.random.uniform(95, 105, 100),
        ...     'close': np.random.uniform(98, 108, 100),
        ...     'volume': np.random.randint(1000, 5000, 100)
        ... })
        >>> result = detect_sweep(data)
    """
    if df.empty:
        return df.copy()
    
    # Validate required columns
    required_cols = ['high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    result = df.copy()
    n = len(result)
    
    # Calculate ATR
    result['atr'] = _calculate_atr(result, window=atr_window)
    
    # Initialize sweep detection columns
    result['brk_dir'] = 0
    result['brk_stretch_pct_atr'] = 0.0
    result['brk_minutes'] = 0
    
    # Need enough data for swing analysis and ATR
    min_data_needed = max(atr_window * 2, 50)  # Ensure sufficient data
    if n < min_data_needed:
        return result
    
    # Detect swings first (using smaller lookback for more sensitive detection)
    swing_lookback = min(10, n // 10)  # Adaptive lookback based on data size
    swing_data = swings(result, lookback=swing_lookback)
    
    # Track active sweeps and their timing
    active_sweep = None
    sweep_start_time = None
    
    # Scan through data looking for sweep patterns
    for i in range(swing_lookback + atr_window, n):
        current_bar = swing_data.iloc[i]
        current_atr = current_bar['atr']
        
        if pd.isna(current_atr) or current_atr <= 0:
            continue
            
        # Get recent swings up to current bar
        recent_data = swing_data.iloc[:i+1]
        recent_swings = get_recent_swings(recent_data, lookback=swing_lookback, max_swings=5)
        
        # Check for bullish sweep (breaking above recent swing high)
        if recent_swings['last_swing_high'] is not None:
            swing_high = recent_swings['last_swing_high']
            
            # Current high breaks above swing high
            if current_bar['high'] > swing_high:
                stretch = current_bar['high'] - swing_high
                stretch_pct_atr = (stretch / current_atr) * 100
                
                # Set sweep information
                result.iloc[i, result.columns.get_loc('brk_dir')] = 1
                result.iloc[i, result.columns.get_loc('brk_stretch_pct_atr')] = stretch_pct_atr
                
                # Track timing if this is a new sweep
                if active_sweep != 'bullish' or sweep_start_time is None:
                    active_sweep = 'bullish'
                    sweep_start_time = i
                
                # Calculate minutes since sweep started
                minutes_since = i - sweep_start_time
                result.iloc[i, result.columns.get_loc('brk_minutes')] = minutes_since
        
        # Check for bearish sweep (breaking below recent swing low)
        if recent_swings['last_swing_low'] is not None:
            swing_low = recent_swings['last_swing_low']
            
            # Current low breaks below swing low
            if current_bar['low'] < swing_low:
                stretch = swing_low - current_bar['low']
                stretch_pct_atr = (stretch / current_atr) * 100
                
                # Set sweep information
                result.iloc[i, result.columns.get_loc('brk_dir')] = -1
                result.iloc[i, result.columns.get_loc('brk_stretch_pct_atr')] = stretch_pct_atr
                
                # Track timing if this is a new sweep
                if active_sweep != 'bearish' or sweep_start_time is None:
                    active_sweep = 'bearish'
                    sweep_start_time = i
                
                # Calculate minutes since sweep started
                minutes_since = i - sweep_start_time
                result.iloc[i, result.columns.get_loc('brk_minutes')] = minutes_since
        
        # Reset sweep tracking if price returns to range
        if active_sweep is not None:
            # Check if sweep has been invalidated (price returned to range)
            if active_sweep == 'bullish' and recent_swings['last_swing_high'] is not None:
                if current_bar['close'] < recent_swings['last_swing_high']:
                    # Bullish sweep invalidated
                    active_sweep = None
                    sweep_start_time = None
            elif active_sweep == 'bearish' and recent_swings['last_swing_low'] is not None:
                if current_bar['close'] > recent_swings['last_swing_low']:
                    # Bearish sweep invalidated
                    active_sweep = None
                    sweep_start_time = None
    
    return result


def _calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        window: Lookback window for ATR calculation
        
    Returns:
        Series with ATR values
    """
    if ta is not None:
        # Use ta library if available (more robust)
        try:
            return ta.volatility.average_true_range(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                window=window
            )
        except Exception:
            pass  # Fall back to manual calculation
    
    # Manual ATR calculation
    if len(df) < 2:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate ATR as exponential moving average of True Range
    atr = true_range.ewm(span=window, adjust=False).mean()
    
    return atr


def get_sweep_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of detected sweeps.
    
    Args:
        df: DataFrame with sweep detection already applied
        
    Returns:
        Dictionary with sweep statistics:
        - total_sweeps: Total number of sweep events
        - bullish_sweeps: Number of bullish sweeps
        - bearish_sweeps: Number of bearish sweeps
        - avg_stretch_atr: Average stretch relative to ATR
        - max_stretch_atr: Maximum stretch relative to ATR
    """
    if 'brk_dir' not in df.columns:
        return {
            'total_sweeps': 0,
            'bullish_sweeps': 0,
            'bearish_sweeps': 0,
            'avg_stretch_atr': 0.0,
            'max_stretch_atr': 0.0
        }
    
    sweep_bars = df[df['brk_dir'] != 0]
    
    if sweep_bars.empty:
        return {
            'total_sweeps': 0,
            'bullish_sweeps': 0,
            'bearish_sweeps': 0,
            'avg_stretch_atr': 0.0,
            'max_stretch_atr': 0.0
        }
    
    return {
        'total_sweeps': len(sweep_bars),
        'bullish_sweeps': len(sweep_bars[sweep_bars['brk_dir'] == 1]),
        'bearish_sweeps': len(sweep_bars[sweep_bars['brk_dir'] == -1]),
        'avg_stretch_atr': sweep_bars['brk_stretch_pct_atr'].mean(),
        'max_stretch_atr': sweep_bars['brk_stretch_pct_atr'].max()
    } 