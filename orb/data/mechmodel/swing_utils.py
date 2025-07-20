"""
Swing detection utilities for market structure analysis.

Identifies swing highs and lows using configurable lookback periods.
Foundation for liquidity sweep and market structure shift detection.
"""

import pandas as pd
import numpy as np
from typing import Optional


def swings(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Identify swing highs and lows in price data.
    
    A swing high is a bar whose high is the highest within the lookback window
    (lookback bars before and after). Similarly for swing lows.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        lookback: Number of bars to look before/after for swing validation
        
    Returns:
        DataFrame copy with added boolean columns:
        - swing_high: True if bar is a swing high
        - swing_low: True if bar is a swing low
        
    Example:
        >>> data = pd.DataFrame({
        ...     'high': [100, 102, 101, 105, 103, 102, 104],
        ...     'low': [99, 100, 99, 103, 101, 100, 102],
        ...     'close': [100, 101, 100, 104, 102, 101, 103]
        ... })
        >>> result = swings(data, lookback=2)
        >>> result[['swing_high', 'swing_low']]
    """
    if df.empty or len(df) < (2 * lookback + 1):
        # Not enough data for swing analysis
        result = df.copy()
        result['swing_high'] = False
        result['swing_low'] = False
        return result
    
    # Validate required columns
    required_cols = ['high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    result = df.copy()
    n = len(result)
    
    # Initialize swing columns
    result['swing_high'] = False
    result['swing_low'] = False
    
    # Check each potential swing point (excluding edges)
    for i in range(lookback, n - lookback):
        current_high = result.iloc[i]['high']
        current_low = result.iloc[i]['low']
        
        # Check if current bar is swing high
        # Must be highest high in (lookback before + current + lookback after) window
        window_start = i - lookback
        window_end = i + lookback + 1
        window_highs = result.iloc[window_start:window_end]['high']
        
        if current_high == window_highs.max():
            # Ensure it's strictly higher than surrounding bars (tie-breaking)
            left_max = result.iloc[window_start:i]['high'].max() if i > window_start else 0
            right_max = result.iloc[i+1:window_end]['high'].max() if i+1 < window_end else 0
            
            if current_high > left_max and current_high > right_max:
                result.iloc[i, result.columns.get_loc('swing_high')] = True
        
        # Check if current bar is swing low  
        # Must be lowest low in (lookback before + current + lookback after) window
        window_lows = result.iloc[window_start:window_end]['low']
        
        if current_low == window_lows.min():
            # Ensure it's strictly lower than surrounding bars (tie-breaking)
            left_min = result.iloc[window_start:i]['low'].min() if i > window_start else float('inf')
            right_min = result.iloc[i+1:window_end]['low'].min() if i+1 < window_end else float('inf')
            
            if current_low < left_min and current_low < right_min:
                result.iloc[i, result.columns.get_loc('swing_low')] = True
    
    return result


def get_recent_swings(df: pd.DataFrame, lookback: int = 20, max_swings: int = 10) -> dict:
    """
    Get the most recent swing highs and lows.
    
    Args:
        df: DataFrame with swing analysis already applied
        lookback: Lookback period used for swing detection
        max_swings: Maximum number of recent swings to return
        
    Returns:
        Dictionary with:
        - 'highs': List of (index, price) tuples for recent swing highs
        - 'lows': List of (index, price) tuples for recent swing lows
        - 'last_swing_high': Most recent swing high price or None
        - 'last_swing_low': Most recent swing low price or None
    """
    if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
        # Run swing detection if not already done
        df = swings(df, lookback=lookback)
    
    # Get recent swing highs
    swing_highs = df[df['swing_high']].tail(max_swings)
    recent_highs = [(idx, row['high']) for idx, row in swing_highs.iterrows()]
    
    # Get recent swing lows  
    swing_lows = df[df['swing_low']].tail(max_swings)
    recent_lows = [(idx, row['low']) for idx, row in swing_lows.iterrows()]
    
    return {
        'highs': recent_highs,
        'lows': recent_lows,
        'last_swing_high': recent_highs[-1][1] if recent_highs else None,
        'last_swing_low': recent_lows[-1][1] if recent_lows else None
    }


def validate_swing_break(df: pd.DataFrame, swing_level: float, 
                        direction: str, current_idx: int) -> bool:
    """
    Validate if a swing level has been definitively broken.
    
    Args:
        df: DataFrame with OHLC data
        swing_level: Price level of the swing high/low
        direction: 'high' for swing high break, 'low' for swing low break
        current_idx: Current bar index
        
    Returns:
        True if swing level has been broken with conviction
    """
    if current_idx >= len(df):
        return False
        
    current_bar = df.iloc[current_idx]
    
    if direction == 'high':
        # For swing high break, need close above the swing high
        return current_bar['close'] > swing_level
    elif direction == 'low':
        # For swing low break, need close below the swing low  
        return current_bar['close'] < swing_level
    else:
        raise ValueError("Direction must be 'high' or 'low'") 