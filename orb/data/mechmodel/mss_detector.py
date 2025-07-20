"""
Market Structure Shift (MSS) detection for trend change identification.

Detects when market structure changes from bullish to bearish or vice versa
based on swing high/low patterns and their sequential relationships.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

from .swing_utils import swings, get_recent_swings


def detect_mss(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detect Market Structure Shifts (MSS) in price data.
    
    Market Structure Shift occurs when:
    - Bullish MSS: Price breaks above a previous swing high after making a higher low
    - Bearish MSS: Price breaks below a previous swing low after making a lower high
    
    This indicates a potential change in market trend direction.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Lookback period for swing detection
        
    Returns:
        DataFrame copy with added column:
        - mss_flip: Market structure change (+1 bullish MSS, -1 bearish MSS, 0 no change)
        
    Example:
        >>> data = pd.DataFrame({
        ...     'high': [100, 102, 101, 105, 103, 102, 107, 106],
        ...     'low': [98, 100, 99, 103, 101, 100, 105, 104],
        ...     'close': [99, 101, 100, 104, 102, 101, 106, 105]
        ... })
        >>> result = detect_mss(data)
        >>> result['mss_flip']
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
    
    # Initialize MSS column
    result['mss_flip'] = 0
    
    # Need sufficient data for swing analysis
    min_data_needed = lookback * 3
    if n < min_data_needed:
        return result
    
    # Detect swings first
    swing_data = swings(result, lookback=lookback)
    
    # Get all swing points for analysis
    swing_highs = []
    swing_lows = []
    
    for i, row in swing_data.iterrows():
        if row['swing_high']:
            swing_highs.append((i, row['high']))
        if row['swing_low']:
            swing_lows.append((i, row['low']))
    
    # Need at least 2 swings of each type for MSS detection
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return result
    
    # Track current market structure state
    current_structure = None  # 'bullish' or 'bearish'
    
    # Analyze price action for MSS patterns
    for i in range(lookback * 2, n):
        current_bar = result.iloc[i]
        
        # Get swings up to current bar
        recent_data = swing_data.iloc[:i+1]
        recent_swings = get_recent_swings(recent_data, lookback=lookback, max_swings=4)
        
        if len(recent_swings['highs']) < 2 or len(recent_swings['lows']) < 2:
            continue
        
        # Check for Bullish MSS
        bullish_mss = _check_bullish_mss(
            current_bar, recent_swings, i
        )
        
        # Check for Bearish MSS  
        bearish_mss = _check_bearish_mss(
            current_bar, recent_swings, i
        )
        
        # Set MSS flags
        if bullish_mss and current_structure != 'bullish':
            result.iloc[i, result.columns.get_loc('mss_flip')] = 1
            current_structure = 'bullish'
        elif bearish_mss and current_structure != 'bearish':
            result.iloc[i, result.columns.get_loc('mss_flip')] = -1
            current_structure = 'bearish'
    
    return result


def _check_bullish_mss(current_bar: pd.Series, recent_swings: dict, current_idx: int) -> bool:
    """
    Check for bullish Market Structure Shift pattern.
    
    Bullish MSS criteria:
    1. Price breaks above a previous swing high
    2. There was a higher low formed after that swing high
    3. Current close is above the swing high level
    
    Args:
        current_bar: Current price bar
        recent_swings: Dictionary with recent swing data
        current_idx: Current bar index
        
    Returns:
        True if bullish MSS detected
    """
    if len(recent_swings['highs']) < 2 or len(recent_swings['lows']) < 1:
        return False
    
    # Get recent swing highs (most recent first)
    recent_highs = sorted(recent_swings['highs'], key=lambda x: x[0], reverse=True)
    recent_lows = sorted(recent_swings['lows'], key=lambda x: x[0], reverse=True)
    
    # Need at least 2 swing highs
    if len(recent_highs) < 2:
        return False
    
    # Get the two most recent swing highs
    latest_high_idx, latest_high_price = recent_highs[0]
    previous_high_idx, previous_high_price = recent_highs[1]
    
    # Check if current price breaks above a previous swing high
    if current_bar['close'] <= previous_high_price:
        return False
    
    # Check if there's a higher low after the swing high we're breaking
    higher_low_found = False
    for low_idx, low_price in recent_lows:
        # Low must be after the swing high we're breaking
        if low_idx > previous_high_idx:
            # Check if this is a higher low compared to previous lows
            for other_low_idx, other_low_price in recent_lows:
                if other_low_idx < previous_high_idx and low_price > other_low_price:
                    higher_low_found = True
                    break
    
    return higher_low_found


def _check_bearish_mss(current_bar: pd.Series, recent_swings: dict, current_idx: int) -> bool:
    """
    Check for bearish Market Structure Shift pattern.
    
    Bearish MSS criteria:
    1. Price breaks below a previous swing low
    2. There was a lower high formed after that swing low
    3. Current close is below the swing low level
    
    Args:
        current_bar: Current price bar
        recent_swings: Dictionary with recent swing data
        current_idx: Current bar index
        
    Returns:
        True if bearish MSS detected
    """
    if len(recent_swings['lows']) < 2 or len(recent_swings['highs']) < 1:
        return False
    
    # Get recent swing lows (most recent first)
    recent_lows = sorted(recent_swings['lows'], key=lambda x: x[0], reverse=True)
    recent_highs = sorted(recent_swings['highs'], key=lambda x: x[0], reverse=True)
    
    # Need at least 2 swing lows
    if len(recent_lows) < 2:
        return False
    
    # Get the two most recent swing lows
    latest_low_idx, latest_low_price = recent_lows[0]
    previous_low_idx, previous_low_price = recent_lows[1]
    
    # Check if current price breaks below a previous swing low
    if current_bar['close'] >= previous_low_price:
        return False
    
    # Check if there's a lower high after the swing low we're breaking
    lower_high_found = False
    for high_idx, high_price in recent_highs:
        # High must be after the swing low we're breaking
        if high_idx > previous_low_idx:
            # Check if this is a lower high compared to previous highs
            for other_high_idx, other_high_price in recent_highs:
                if other_high_idx < previous_low_idx and high_price < other_high_price:
                    lower_high_found = True
                    break
    
    return lower_high_found


def get_mss_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of detected Market Structure Shifts.
    
    Args:
        df: DataFrame with MSS detection already applied
        
    Returns:
        Dictionary with MSS statistics:
        - total_mss: Total number of MSS events
        - bullish_mss: Number of bullish MSS events
        - bearish_mss: Number of bearish MSS events
        - last_mss: Last MSS direction (1, -1, or 0)
        - mss_frequency: Average bars between MSS events
    """
    if 'mss_flip' not in df.columns:
        return {
            'total_mss': 0,
            'bullish_mss': 0,
            'bearish_mss': 0,
            'last_mss': 0,
            'mss_frequency': 0
        }
    
    mss_events = df[df['mss_flip'] != 0]
    
    if mss_events.empty:
        return {
            'total_mss': 0,
            'bullish_mss': 0,
            'bearish_mss': 0,
            'last_mss': 0,
            'mss_frequency': 0
        }
    
    total_mss = len(mss_events)
    bullish_mss = len(mss_events[mss_events['mss_flip'] == 1])
    bearish_mss = len(mss_events[mss_events['mss_flip'] == -1])
    last_mss = mss_events['mss_flip'].iloc[-1]
    
    # Calculate average frequency (bars between MSS events)
    if total_mss > 1:
        mss_indices = mss_events.index.tolist()
        intervals = [mss_indices[i] - mss_indices[i-1] for i in range(1, len(mss_indices))]
        mss_frequency = np.mean(intervals)
    else:
        mss_frequency = len(df)
    
    return {
        'total_mss': total_mss,
        'bullish_mss': bullish_mss,
        'bearish_mss': bearish_mss,
        'last_mss': int(last_mss),
        'mss_frequency': mss_frequency
    }


def validate_mss_signal(df: pd.DataFrame, mss_idx: int, confirmation_bars: int = 3) -> bool:
    """
    Validate an MSS signal with price confirmation.
    
    Args:
        df: DataFrame with price data
        mss_idx: Index where MSS occurred
        confirmation_bars: Number of bars to check for confirmation
        
    Returns:
        True if MSS is confirmed by subsequent price action
    """
    if mss_idx >= len(df) or 'mss_flip' not in df.columns:
        return False
    
    mss_direction = df.iloc[mss_idx]['mss_flip']
    if mss_direction == 0:
        return False
    
    # Check confirmation in subsequent bars
    end_idx = min(mss_idx + confirmation_bars + 1, len(df))
    confirmation_data = df.iloc[mss_idx+1:end_idx]
    
    if confirmation_data.empty:
        return False
    
    if mss_direction == 1:  # Bullish MSS
        # Look for continued higher closes
        mss_close = df.iloc[mss_idx]['close']
        higher_closes = (confirmation_data['close'] > mss_close).sum()
        return higher_closes >= (confirmation_bars // 2)
    
    elif mss_direction == -1:  # Bearish MSS
        # Look for continued lower closes
        mss_close = df.iloc[mss_idx]['close']
        lower_closes = (confirmation_data['close'] < mss_close).sum()
        return lower_closes >= (confirmation_bars // 2)
    
    return False 