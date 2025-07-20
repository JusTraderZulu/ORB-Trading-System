"""
Institutional Fair Value Gap (iFVG) detection for order flow analysis.

Identifies significant price gaps with institutional volume characteristics
that price later revisits, indicating potential support/resistance zones.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict


def find_ifvg(df: pd.DataFrame, max_look_ahead: int = 20) -> pd.DataFrame:
    """
    Find Institutional Fair Value Gaps (iFVG) in price data.
    
    iFVG criteria (as specified by user):
    1. Three-candle gap must be ≥ 0.05% of price
    2. Gap-creating candle volume ≥ 1.3x its 20-bar average  
    3. Price must trade back through the gap to confirm it
    4. Save midpoint, top, bottom, and size when confirmed
    
    Args:
        df: DataFrame with OHLC and volume data
        max_look_ahead: Maximum bars to look ahead for gap fill confirmation
        
    Returns:
        DataFrame copy with added columns:
        - ifvg_mid: Midpoint of confirmed iFVG (NaN if no gap)
        - ifvg_size_pct: Size of iFVG as percentage of price
        - ifvg_top: Top of the iFVG 
        - ifvg_bottom: Bottom of the iFVG
        - ifvg_confirmed: Whether gap was filled/confirmed
        
    Example:
        >>> data = pd.DataFrame({
        ...     'high': [100, 102, 105, 103, 104],
        ...     'low': [98, 100, 102, 101, 102],
        ...     'close': [99, 101, 104, 102, 103],
        ...     'volume': [1000, 1500, 3000, 1200, 1100]
        ... })
        >>> result = find_ifvg(data)
    """
    if df.empty or len(df) < 3:
        return df.copy()
    
    # Validate required columns
    required_cols = ['high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check if volume is available (required for institutional validation)
    has_volume = 'volume' in df.columns
    if not has_volume:
        # For FX without real volume, use tick count or number of trades
        if 'num_trades' in df.columns:
            df = df.copy()
            df['volume'] = df['num_trades']
            has_volume = True
        else:
            # Create synthetic volume based on price movement
            df = df.copy()
            df['volume'] = (df['high'] - df['low']) * 1000  # Synthetic volume
            has_volume = True
    
    result = df.copy()
    n = len(result)
    
    # Initialize iFVG columns
    result['ifvg_mid'] = np.nan
    result['ifvg_size_pct'] = 0.0
    result['ifvg_top'] = np.nan
    result['ifvg_bottom'] = np.nan
    result['ifvg_confirmed'] = False
    
    # Calculate 20-period rolling average volume for institutional validation
    result['volume_ma20'] = result['volume'].rolling(window=20, min_periods=5).mean()
    
    # Scan for potential iFVGs (need at least 3 bars)
    for i in range(2, min(n - max_look_ahead, n)):
        # Three-candle pattern: bars i-2, i-1, i
        bar1 = result.iloc[i-2]  # First bar
        bar2 = result.iloc[i-1]  # Gap-creating bar (middle)
        bar3 = result.iloc[i]    # Third bar
        
        # Check for gap patterns
        bullish_gap = _check_bullish_gap(bar1, bar2, bar3)
        bearish_gap = _check_bearish_gap(bar1, bar2, bar3)
        
        if not bullish_gap and not bearish_gap:
            continue
        
        # Determine gap characteristics
        if bullish_gap:
            gap_top = bar2['low']    # Bottom of middle bar
            gap_bottom = bar1['high']  # Top of first bar
            gap_type = 'bullish'
        else:  # bearish_gap
            gap_top = bar1['low']    # Bottom of first bar  
            gap_bottom = bar2['high']  # Top of middle bar
            gap_type = 'bearish'
        
        gap_size = gap_top - gap_bottom
        gap_mid = (gap_top + gap_bottom) / 2
        
        # Validate gap size (≥ 0.05% of price)
        reference_price = bar2['close']
        gap_size_pct = (gap_size / reference_price) * 100
        
        if gap_size_pct < 0.05:
            continue  # Gap too small
        
        # Validate institutional volume (≥ 1.3x 20-bar average)
        if has_volume and not pd.isna(bar2['volume_ma20']):
            volume_ratio = bar2['volume'] / bar2['volume_ma20']
            if volume_ratio < 1.3:
                continue  # Volume not institutional
        
        # Look ahead to see if gap gets filled (price trades through it)
        gap_filled = False
        fill_idx = None
        
        end_idx = min(i + max_look_ahead + 1, n)
        for j in range(i + 1, end_idx):
            future_bar = result.iloc[j]
            
            # Check if price trades through the gap
            if bullish_gap:
                # For bullish gap, look for price to trade back down through gap
                if future_bar['low'] <= gap_bottom:
                    gap_filled = True
                    fill_idx = j
                    break
            else:  # bearish gap
                # For bearish gap, look for price to trade back up through gap
                if future_bar['high'] >= gap_top:
                    gap_filled = True
                    fill_idx = j
                    break
        
        # Only record confirmed iFVGs (gaps that get filled)
        if gap_filled and fill_idx is not None:
            # Set iFVG data at the gap formation bar (bar2)
            idx = i - 1  # Middle bar where gap was created
            result.iloc[idx, result.columns.get_loc('ifvg_mid')] = gap_mid
            result.iloc[idx, result.columns.get_loc('ifvg_size_pct')] = gap_size_pct
            result.iloc[idx, result.columns.get_loc('ifvg_top')] = gap_top
            result.iloc[idx, result.columns.get_loc('ifvg_bottom')] = gap_bottom
            result.iloc[idx, result.columns.get_loc('ifvg_confirmed')] = True
    
    # Clean up temporary columns
    if 'volume_ma20' in result.columns:
        result = result.drop('volume_ma20', axis=1)
    
    return result


def _check_bullish_gap(bar1: pd.Series, bar2: pd.Series, bar3: pd.Series) -> bool:
    """
    Check for bullish gap pattern (gap up).
    
    Bullish gap: bar2.low > bar1.high and bar3.low > bar1.high
    
    Args:
        bar1: First bar
        bar2: Middle bar (potential gap creator)
        bar3: Third bar
        
    Returns:
        True if bullish gap pattern detected
    """
    # Gap up: middle bar's low is above first bar's high
    gap_exists = bar2['low'] > bar1['high']
    
    # Third bar should also maintain the gap (not immediately fill)
    gap_maintained = bar3['low'] > bar1['high']
    
    return gap_exists and gap_maintained


def _check_bearish_gap(bar1: pd.Series, bar2: pd.Series, bar3: pd.Series) -> bool:
    """
    Check for bearish gap pattern (gap down).
    
    Bearish gap: bar2.high < bar1.low and bar3.high < bar1.low
    
    Args:
        bar1: First bar
        bar2: Middle bar (potential gap creator)
        bar3: Third bar
        
    Returns:
        True if bearish gap pattern detected
    """
    # Gap down: middle bar's high is below first bar's low
    gap_exists = bar2['high'] < bar1['low']
    
    # Third bar should also maintain the gap (not immediately fill)
    gap_maintained = bar3['high'] < bar1['low']
    
    return gap_exists and gap_maintained


def get_ifvg_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get summary statistics of detected iFVGs.
    
    Args:
        df: DataFrame with iFVG detection already applied
        
    Returns:
        Dictionary with iFVG statistics
    """
    if 'ifvg_confirmed' not in df.columns:
        return {
            'total_ifvgs': 0,
            'avg_gap_size_pct': 0.0,
            'max_gap_size_pct': 0.0,
            'recent_ifvg_mid': None,
            'recent_ifvg_size': 0.0
        }
    
    confirmed_ifvgs = df[df['ifvg_confirmed'] == True]
    
    if confirmed_ifvgs.empty:
        return {
            'total_ifvgs': 0,
            'avg_gap_size_pct': 0.0,
            'max_gap_size_pct': 0.0,
            'recent_ifvg_mid': None,
            'recent_ifvg_size': 0.0
        }
    
    return {
        'total_ifvgs': len(confirmed_ifvgs),
        'avg_gap_size_pct': confirmed_ifvgs['ifvg_size_pct'].mean(),
        'max_gap_size_pct': confirmed_ifvgs['ifvg_size_pct'].max(),
        'recent_ifvg_mid': confirmed_ifvgs['ifvg_mid'].iloc[-1],
        'recent_ifvg_size': confirmed_ifvgs['ifvg_size_pct'].iloc[-1]
    }


def get_active_ifvgs(df: pd.DataFrame, current_price: float, 
                     proximity_pct: float = 2.0) -> List[Dict]:
    """
    Get iFVGs that are near current price and potentially relevant.
    
    Args:
        df: DataFrame with iFVG detection
        current_price: Current market price
        proximity_pct: Consider iFVGs within this percentage of current price
        
    Returns:
        List of relevant iFVG dictionaries
    """
    if 'ifvg_confirmed' not in df.columns:
        return []
    
    confirmed_ifvgs = df[df['ifvg_confirmed'] == True]
    
    if confirmed_ifvgs.empty:
        return []
    
    active_ifvgs = []
    proximity_threshold = current_price * (proximity_pct / 100)
    
    for idx, row in confirmed_ifvgs.iterrows():
        ifvg_mid = row['ifvg_mid']
        
        # Check if iFVG is near current price
        distance = abs(current_price - ifvg_mid)
        
        if distance <= proximity_threshold:
            active_ifvgs.append({
                'index': idx,
                'midpoint': ifvg_mid,
                'top': row['ifvg_top'],
                'bottom': row['ifvg_bottom'],
                'size_pct': row['ifvg_size_pct'],
                'distance_from_price': distance,
                'distance_pct': (distance / current_price) * 100,
                'above_price': ifvg_mid > current_price
            })
    
    # Sort by proximity to current price
    active_ifvgs.sort(key=lambda x: x['distance_from_price'])
    
    return active_ifvgs


def validate_ifvg_quality(df: pd.DataFrame, ifvg_idx: int) -> Dict[str, any]:
    """
    Validate the quality of a detected iFVG.
    
    Args:
        df: DataFrame with price data
        ifvg_idx: Index of the iFVG to validate
        
    Returns:
        Dictionary with quality metrics
    """
    if ifvg_idx >= len(df) or 'ifvg_confirmed' not in df.columns:
        return {'valid': False, 'reason': 'Invalid index or missing iFVG data'}
    
    ifvg_row = df.iloc[ifvg_idx]
    
    if not ifvg_row['ifvg_confirmed']:
        return {'valid': False, 'reason': 'iFVG not confirmed'}
    
    # Calculate quality metrics
    gap_size_pct = ifvg_row['ifvg_size_pct']
    
    # Check volume context if available
    volume_quality = 'unknown'
    if 'volume' in df.columns:
        pre_gap_volume = df.iloc[max(0, ifvg_idx-5):ifvg_idx]['volume'].mean()
        gap_volume = ifvg_row['volume']
        
        if gap_volume > pre_gap_volume * 1.5:
            volume_quality = 'high'
        elif gap_volume > pre_gap_volume * 1.2:
            volume_quality = 'medium'
        else:
            volume_quality = 'low'
    
    # Determine overall quality
    quality_score = 0
    
    if gap_size_pct >= 0.1:  # Large gap
        quality_score += 3
    elif gap_size_pct >= 0.07:  # Medium gap
        quality_score += 2
    else:  # Small gap
        quality_score += 1
    
    if volume_quality == 'high':
        quality_score += 2
    elif volume_quality == 'medium':
        quality_score += 1
    
    return {
        'valid': True,
        'quality_score': quality_score,
        'gap_size_pct': gap_size_pct,
        'volume_quality': volume_quality,
        'quality_tier': 'high' if quality_score >= 4 else 'medium' if quality_score >= 2 else 'low'
    } 