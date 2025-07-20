"""
Mechanical model pipeline for running all detectors and aggregating results.

Coordinates swing detection, sweep analysis, MSS detection, SMT divergence,
and iFVG identification into a unified feature extraction pipeline.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import warnings

from .swing_utils import swings
from .sweep_detector import detect_sweep
from .mss_detector import detect_mss
from .smt_divergence import smt_flag
from .ifvg_finder import find_ifvg


def run_mech_pipeline(
    min_df: pd.DataFrame, 
    smt_pair: Optional[str] = None,
    symbol: str = '',
    liquid_session_only: bool = True,
    swing_lookback: int = 10,
    atr_window: int = 14,
    ifvg_lookahead: int = 20
) -> pd.DataFrame:
    """
    Run complete mechanical model analysis pipeline.
    
    Processes minute-level data through all mechanical model detectors:
    1. Swing detection (highs/lows)
    2. Liquidity sweep detection with ATR normalization
    3. Market Structure Shift (MSS) identification
    4. Smart Money Tool (SMT) divergence analysis
    5. Institutional Fair Value Gap (iFVG) detection
    
    Args:
        min_df: Minute-level OHLCV data
        smt_pair: Optional secondary instrument for SMT analysis
        symbol: Primary symbol name for SMT pair detection
        liquid_session_only: Filter to 08:00-17:00 UTC liquid session
        swing_lookback: Lookback period for swing detection
        atr_window: Window for ATR calculation
        ifvg_lookahead: Lookback for iFVG confirmation
        
    Returns:
        DataFrame with all mechanical model features added
        
    Example:
        >>> data = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
        ...     'high': np.random.uniform(100, 110, 1000),
        ...     'low': np.random.uniform(95, 105, 1000),
        ...     'close': np.random.uniform(98, 108, 1000),
        ...     'volume': np.random.randint(1000, 5000, 1000)
        ... })
        >>> result = run_mech_pipeline(data, symbol='BTC-USD')
    """
    if min_df.empty:
        return min_df.copy()
    
    # Validate required columns
    required_cols = ['high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in min_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    result = min_df.copy()
    
    # Filter to liquid session if requested (crypto/FX focus)
    if liquid_session_only and 'timestamp' in result.columns:
        result = _filter_liquid_session(result)
        if result.empty:
            warnings.warn("No data remaining after liquid session filter")
            return result
    
    # Ensure sufficient data for analysis
    min_required = max(swing_lookback * 3, atr_window * 2, 50)
    if len(result) < min_required:
        warnings.warn(f"Insufficient data for mechanical analysis. Need {min_required}, got {len(result)}")
        return _add_empty_mech_features(result)
    
    try:
        # 1. Swing Detection (foundation for other features)
        result = swings(result, lookback=swing_lookback)
        
        # 2. Liquidity Sweep Detection
        result = detect_sweep(result, atr_window=atr_window)
        
        # 3. Market Structure Shift Detection
        result = detect_mss(result, lookback=swing_lookback)
        
        # 4. Smart Money Tool Divergence (if secondary data available)
        if smt_pair is not None:
            # In production, would load secondary instrument data
            # For now, create placeholder SMT features
            smt_flags = smt_flag(result, df_secondary=None, primary_symbol=symbol)
        else:
            smt_flags = pd.Series([0] * len(result), index=result.index, name='smt_flag')
        
        result['smt_divergence'] = smt_flags
        
        # 5. Institutional Fair Value Gap Detection
        result = find_ifvg(result, max_look_ahead=ifvg_lookahead)
        
        # Add summary statistics as features
        result = _add_mech_summary_features(result)
        
    except Exception as e:
        warnings.warn(f"Error in mechanical pipeline: {e}")
        result = _add_empty_mech_features(result)
    
    return result


def _filter_liquid_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to liquid overlap session (08:00-17:00 UTC).
    
    This captures London + New York morning overlap, the most liquid
    session for crypto and FX markets.
    
    Args:
        df: DataFrame with timestamp column
        
    Returns:
        Filtered DataFrame
    """
    if 'timestamp' not in df.columns:
        return df
    
    # Ensure timestamp is timezone aware (assume UTC if naive)
    if df['timestamp'].dt.tz is None:
        df = df.copy()
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    else:
        df = df.copy()
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
    
    # Filter to 08:00-17:00 UTC
    mask = (
        (df['timestamp'].dt.hour >= 8) & 
        (df['timestamp'].dt.hour < 17)
    )
    
    return df[mask].copy()


def _add_empty_mech_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add empty mechanical model features when analysis cannot be performed."""
    result = df.copy()
    
    # Swing features
    if 'swing_high' not in result.columns:
        result['swing_high'] = False
        result['swing_low'] = False
    
    # Sweep features
    sweep_features = ['brk_dir', 'brk_stretch_pct_atr', 'brk_minutes']
    for feature in sweep_features:
        if feature not in result.columns:
            if feature == 'brk_dir':
                result[feature] = 0
            elif feature == 'brk_minutes':
                result[feature] = 0
            else:
                result[feature] = 0.0
    
    # MSS features
    if 'mss_flip' not in result.columns:
        result['mss_flip'] = 0
    
    # SMT features
    if 'smt_divergence' not in result.columns:
        result['smt_divergence'] = 0
    
    # iFVG features
    ifvg_features = ['ifvg_mid', 'ifvg_size_pct', 'ifvg_top', 'ifvg_bottom', 'ifvg_confirmed']
    for feature in ifvg_features:
        if feature not in result.columns:
            if feature == 'ifvg_confirmed':
                result[feature] = False
            elif feature == 'ifvg_size_pct':
                result[feature] = 0.0
            else:
                result[feature] = np.nan
    
    return result


def _add_mech_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add summary features from mechanical model analysis."""
    result = df.copy()
    n = len(result)
    
    # Summary features to add to the last row (for daily aggregation)
    summary_features = {}
    
    # Swing summary
    if 'swing_high' in result.columns and 'swing_low' in result.columns:
        summary_features['swing_high_count'] = result['swing_high'].sum()
        summary_features['swing_low_count'] = result['swing_low'].sum()
    
    # Sweep summary
    if 'brk_dir' in result.columns:
        sweeps = result[result['brk_dir'] != 0]
        summary_features['sweep_count'] = len(sweeps)
        summary_features['bullish_sweep_count'] = len(sweeps[sweeps['brk_dir'] == 1])
        summary_features['bearish_sweep_count'] = len(sweeps[sweeps['brk_dir'] == -1])
        
        if len(sweeps) > 0 and 'brk_stretch_pct_atr' in result.columns:
            summary_features['max_sweep_strength'] = sweeps['brk_stretch_pct_atr'].max()
            summary_features['avg_sweep_strength'] = sweeps['brk_stretch_pct_atr'].mean()
    
    # MSS summary
    if 'mss_flip' in result.columns:
        mss_events = result[result['mss_flip'] != 0]
        summary_features['mss_count'] = len(mss_events)
        summary_features['last_mss_direction'] = result['mss_flip'].iloc[-1] if n > 0 else 0
    
    # iFVG summary
    if 'ifvg_confirmed' in result.columns:
        confirmed_ifvgs = result[result['ifvg_confirmed'] == True]
        summary_features['ifvg_count'] = len(confirmed_ifvgs)
        
        if len(confirmed_ifvgs) > 0 and 'ifvg_size_pct' in result.columns:
            summary_features['max_ifvg_size'] = confirmed_ifvgs['ifvg_size_pct'].max()
            summary_features['recent_ifvg_mid'] = confirmed_ifvgs['ifvg_mid'].iloc[-1]
        else:
            summary_features['max_ifvg_size'] = 0.0
            summary_features['recent_ifvg_mid'] = 0.0
    else:
        summary_features['ifvg_count'] = 0
        summary_features['max_ifvg_size'] = 0.0
        summary_features['recent_ifvg_mid'] = 0.0
    
    # Add summary features as new columns (all rows get same values for simplicity)
    for feature, value in summary_features.items():
        result[feature] = value
    
    return result


def get_mech_feature_names() -> Dict[str, list]:
    """
    Get the names of all mechanical model features by category.
    
    Returns:
        Dictionary mapping feature categories to feature name lists
    """
    return {
        'swing': ['swing_high', 'swing_low'],
        'sweep': ['brk_dir', 'brk_stretch_pct_atr', 'brk_minutes'],
        'mss': ['mss_flip'],
        'smt': ['smt_divergence'],
        'ifvg': ['ifvg_mid', 'ifvg_size_pct', 'ifvg_top', 'ifvg_bottom', 'ifvg_confirmed'],
        'summary': [
            'swing_high_count', 'swing_low_count',
            'sweep_count', 'bullish_sweep_count', 'bearish_sweep_count',
            'max_sweep_strength', 'avg_sweep_strength',
            'mss_count', 'last_mss_direction',
            'ifvg_count', 'max_ifvg_size', 'recent_ifvg_mid'
        ]
    }


def validate_mech_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate mechanical model features in a DataFrame.
    
    Args:
        df: DataFrame with mechanical model features
        
    Returns:
        Dictionary with validation results
    """
    feature_categories = get_mech_feature_names()
    all_features = []
    for features in feature_categories.values():
        all_features.extend(features)
    
    results = {
        'total_expected_features': len(all_features),
        'features_present': 0,
        'missing_features': [],
        'feature_coverage': 0.0,
        'categories': {}
    }
    
    for category, features in feature_categories.items():
        present = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        
        results['categories'][category] = {
            'expected': len(features),
            'present': len(present),
            'missing': missing,
            'coverage': len(present) / len(features) if features else 1.0
        }
        
        results['features_present'] += len(present)
        results['missing_features'].extend(missing)
    
    results['feature_coverage'] = results['features_present'] / results['total_expected_features']
    
    return results 