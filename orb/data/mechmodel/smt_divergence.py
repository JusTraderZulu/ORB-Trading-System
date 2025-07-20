"""
Smart Money Tool (SMT) divergence detection for multi-instrument analysis.

Detects divergences between correlated pairs (BTC/ETH, EUR/GBP) where one
instrument sweeps liquidity while the other doesn't, indicating potential
institutional manipulation or divergent strength.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

from .sweep_detector import detect_sweep


# SMT correlation pairs mapping
SMT_PAIRS = {
    'BTC-USD': 'ETH-USD',
    'BTCUSD': 'ETHUSD', 
    'BTC': 'ETH',
    'ETH-USD': 'BTC-USD',
    'ETHUSD': 'BTCUSD',
    'ETH': 'BTC',
    'EUR-USD': 'GBP-USD',
    'EURUSD': 'GBPUSD',
    'EUR': 'GBP', 
    'GBP-USD': 'EUR-USD',
    'GBPUSD': 'EURUSD',
    'GBP': 'EUR'
}


def smt_flag(df_primary: pd.DataFrame, df_secondary: Optional[pd.DataFrame] = None,
             primary_symbol: str = '', atr_window: int = 14) -> pd.Series:
    """
    Detect Smart Money Tool (SMT) divergence between correlated instruments.
    
    SMT divergence occurs when one instrument in a correlated pair sweeps liquidity
    (breaks swing levels) while the other doesn't, suggesting manipulation or
    divergent institutional interest.
    
    Args:
        df_primary: Primary instrument price data
        df_secondary: Secondary instrument price data (optional)
        primary_symbol: Symbol name for auto-pairing detection
        atr_window: Window for ATR calculation in sweep detection
        
    Returns:
        Series with SMT flags (1 = divergence detected, 0 = no divergence)
        
    Example:
        >>> btc_data = pd.DataFrame({...})  # BTC price data
        >>> eth_data = pd.DataFrame({...})  # ETH price data  
        >>> smt_flags = smt_flag(btc_data, eth_data, 'BTC-USD')
    """
    # Initialize result series
    result = pd.Series([0] * len(df_primary), index=df_primary.index, name='smt_flag')
    
    if df_primary.empty:
        return result
    
    # Determine secondary instrument
    if df_secondary is None:
        # Try to auto-detect pair from symbol name
        secondary_symbol = _get_smt_pair(primary_symbol)
        if secondary_symbol is None:
            # No SMT pair defined for this symbol
            return result
        # In real implementation, would load secondary data here
        # For now, return no divergence
        return result
    
    if df_secondary.empty or len(df_secondary) != len(df_primary):
        return result
    
    # Detect sweeps in both instruments
    primary_sweeps = detect_sweep(df_primary, atr_window=atr_window)
    secondary_sweeps = detect_sweep(df_secondary, atr_window=atr_window)
    
    # Align the data by index/timestamp
    if 'timestamp' in df_primary.columns and 'timestamp' in df_secondary.columns:
        primary_sweeps = primary_sweeps.set_index('timestamp')
        secondary_sweeps = secondary_sweeps.set_index('timestamp')
        
        # Align indices
        common_index = primary_sweeps.index.intersection(secondary_sweeps.index)
        if len(common_index) < len(primary_sweeps) * 0.8:
            # Not enough common data points
            return result
            
        primary_sweeps = primary_sweeps.reindex(common_index)
        secondary_sweeps = secondary_sweeps.reindex(common_index)
    
    # Detect SMT divergence patterns
    divergence_flags = _detect_smt_divergence(primary_sweeps, secondary_sweeps)
    
    # Map back to original index
    if len(divergence_flags) == len(result):
        result = divergence_flags
    else:
        # Handle index alignment issues
        for i, flag in enumerate(divergence_flags):
            if i < len(result):
                result.iloc[i] = flag
    
    return result


def _get_smt_pair(symbol: str) -> Optional[str]:
    """
    Get the SMT correlation pair for a given symbol.
    
    Args:
        symbol: Primary symbol name
        
    Returns:
        Secondary symbol name or None if no pair defined
    """
    # Normalize symbol name
    symbol_clean = symbol.upper().replace('-', '').replace('_', '')
    
    # Check direct mappings
    for key, value in SMT_PAIRS.items():
        key_clean = key.upper().replace('-', '').replace('_', '')
        if symbol_clean == key_clean or symbol_clean.startswith(key_clean):
            return value
    
    return None


def _detect_smt_divergence(df_primary: pd.DataFrame, df_secondary: pd.DataFrame) -> pd.Series:
    """
    Detect SMT divergence patterns between two instruments.
    
    SMT divergence criteria:
    1. One instrument shows a liquidity sweep (brk_dir != 0)
    2. The other instrument doesn't sweep at the same time
    3. The divergence persists for at least a few bars
    
    Args:
        df_primary: Primary instrument with sweep detection
        df_secondary: Secondary instrument with sweep detection
        
    Returns:
        Series with divergence flags
    """
    n = len(df_primary)
    divergence_flags = pd.Series([0] * n, index=df_primary.index)
    
    if 'brk_dir' not in df_primary.columns or 'brk_dir' not in df_secondary.columns:
        return divergence_flags
    
    # Look for divergence patterns
    lookback_window = 5  # Check divergence over small time window
    
    for i in range(lookback_window, n):
        # Check recent sweep activity in both instruments
        primary_recent = df_primary.iloc[i-lookback_window:i+1]
        secondary_recent = df_secondary.iloc[i-lookback_window:i+1]
        
        # Check for primary sweep without secondary sweep
        primary_sweep = (primary_recent['brk_dir'] != 0).any()
        secondary_sweep = (secondary_recent['brk_dir'] != 0).any()
        
        # Divergence detected if one sweeps but not the other
        if primary_sweep and not secondary_sweep:
            divergence_flags.iloc[i] = 1
        elif secondary_sweep and not primary_sweep:
            divergence_flags.iloc[i] = 1
        
        # Additional criteria: Check for opposite direction sweeps
        if primary_sweep and secondary_sweep:
            primary_direction = primary_recent[primary_recent['brk_dir'] != 0]['brk_dir'].iloc[-1]
            secondary_direction = secondary_recent[secondary_recent['brk_dir'] != 0]['brk_dir'].iloc[-1]
            
            # Opposite directions indicate divergence
            if primary_direction * secondary_direction < 0:
                divergence_flags.iloc[i] = 1
    
    return divergence_flags


def get_smt_summary(primary_df: pd.DataFrame, secondary_df: Optional[pd.DataFrame] = None,
                   primary_symbol: str = '') -> Dict[str, any]:
    """
    Get summary statistics of SMT divergence analysis.
    
    Args:
        primary_df: Primary instrument data
        secondary_df: Secondary instrument data
        primary_symbol: Primary symbol name
        
    Returns:
        Dictionary with SMT analysis results
    """
    summary = {
        'smt_pair_detected': False,
        'secondary_symbol': None,
        'total_divergences': 0,
        'divergence_frequency': 0,
        'correlation_available': False
    }
    
    # Check if SMT pair is available
    secondary_symbol = _get_smt_pair(primary_symbol)
    summary['secondary_symbol'] = secondary_symbol
    summary['smt_pair_detected'] = secondary_symbol is not None
    
    if secondary_df is None or secondary_df.empty:
        return summary
    
    summary['correlation_available'] = True
    
    # Run SMT analysis
    smt_flags = smt_flag(primary_df, secondary_df, primary_symbol)
    
    # Calculate statistics
    divergences = smt_flags[smt_flags == 1]
    summary['total_divergences'] = len(divergences)
    
    if len(divergences) > 0:
        # Calculate average bars between divergences
        divergence_indices = divergences.index.tolist()
        if len(divergence_indices) > 1:
            intervals = [divergence_indices[i] - divergence_indices[i-1] 
                        for i in range(1, len(divergence_indices))]
            summary['divergence_frequency'] = np.mean(intervals)
        else:
            summary['divergence_frequency'] = len(primary_df)
    
    return summary


def validate_smt_setup(primary_symbol: str, secondary_data_available: bool = False) -> Dict[str, any]:
    """
    Validate if SMT analysis can be performed for a given symbol.
    
    Args:
        primary_symbol: Symbol to check for SMT pair
        secondary_data_available: Whether secondary instrument data is available
        
    Returns:
        Dictionary with validation results
    """
    secondary_symbol = _get_smt_pair(primary_symbol)
    
    return {
        'can_perform_smt': secondary_symbol is not None and secondary_data_available,
        'primary_symbol': primary_symbol,
        'secondary_symbol': secondary_symbol,
        'supported_pairs': list(set(SMT_PAIRS.values())),
        'pair_type': _get_pair_type(primary_symbol),
        'reason': _get_validation_reason(primary_symbol, secondary_symbol, secondary_data_available)
    }


def _get_pair_type(symbol: str) -> str:
    """Determine if symbol is crypto or forex."""
    crypto_indicators = ['BTC', 'ETH', 'CRYPTO']
    forex_indicators = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
    
    symbol_upper = symbol.upper()
    
    for indicator in crypto_indicators:
        if indicator in symbol_upper:
            return 'crypto'
    
    for indicator in forex_indicators:
        if indicator in symbol_upper:
            return 'forex'
    
    return 'unknown'


def _get_validation_reason(primary_symbol: str, secondary_symbol: Optional[str], 
                          secondary_available: bool) -> str:
    """Get reason why SMT analysis can or cannot be performed."""
    if secondary_symbol is None:
        return f"No SMT pair defined for {primary_symbol}. Supported: {list(SMT_PAIRS.keys())}"
    
    if not secondary_available:
        return f"Secondary data for {secondary_symbol} not available"
    
    return f"SMT analysis ready for {primary_symbol} vs {secondary_symbol}" 