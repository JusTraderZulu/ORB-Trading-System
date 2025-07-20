"""
Mechanical model feature detectors for advanced market structure analysis.

This sub-package implements sophisticated market microstructure detection:
- Swing highs/lows identification
- Liquidity sweep detection with ATR validation
- Market structure shift (MSS) analysis
- Smart Money Tool (SMT) divergence detection
- Institutional Fair Value Gap (iFVG) identification

Designed for crypto and FX markets with 24/7 trading.
"""

from .swing_utils import swings
from .sweep_detector import detect_sweep
from .mss_detector import detect_mss
from .smt_divergence import smt_flag
from .ifvg_finder import find_ifvg
from .pipeline import run_mech_pipeline, get_mech_feature_names, validate_mech_features

__all__ = [
    'swings',
    'detect_sweep', 
    'detect_mss',
    'smt_flag',
    'find_ifvg',
    'run_mech_pipeline',
    'get_mech_feature_names',
    'validate_mech_features'
] 