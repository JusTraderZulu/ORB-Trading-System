"""
NYSE trading calendar utilities for ORB system.

Provides functions for:
- Determining trading days and market holidays
- Getting previous trading sessions  
- Checking if a retrain is due based on session count
"""

import datetime
from datetime import date
from typing import List, Union
import pandas as pd
import pandas_market_calendars as mcal
import pytz

__all__ = [
    "trading_days",
    "nth_prev_session", 
    "is_retrain_day",
    "get_market_timezone",
]


def get_market_timezone() -> pytz.BaseTzInfo:
    """Get the market timezone (America/New_York)."""
    return pytz.timezone('America/New_York')


def trading_days(
    start: Union[str, datetime.datetime, date],
    end: Union[str, datetime.datetime, date]
) -> List[date]:
    """
    Get trading days between start and end dates (excluding NYSE holidays).
    
    Uses pandas-market-calendars for accurate NYSE holiday handling.
    All dates returned are in America/New_York timezone.
    
    Args:
        start: Start date (inclusive)
        end: End date (inclusive)
        
    Returns:
        List of trading dates
        
    Examples:
        >>> trading_days('2025-07-01', '2025-07-07')
        [date(2025, 7, 1), date(2025, 7, 2), date(2025, 7, 3), date(2025, 7, 7)]
        # Note: 2025-07-04 (Independence Day) is excluded
    """
    # Get NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Convert inputs to pandas timestamps
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    # Handle invalid ranges gracefully
    if start_dt > end_dt:
        return []
    
    # Get valid trading days
    schedule = nyse.schedule(start_date=start_dt, end_date=end_dt)
    
    # Extract dates and convert to date objects
    trading_dates = [ts.date() for ts in schedule.index]
    
    return trading_dates


def nth_prev_session(target_date: date, n: int) -> date:
    """
    Get the nth previous trading session from a reference date.
    
    Args:
        target_date: Reference date
        n: Number of sessions back (must be positive)
        
    Returns:
        Date of the nth previous trading session
        
    Raises:
        ValueError: If n <= 0 or insufficient trading days found
        
    Examples:
        >>> nth_prev_session(date(2025, 7, 14), 3)
        date(2025, 7, 9)
        # Counts back: 2025-07-11 (1st), 2025-07-10 (2nd), 2025-07-09 (3rd)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    
    # Get NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Calculate sufficient lookback period (account for weekends + holidays)
    # Use 2.5x multiplier to handle holiday clusters
    lookback_days = max(int(n * 2.5), 30)
    start_date = target_date - datetime.timedelta(days=lookback_days)
    
    # Get trading days up to (but not including) reference date
    end_date = target_date - datetime.timedelta(days=1)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    if len(schedule) < n:
        # Extend lookback if needed
        extended_start = target_date - datetime.timedelta(days=lookback_days * 3)
        schedule = nyse.schedule(start_date=extended_start, end_date=end_date)
    
    if len(schedule) < n:
        raise ValueError(f"Cannot find {n} trading sessions before {target_date}")
    
    # Return the nth previous session (counting from most recent)
    return schedule.index[-n].date()


def is_retrain_day(target_date: date, gap: int = 40) -> bool:
    """
    Check if it's time to retrain the model based on trading sessions elapsed.
    
    Determines if the given date represents a retrain day by checking if it's
    been at least 'gap' trading sessions since the model training epoch.
    For simplicity, this implementation treats the first NYSE trading session
    of 2025 as the epoch (baseline).
    
    Args:
        target_date: Date to check for retraining
        gap: Number of trading sessions between retrains (default: 40)
        
    Returns:
        True if retrain is due, False otherwise
        
    Examples:
        >>> # First trading session of 2025 should trigger retrain
        >>> is_retrain_day(date(2025, 1, 2), gap=40)  # 2025-01-02 is first NYSE session
        True
        >>> # 39 sessions later should not trigger
        >>> is_retrain_day(date(2025, 2, 20), gap=40)  # Assuming ~39 sessions elapsed
        False
        >>> # 40+ sessions later should trigger
        >>> is_retrain_day(date(2025, 2, 21), gap=40)  # Assuming 40+ sessions elapsed  
        True
    """
    # Define training epoch as first NYSE trading session of 2025
    epoch_start = date(2025, 1, 1)
    
    # Get NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Find first trading session of 2025
    epoch_schedule = nyse.schedule(start_date=epoch_start, end_date=date(2025, 1, 10))
    if len(epoch_schedule) == 0:
        # Fallback to Jan 2, 2025 if schedule lookup fails
        training_epoch = date(2025, 1, 2)
    else:
        training_epoch = epoch_schedule.index[0].date()
    
    # If checking the epoch date itself, it's a retrain day
    if target_date <= training_epoch:
        return True
    
    # Count trading sessions between epoch and given date (inclusive of given date)
    schedule = nyse.schedule(start_date=training_epoch, end_date=target_date)
    sessions_elapsed = len(schedule) - 1  # Exclude the epoch day itself
    
    # Check if we've hit the gap threshold
    return sessions_elapsed > 0 and (sessions_elapsed % gap) == 0


 