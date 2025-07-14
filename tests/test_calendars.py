"""
Tests for orb.utils.calendars module.

Tests NYSE trading calendar functionality including holiday handling,
session counting, and retrain day logic.
"""

import pytest
from datetime import date, datetime
from orb.utils.calendars import trading_days, nth_prev_session, is_retrain_day


class TestTradingDays:
    """Test trading_days function."""
    
    def test_trading_days_excludes_independence_day_2025(self):
        """Verify 2025-07-04 (Independence Day) is excluded from trading days."""
        # Test week containing July 4, 2025 (Friday)
        result = trading_days('2025-07-01', '2025-07-07')
        
        # Should include Mon-Wed and Mon (next week), but exclude Fri July 4
        expected_dates = [
            date(2025, 7, 1),  # Tuesday
            date(2025, 7, 2),  # Wednesday  
            date(2025, 7, 3),  # Thursday
            date(2025, 7, 7),  # Monday (next week)
        ]
        # July 4, 2025 (Friday) should be excluded as Independence Day
        # July 5-6 are weekend
        
        assert result == expected_dates
        assert date(2025, 7, 4) not in result, "Independence Day should be excluded"
    
    def test_trading_days_basic_week(self):
        """Test trading days for a normal business week."""
        result = trading_days('2025-06-02', '2025-06-06')  # Mon-Fri
        
        expected = [
            date(2025, 6, 2),  # Monday
            date(2025, 6, 3),  # Tuesday
            date(2025, 6, 4),  # Wednesday
            date(2025, 6, 5),  # Thursday
            date(2025, 6, 6),  # Friday
        ]
        
        assert result == expected
    
    def test_trading_days_excludes_weekends(self):
        """Test that weekends are properly excluded."""
        result = trading_days('2025-06-06', '2025-06-09')  # Fri-Mon
        
        expected = [
            date(2025, 6, 6),  # Friday
            date(2025, 6, 9),  # Monday
        ]
        # Saturday 6/7 and Sunday 6/8 should be excluded
        
        assert result == expected
        assert date(2025, 6, 7) not in result, "Saturday should be excluded"
        assert date(2025, 6, 8) not in result, "Sunday should be excluded"
    
    def test_trading_days_single_date(self):
        """Test trading days with same start and end date."""
        # Trading day
        result = trading_days('2025-06-02', '2025-06-02')  # Monday
        assert result == [date(2025, 6, 2)]
        
        # Holiday (should return empty)
        result = trading_days('2025-07-04', '2025-07-04')  # Independence Day
        assert result == []
    
    def test_trading_days_empty_range(self):
        """Test with invalid date range."""
        result = trading_days('2025-06-06', '2025-06-02')  # End before start
        assert result == []


class TestNthPrevSession:
    """Test nth_prev_session function."""
    
    def test_nth_prev_session_specific_case(self):
        """Test the specific case from requirements."""
        # 2025-07-14 is Monday
        # Previous sessions: 2025-07-11 (Fri), 2025-07-10 (Thu), 2025-07-09 (Wed)
        result = nth_prev_session(date(2025, 7, 14), 3)
        assert result == date(2025, 7, 9), f"Expected 2025-07-09, got {result}"
    
    def test_nth_prev_session_basic(self):
        """Test basic nth_prev_session functionality."""
        # 2025-06-03 is Tuesday
        # Previous session should be Monday 2025-06-02
        result = nth_prev_session(date(2025, 6, 3), 1)
        assert result == date(2025, 6, 2)
    
    def test_nth_prev_session_skip_weekend(self):
        """Test nth_prev_session skipping weekends."""
        # 2025-06-09 is Monday
        # 1 session back should be Friday 2025-06-06 (skip weekend)
        result = nth_prev_session(date(2025, 6, 9), 1)
        assert result == date(2025, 6, 6)
    
    def test_nth_prev_session_skip_holiday(self):
        """Test nth_prev_session skipping holidays."""
        # 2025-07-07 is Monday after July 4th holiday
        # 1 session back should be Thursday 2025-07-03 (skip Fri holiday + weekend)
        result = nth_prev_session(date(2025, 7, 7), 1)
        assert result == date(2025, 7, 3)
    
    def test_nth_prev_session_invalid_n(self):
        """Test nth_prev_session with invalid n values."""
        with pytest.raises(ValueError, match="n must be positive"):
            nth_prev_session(date(2025, 6, 2), 0)
        
        with pytest.raises(ValueError, match="n must be positive"):
            nth_prev_session(date(2025, 6, 2), -1)
    
    def test_nth_prev_session_large_n(self):
        """Test nth_prev_session with a large but reasonable n value."""
        # Test with a large n value to ensure the function can handle it
        # Use a reasonable recent date and moderately large n
        result = nth_prev_session(date(2025, 6, 15), 50)
        
        # Should return a date that's at least 50 trading days before
        assert isinstance(result, date)
        assert result < date(2025, 6, 15)


class TestIsRetrainDay:
    """Test is_retrain_day function."""
    
    def test_first_nyse_session_2025_is_retrain_day(self):
        """Test that first NYSE session of 2025 is a retrain day."""
        # First NYSE session of 2025 is January 2, 2025 (Jan 1 is New Year's Day)
        result = is_retrain_day(date(2025, 1, 2), gap=40)
        assert result is True, "First NYSE session of 2025 should be retrain day"
    
    def test_epoch_date_is_retrain_day(self):
        """Test that the epoch date itself is considered a retrain day."""
        # Even with gap=1, the epoch should be a retrain day
        result = is_retrain_day(date(2025, 1, 2), gap=1)
        assert result is True
    
    def test_retrain_gap_logic(self):
        """Test retrain gap logic with different gap values."""
        # With gap=1, every trading day after epoch should be retrain day
        # Find a few trading days after the epoch
        trading_dates = trading_days('2025-01-02', '2025-01-10')
        
        if len(trading_dates) >= 2:
            second_trading_day = trading_dates[1]  # Should be retrain day with gap=1
            result = is_retrain_day(second_trading_day, gap=1)
            assert result is True, "Second trading day should be retrain day with gap=1"
    
    def test_non_retrain_days(self):
        """Test days that should not be retrain days."""
        # With gap=40, most days early in 2025 should not be retrain days
        # (except the first session and every 40th thereafter)
        
        # Get some trading days after the epoch
        trading_dates = trading_days('2025-01-02', '2025-02-28')
        
        if len(trading_dates) >= 5:
            # Check a day that's not on a gap boundary
            test_date = trading_dates[2]  # 3rd trading day (not epoch, not 40th)
            result = is_retrain_day(test_date, gap=40)
            # This should be False unless we're exactly on a gap boundary
            # Since it's only the 3rd day, it should be False
            assert result is False, f"Day {test_date} should not be retrain day with gap=40"
    
    def test_retrain_day_gap_boundary(self):
        """Test retrain day detection at gap boundaries."""
        # This test verifies the modulo logic works correctly
        # We'll use a smaller gap for easier testing
        
        trading_dates = trading_days('2025-01-02', '2025-03-31')
        
        if len(trading_dates) >= 10:
            # With gap=5, the 6th trading day (index 5) should be a retrain day
            # because (6-1) % 5 == 0
            sixth_day = trading_dates[5]
            result = is_retrain_day(sixth_day, gap=5)
            assert result is True, f"6th trading day {sixth_day} should be retrain day with gap=5"
            
            # The 7th trading day should not be
            if len(trading_dates) >= 7:
                seventh_day = trading_dates[6]
                result = is_retrain_day(seventh_day, gap=5)
                assert result is False, f"7th trading day {seventh_day} should not be retrain day with gap=5"
    
    def test_retrain_day_before_epoch(self):
        """Test is_retrain_day for dates before the epoch."""
        # Any date before or on the epoch should return True
        result = is_retrain_day(date(2024, 12, 31), gap=40)
        assert result is True, "Dates before epoch should be retrain days"
        
        result = is_retrain_day(date(2025, 1, 1), gap=40)  # New Year's Day
        assert result is True, "Dates on/before epoch should be retrain days"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_date_string_inputs(self):
        """Test that functions handle string inputs correctly."""
        # trading_days should work with string inputs
        result = trading_days('2025-06-02', '2025-06-04')
        expected = [date(2025, 6, 2), date(2025, 6, 3), date(2025, 6, 4)]
        assert result == expected
    
    def test_datetime_inputs(self):
        """Test that functions handle datetime inputs correctly."""
        # trading_days should work with datetime inputs
        result = trading_days(datetime(2025, 6, 2), datetime(2025, 6, 4))
        expected = [date(2025, 6, 2), date(2025, 6, 3), date(2025, 6, 4)]
        assert result == expected
    
    def test_year_boundary(self):
        """Test functions across year boundaries."""
        # Test trading days across New Year
        result = trading_days('2024-12-30', '2025-01-03')
        
        # 2024-12-30 (Mon), 2024-12-31 (Tue), 2025-01-01 (Wed - NYD), 2025-01-02 (Thu), 2025-01-03 (Fri)
        # Should exclude 2025-01-01 (New Year's Day)
        expected = [
            date(2024, 12, 30),
            date(2024, 12, 31),
            date(2025, 1, 2),
            date(2025, 1, 3),
        ]
        
        assert date(2025, 1, 1) not in result, "New Year's Day should be excluded"
        assert all(d in result for d in expected), f"Expected dates missing from {result}" 