"""
Polygon API data loader for ORB system.

Downloads 1-minute bar data from Polygon API and consolidates into Parquet format.
Handles timezone conversion from UTC to America/New_York.
"""

import json
import gzip
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pytz
from tqdm import tqdm

from ..utils.logging import LoggingMixin
from ..utils.calendars import trading_days, get_market_timezone


class PolygonLoader(LoggingMixin):
    """
    Loader for Polygon API data with rate limiting and error handling.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.polygon.io",
        max_requests_per_minute: int = 5,
        raw_data_dir: str = "data/raw",
        minute_data_dir: str = "data/minute"
    ):
        """
        Initialize Polygon loader.
        
        Args:
            api_key: Polygon API key
            base_url: Base URL for Polygon API
            max_requests_per_minute: Rate limit for API requests
            raw_data_dir: Directory for raw JSON data
            minute_data_dir: Directory for processed Parquet data
        """
        self.api_key = api_key
        self.base_url = base_url
        self.max_requests_per_minute = max_requests_per_minute
        self.raw_data_dir = Path(raw_data_dir)
        self.minute_data_dir = Path(minute_data_dir)
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.minute_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Rate limiting
        self.min_request_interval = 60.0 / max_requests_per_minute
        self.last_request_time = 0
        
        self.log_info(f"Initialized Polygon loader with API key: {api_key[:8]}...")
        
    def _rate_limit(self) -> None:
        """Apply rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        
    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request with rate limiting and error handling.
        
        Args:
            url: API endpoint URL
            params: Request parameters
            
        Returns:
            JSON response data
        """
        self._rate_limit()
        
        params['apikey'] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.log_error(f"API request failed: {e}")
            raise
    
    def download_month(
        self,
        symbol: str,
        year: int,
        month: int,
        multiplier: int = 1,
        timespan: str = "minute"
    ) -> None:
        """
        Download monthly data for a symbol from Polygon API.
        
        Calls Polygon /v2/aggs/ticker/{sym}/range/1/minute/{start}/{end}
        Writes compressed JSON to data/raw/{sym}/{yyyymm}.json.gz
        Handles pagination (50000-row limit) and API-key headers
        Respects rate-limit: sleep if status 429
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            year: Year (e.g., 2024)
            month: Month (1-12)
            multiplier: Timespan multiplier (default: 1)
            timespan: Timespan unit (default: 'minute')
            
        Returns:
            None
            
        Raises:
            Exception: If API call fails or data cannot be saved
        """
        # Create date range for the month
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1)
        else:
            end_date = date(year, month + 1, 1)
        
        # Format dates for API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # API endpoint
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000  # Maximum results per request
        }
        
        self.log_info(f"Downloading {symbol} data for {year}-{month:02d}")
        
        try:
            all_results = []
            next_url = None
            
            while True:
                if next_url:
                    response = self._make_request(next_url, {})
                else:
                    response = self._make_request(url, params)
                
                if response.get('status') != 'OK':
                    self.log_error(f"API returned status: {response.get('status')}")
                    raise Exception(f"Polygon API error: {response.get('status')}")
                
                # Handle rate limiting
                if response.get('status') == 'ERROR' and 'too many requests' in str(response.get('message', '')).lower():
                    self.log_warning("Rate limit hit, sleeping for 60 seconds...")
                    time.sleep(60)
                    continue
                
                results = response.get('results', [])
                all_results.extend(results)
                
                # Check for pagination
                next_url = response.get('next_url')
                if not next_url:
                    break
                    
                self.log_debug(f"Got {len(results)} bars, continuing with pagination...")
            
            # Save raw data
            output_dir = self.raw_data_dir / symbol
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{year}{month:02d}.json.gz"
            filepath = output_dir / filename
            
            data = {
                'symbol': symbol,
                'results': all_results,
                'metadata': {
                    'year': year,
                    'month': month,
                    'download_time': datetime.now().isoformat(),
                    'count': len(all_results)
                }
            }
            
            with gzip.open(filepath, 'wt') as f:
                json.dump(data, f)
            
            self.log_info(f"Saved {len(all_results)} bars to {filepath}")
            
        except Exception as e:
            self.log_error(f"Error downloading {symbol} for {year}-{month:02d}: {e}")
            raise
    
    def build_parquet(self, symbol: str) -> Path:
        """
        Build consolidated Parquet file from raw JSON data.
        
        Reads every raw JSON file for symbol
        Converts "t" (ms since epoch) → America/New_York timestamp index
        Drops rows outside 09:25–16:00
        Saves consolidated Parquet to data/minute/{sym}.parquet
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Path to the created Parquet file
            
        Raises:
            Exception: If no raw data found or processing fails
        """
        raw_dir = self.raw_data_dir / symbol
        
        if not raw_dir.exists():
            self.log_error(f"No raw data found for {symbol}")
            raise Exception(f"No raw data found for {symbol}")
        
        self.log_info(f"Building Parquet file for {symbol}")
        
        try:
            all_data = []
            
            # Process all JSON files for this symbol
            json_files = list(raw_dir.glob("*.json.gz"))
            
            if not json_files:
                self.log_error(f"No JSON files found for {symbol}")
                raise Exception(f"No JSON files found for {symbol}")
                
            for json_file in tqdm(json_files, desc=f"Processing {symbol}"):
                try:
                    with gzip.open(json_file, 'rt') as f:
                        data = json.load(f)
                    
                    results = data.get('results', [])
                    if not results:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(results)
                    
                    # Rename columns to standard format
                    column_mapping = {
                        't': 'timestamp',
                        'o': 'open',
                        'h': 'high',
                        'l': 'low',
                        'c': 'close',
                        'v': 'volume',
                        'vw': 'vwap',
                        'n': 'num_trades'
                    }
                    df = df.rename(columns=column_mapping)
                    
                    # Convert timestamp from milliseconds to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Convert from UTC to Eastern time
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(get_market_timezone())
                    
                    # Add symbol column
                    df['symbol'] = symbol
                    
                    # Select and order columns
                    columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'num_trades']
                    df = df[columns]
                    
                    all_data.append(df)
                    
                except Exception as e:
                    self.log_error(f"Error processing {json_file}: {e}")
                    continue
            
            if not all_data:
                self.log_error(f"No valid data found for {symbol}")
                raise Exception(f"No valid data found for {symbol}")
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Sort by timestamp
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['timestamp', 'symbol'])
            
            # Filter to trading hours only (9:25 AM - 4:00 PM ET)
            combined_df = self._filter_trading_hours(combined_df, market_start="09:25")
            
            # Save to Parquet
            parquet_path = self.minute_data_dir / f"{symbol}.parquet"
            combined_df.to_parquet(parquet_path, index=False)
            
            self.log_info(f"Saved {len(combined_df)} records to {parquet_path}")
            return parquet_path
            
        except Exception as e:
            self.log_error(f"Error building Parquet for {symbol}: {e}")
            raise
    
    def _filter_trading_hours(
        self,
        df: pd.DataFrame,
        market_start: str = "09:25",
        market_end: str = "16:00"
    ) -> pd.DataFrame:
        """
        Filter DataFrame to trading hours only.
        
        Args:
            df: DataFrame with timestamp column
            market_start: Market open time (HH:MM)
            market_end: Market close time (HH:MM)
            
        Returns:
            Filtered DataFrame
        """
        # Convert market times to time objects
        start_time = pd.to_datetime(market_start, format='%H:%M').time()
        end_time = pd.to_datetime(market_end, format='%H:%M').time()
        
        # Filter by time
        mask = (df['timestamp'].dt.time >= start_time) & (df['timestamp'].dt.time <= end_time)
        
        # Filter by trading days
        trading_dates = trading_days(df['timestamp'].min().date(), df['timestamp'].max().date())
        # Convert to date objects if it's a DatetimeIndex
        if hasattr(trading_dates, 'date'):
            trading_dates_set = set(trading_dates.date)
        else:
            # If it's already a list of dates
            trading_dates_set = set(trading_dates)
        
        date_mask = df['timestamp'].dt.date.isin(trading_dates_set)
        
        return df[mask & date_mask].copy()
    
    def get_available_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get information about available data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with data availability info or None if no data
        """
        raw_dir = self.raw_data_dir / symbol
        
        if not raw_dir.exists():
            return None
        
        info_list = []
        
        for json_file in raw_dir.glob("*.json.gz"):
            try:
                with gzip.open(json_file, 'rt') as f:
                    data = json.load(f)
                
                metadata = data.get('metadata', {})
                results = data.get('results', [])
                
                if results:
                    timestamps = [r['t'] for r in results]
                    start_ts = min(timestamps)
                    end_ts = max(timestamps)
                    
                    info_list.append({
                        'file': json_file.name,
                        'year': metadata.get('year'),
                        'month': metadata.get('month'),
                        'count': len(results),
                        'start_date': pd.to_datetime(start_ts, unit='ms').strftime('%Y-%m-%d'),
                        'end_date': pd.to_datetime(end_ts, unit='ms').strftime('%Y-%m-%d'),
                        'download_time': metadata.get('download_time')
                    })
            except Exception as e:
                self.log_error(f"Error reading {json_file}: {e}")
                continue
        
        return pd.DataFrame(info_list) if info_list else None


def download_month(sym: str, year: int, month: int) -> None:
    """
    Download monthly data for a symbol from Polygon API.
    
    Args:
        sym: Stock symbol
        year: Year  
        month: Month (1-12)
        
    Returns:
        None
        
    Raises:
        Exception: If API key not found or download fails
    """
    import os
    
    # Try to load from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed, continue with system env vars
    
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        raise Exception("POLYGON_API_KEY environment variable not set. " +
                       "Set it in your shell or add to .env file in project root.")
    
    loader = PolygonLoader(api_key=api_key, raw_data_dir="data/raw")
    loader.download_month(sym, year, month)


def build_parquet(sym: str) -> Path:
    """
    Build consolidated Parquet file from raw JSON data.
    
    Args:
        sym: Stock symbol
        
    Returns:
        Path to the created Parquet file
        
    Raises:
        Exception: If processing fails
    """
    # Create a dummy loader instance (API key not needed for parquet building)
    loader = PolygonLoader(
        api_key="dummy",
        raw_data_dir="data/raw", 
        minute_data_dir="data/minute"
    )
    return loader.build_parquet(sym) 