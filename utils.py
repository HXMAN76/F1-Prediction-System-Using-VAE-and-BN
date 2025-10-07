import pandas as pd
import numpy as np
import fastf1
import warnings
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime

# Suppress FastF1 warnings for cleaner output
warnings.filterwarnings('ignore')
fastf1.Cache.enable_cache('cache')

def safe_float_convert(value, default=0.0):
    """Safely convert value to float, return default if conversion fails."""
    try:
        if pd.isna(value) or value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_time_convert(time_obj, default=0.0):
    """Safely convert FastF1 time object to seconds."""
    try:
        if pd.isna(time_obj) or time_obj is None:
            return default
        if hasattr(time_obj, 'total_seconds'):
            return time_obj.total_seconds()
        return float(time_obj)
    except (ValueError, TypeError, AttributeError):
        return default

def calculate_consistency(lap_times: List[float]) -> float:
    """Calculate lap time consistency (standard deviation)."""
    try:
        valid_times = [t for t in lap_times if not pd.isna(t) and t > 0]
        if len(valid_times) < 2:
            return 0.0
        return np.std(valid_times)
    except:
        return 0.0

def extract_tyre_compounds(tyre_data) -> str:
    """Extract unique tyre compounds used during race."""
    try:
        if pd.isna(tyre_data).all():
            return "UNKNOWN"
        compounds = tyre_data.dropna().unique()
        return ",".join(sorted(compounds))
    except:
        return "UNKNOWN"

def calculate_positions_change(start_pos: int, finish_pos: int) -> int:
    """Calculate positions gained/lost (positive = gained, negative = lost)."""
    try:
        if pd.isna(start_pos) or pd.isna(finish_pos):
            return 0
        return int(start_pos) - int(finish_pos)
    except:
        return 0

def process_telemetry_data(telemetry_df) -> Dict:
    """Process telemetry data and return aggregated metrics."""
    try:
        if telemetry_df is None or telemetry_df.empty:
            return {
                'avg_speed': 0.0, 'max_speed': 0.0, 'min_speed': 0.0,
                'throttle_avg': 0.0, 'throttle_100_time': 0.0,
                'brake_avg': 0.0, 'brake_applications': 0,
                'avg_gear': 0.0, 'gear_changes': 0,
                'avg_rpm': 0.0, 'max_rpm': 0.0,
                'drs_usage': 0.0
            }
        
        # Speed metrics
        speed_data = telemetry_df['Speed'].dropna()
        avg_speed = speed_data.mean() if not speed_data.empty else 0.0
        max_speed = speed_data.max() if not speed_data.empty else 0.0
        min_speed = speed_data.min() if not speed_data.empty else 0.0
        
        # Throttle metrics
        throttle_data = telemetry_df['Throttle'].dropna()
        throttle_avg = throttle_data.mean() if not throttle_data.empty else 0.0
        throttle_100_time = (throttle_data == 100).sum() / len(throttle_data) * 100 if not throttle_data.empty else 0.0
        
        # Brake metrics
        brake_data = telemetry_df['Brake'].dropna()
        brake_avg = brake_data.mean() if not brake_data.empty else 0.0
        brake_applications = (brake_data > 0).sum() if not brake_data.empty else 0
        
        # Gear metrics
        gear_data = telemetry_df['nGear'].dropna()
        avg_gear = gear_data.mean() if not gear_data.empty else 0.0
        gear_changes = (gear_data.diff().dropna() != 0).sum() if not gear_data.empty else 0
        
        # RPM metrics
        rpm_data = telemetry_df['RPM'].dropna()
        avg_rpm = rpm_data.mean() if not rpm_data.empty else 0.0
        max_rpm = rpm_data.max() if not rpm_data.empty else 0.0
        
        # DRS usage
        drs_data = telemetry_df['DRS'].dropna() if 'DRS' in telemetry_df.columns else pd.Series()
        drs_usage = (drs_data > 0).sum() / len(drs_data) * 100 if not drs_data.empty else 0.0
        
        return {
            'avg_speed': safe_float_convert(avg_speed),
            'max_speed': safe_float_convert(max_speed),
            'min_speed': safe_float_convert(min_speed),
            'throttle_avg': safe_float_convert(throttle_avg),
            'throttle_100_time': safe_float_convert(throttle_100_time),
            'brake_avg': safe_float_convert(brake_avg),
            'brake_applications': int(brake_applications),
            'avg_gear': safe_float_convert(avg_gear),
            'gear_changes': int(gear_changes),
            'avg_rpm': safe_float_convert(avg_rpm),
            'max_rpm': safe_float_convert(max_rpm),
            'drs_usage': safe_float_convert(drs_usage)
        }
    except Exception as e:
        print(f"Error processing telemetry data: {e}")
        return {
            'avg_speed': 0.0, 'max_speed': 0.0, 'min_speed': 0.0,
            'throttle_avg': 0.0, 'throttle_100_time': 0.0,
            'brake_avg': 0.0, 'brake_applications': 0,
            'avg_gear': 0.0, 'gear_changes': 0,
            'avg_rpm': 0.0, 'max_rpm': 0.0,
            'drs_usage': 0.0
        }

def create_output_directories(base_path: str):
    """Create necessary output directories."""
    directories = [
        os.path.join(base_path, 'raw'),
        os.path.join(base_path, 'processed'),
        'cache'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def log_progress(message: str, year: int = None):
    """Log progress with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    year_str = f" [{year}]" if year else ""
    print(f"[{timestamp}]{year_str} {message}")