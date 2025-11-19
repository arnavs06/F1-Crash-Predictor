"""
Data collection module for F1 telemetry data from OpenF1 API.
Automated ETL pipeline to gather race data, lap times, telemetry, and weather information.
"""

import requests
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Optional
from tqdm import tqdm
import json
import os


class OpenF1DataCollector:
    """Collects F1 data from the OpenF1 API."""
    
    BASE_URL = "https://api.openf1.org/v1"
    
    def __init__(self):
        self.session = requests.Session()
        
    def get_sessions(self, year: int = 2024, limit: int = 20) -> pd.DataFrame:
        """Get race sessions for a given year."""
        url = f"{self.BASE_URL}/sessions"
        params = {
            'year': year,
            'session_type': 'Race'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            sessions = response.json()
            
            if isinstance(sessions, list) and len(sessions) > 0:
                df = pd.DataFrame(sessions)
                return df.head(limit)
            else:
                print(f"No sessions found for year {year}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching sessions: {e}")
            return pd.DataFrame()
    
    def get_lap_data(self, session_key: int) -> pd.DataFrame:
        """Get lap data for a specific session."""
        url = f"{self.BASE_URL}/laps"
        params = {'session_key': session_key}
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            laps = response.json()
            
            if isinstance(laps, list) and len(laps) > 0:
                return pd.DataFrame(laps)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching lap data for session {session_key}: {e}")
            return pd.DataFrame()
    
    def get_driver_data(self, session_key: int) -> pd.DataFrame:
        """Get driver information for a specific session."""
        url = f"{self.BASE_URL}/drivers"
        params = {'session_key': session_key}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            drivers = response.json()
            
            if isinstance(drivers, list) and len(drivers) > 0:
                return pd.DataFrame(drivers)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching driver data: {e}")
            return pd.DataFrame()
    
    def get_weather_data(self, session_key: int) -> pd.DataFrame:
        """Get weather data for a specific session."""
        url = f"{self.BASE_URL}/weather"
        params = {'session_key': session_key}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            weather = response.json()
            
            if isinstance(weather, list) and len(weather) > 0:
                return pd.DataFrame(weather)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return pd.DataFrame()
    
    def get_position_data(self, session_key: int, driver_number: Optional[int] = None) -> pd.DataFrame:
        """Get position data for a specific session."""
        url = f"{self.BASE_URL}/position"
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            positions = response.json()
            
            if isinstance(positions, list) and len(positions) > 0:
                df = pd.DataFrame(positions)
                # Sample the data if too large
                if len(df) > 1000:
                    df = df.sample(1000, random_state=42)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching position data: {e}")
            return pd.DataFrame()
    
    def collect_comprehensive_data(self, year: int = 2024, max_sessions: int = 5) -> pd.DataFrame:
        """
        Collect comprehensive F1 data including laps, weather, and positions.
        
        Args:
            year: Year to collect data from
            max_sessions: Maximum number of sessions to collect
        
        Returns:
            DataFrame with comprehensive telemetry data
        """
        print(f"\n{'='*60}")
        print(f"Collecting F1 Data from OpenF1 API (Year: {year})")
        print(f"{'='*60}\n")
        
        # Get sessions
        print("Fetching sessions...")
        sessions_df = self.get_sessions(year=year, limit=max_sessions)
        
        if sessions_df.empty:
            print("No sessions found. Creating synthetic data...")
            return self._create_synthetic_data()
        
        print(f"Found {len(sessions_df)} sessions")
        
        all_data = []
        
        for idx, session in tqdm(sessions_df.iterrows(), total=len(sessions_df), desc="Processing sessions"):
            session_key = session.get('session_key')
            if not session_key:
                continue
            
            # Get lap data
            lap_data = self.get_lap_data(session_key)
            if lap_data.empty:
                continue
            
            # Get weather data
            weather_data = self.get_weather_data(session_key)
            
            # Add session info
            lap_data['session_key'] = session_key
            lap_data['session_name'] = session.get('session_name', f'Session_{session_key}')
            lap_data['circuit'] = session.get('circuit_short_name', 'Unknown')
            lap_data['country'] = session.get('country_name', 'Unknown')
            
            # Merge weather data if available
            if not weather_data.empty:
                # Take average weather conditions for the session
                avg_weather = weather_data.select_dtypes(include=[np.number]).mean()
                for col in avg_weather.index:
                    lap_data[f'weather_{col}'] = avg_weather[col]
                
                # Add categorical weather data
                if 'rainfall' in weather_data.columns:
                    lap_data['weather_rainfall'] = weather_data['rainfall'].mode()[0] if len(weather_data) > 0 else 0
                if 'track_temperature' in weather_data.columns:
                    lap_data['weather_track_temp'] = weather_data['track_temperature'].mean()
                if 'air_temperature' in weather_data.columns:
                    lap_data['weather_air_temp'] = weather_data['air_temperature'].mean()
            
            all_data.append(lap_data)
            
            # Rate limiting
            time.sleep(0.5)
        
        if not all_data:
            print("No data collected. Creating synthetic data...")
            return self._create_synthetic_data()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\nCollected {len(combined_df)} lap records from {len(sessions_df)} sessions")
        
        return combined_df
    
    def _create_synthetic_data(self, n_records: int = 1500) -> pd.DataFrame:
        """
        Create synthetic F1 telemetry data for testing.
        This simulates realistic racing data with various parameters.
        """
        print(f"\nGenerating {n_records} synthetic lap records...")
        
        np.random.seed(42)
        
        # Driver numbers (typical F1 grid)
        drivers = list(range(1, 21))
        
        # Circuits
        circuits = ['Monaco', 'Silverstone', 'Monza', 'Spa', 'Suzuka', 'COTA', 'Interlagos']
        
        data = []
        
        for i in range(n_records):
            lap_num = (i % 60) + 1  # Up to 60 laps
            driver = np.random.choice(drivers)
            circuit = np.random.choice(circuits)
            
            # Base lap time with variation
            base_time = 85 + np.random.normal(0, 3)  # ~85 seconds base
            
            # Tire degradation effect (increases lap time)
            tire_age = np.random.randint(1, 30)
            tire_degradation = tire_age * 0.05
            
            # Weather effects
            track_temp = np.random.uniform(25, 45)
            air_temp = np.random.uniform(20, 35)
            rainfall = np.random.choice([0, 1], p=[0.85, 0.15])  # 15% chance of rain
            
            if rainfall:
                base_time += np.random.uniform(5, 15)  # Rain adds time
            
            lap_time = base_time + tire_degradation + np.random.normal(0, 1)
            
            # Position and changes
            position = np.random.randint(1, 21)
            
            # Speed (km/h)
            speed = np.random.uniform(200, 340)
            if rainfall:
                speed *= 0.85  # Lower speed in rain
            
            # Sector times (should sum approximately to lap time)
            s1 = lap_time * np.random.uniform(0.30, 0.35)
            s2 = lap_time * np.random.uniform(0.32, 0.37)
            s3 = lap_time - s1 - s2
            
            record = {
                'driver_number': driver,
                'lap_number': lap_num,
                'lap_duration': lap_time,
                'duration_sector_1': s1,
                'duration_sector_2': s2,
                'duration_sector_3': s3,
                'speed': speed,
                'position': position,
                'circuit': circuit,
                'tire_age': tire_age,
                'weather_track_temp': track_temp,
                'weather_air_temp': air_temp,
                'weather_rainfall': rainfall,
                'session_key': 1000 + (i // 100),  # Multiple sessions
                'session_name': f'Race_{(i // 100) + 1}',
                'country': circuit,
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        print(f"Generated {len(df)} synthetic lap records")
        
        return df


def main():
    """Main data collection pipeline."""
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    
    # Initialize collector
    collector = OpenF1DataCollector()
    
    # Collect data
    print("Starting data collection...")
    df = collector.collect_comprehensive_data(year=2024, max_sessions=5)
    
    # Save raw data
    output_path = 'data/raw/f1_telemetry_raw.csv'
    df.to_csv(output_path, index=False)
    print(f"\nRaw data saved to {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    return df


if __name__ == "__main__":
    main()

