"""
Cultural timeline and event tracking for correlation analysis.
"""
import pandas as pd
import datetime
from typing import Dict, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import CULTURAL_TIMELINE, RAW_DATA_DIR

class TimelineEvents:
    """Manage cultural timeline events for correlation analysis."""
    
    def __init__(self):
        """Initialize with predefined events."""
        self.events = CULTURAL_TIMELINE.copy()
    
    def add_event(self, date: str, description: str):
        """
        Add a new event to the timeline.
        
        Args:
            date: Date in YYYY-MM-DD format
            description: Event description
        """
        self.events[date] = description
    
    def get_events_df(self) -> pd.DataFrame:
        """
        Get events as a DataFrame.
        
        Returns:
            DataFrame with date and event columns
        """
        events_list = []
        for date_str, description in self.events.items():
            events_list.append({
                'date': pd.to_datetime(date_str),
                'event': description,
                'event_type': self._categorize_event(description)
            })
        
        df = pd.DataFrame(events_list)
        return df.sort_values('date')
    
    def _categorize_event(self, description: str) -> str:
        """
        Categorize events based on description.
        
        Args:
            description: Event description
            
        Returns:
            Event category
        """
        description_lower = description.lower()
        
        if 'ban' in description_lower or 'banned' in description_lower:
            return 'platform_ban'
        elif 'arrest' in description_lower:
            return 'legal_action'
        elif 'release' in description_lower:
            return 'legal_release'
        elif 'rise' in description_lower or 'peak' in description_lower:
            return 'popularity_peak'
        elif 'interview' in description_lower:
            return 'media_appearance'
        else:
            return 'other'
    
    def get_events_in_range(self, 
                           start_date: datetime.datetime,
                           end_date: datetime.datetime) -> pd.DataFrame:
        """
        Get events within a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Filtered DataFrame
        """
        df = self.get_events_df()
        return df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    def create_event_markers(self, 
                           date_column: pd.Series) -> pd.DataFrame:
        """
        Create event markers for time series plotting.
        
        Args:
            date_column: Series of dates from your data
            
        Returns:
            DataFrame with event markers
        """
        events_df = self.get_events_df()
        
        # Find events that fall within the data range
        min_date = date_column.min()
        max_date = date_column.max()
        
        relevant_events = events_df[
            (events_df['date'] >= min_date) & 
            (events_df['date'] <= max_date)
        ].copy()
        
        return relevant_events
    
    def save_timeline(self, filename: str = 'cultural_timeline'):
        """Save timeline to CSV."""
        df = self.get_events_df()
        filepath = RAW_DATA_DIR / f"{filename}.csv"
        df.to_csv(filepath, index=False)
        print(f"Timeline saved to {filepath}")

def create_extended_timeline() -> TimelineEvents:
    """
    Create an extended timeline with additional events.
    
    Returns:
        TimelineEvents object with extended timeline
    """
    timeline = TimelineEvents()
    
    # Add more Andrew Tate related events
    additional_events = {
        '2017-01-01': 'Andrew Tate starts gaining online presence',
        '2020-01-01': 'Andrew Tate content begins appearing on TikTok',
        '2021-06-01': 'Andrew Tate moves to Romania',
        '2022-01-01': 'Peak Andrew Tate TikTok virality begins',
        '2022-07-01': 'Andrew Tate Twitter account gains massive following',
        '2022-08-01': 'Andrew Tate appears on various podcasts',
        '2022-09-01': 'Backlash and criticism intensifies',
        '2023-01-01': 'Andrew Tate legal proceedings ongoing',
        
        # Other red-pill influencer events
        '2020-06-01': 'Fresh & Fit podcast launches',
        '2021-03-01': 'Sneako gains popularity on YouTube',
        '2022-02-01': 'Red-pill content surge on multiple platforms',
        
        # Platform policy changes
        '2022-01-01': 'TikTok begins cracking down on misogynistic content',
        '2022-06-01': 'YouTube updates hate speech policies',
        '2023-01-01': 'Twitter/X policy changes under new ownership',
        
        # Cultural/political events
        '2022-06-24': 'Roe v. Wade overturned - cultural tensions rise',
        '2021-12-01': 'Me Too movement continues influencing discourse',
        '2020-03-01': 'COVID-19 pandemic affects online behavior'
    }
    
    for date, description in additional_events.items():
        timeline.add_event(date, description)
    
    return timeline

def analyze_event_correlation_windows(events_df: pd.DataFrame, 
                                    window_days: int = 30) -> pd.DataFrame:
    """
    Create correlation analysis windows around events.
    
    Args:
        events_df: DataFrame with events
        window_days: Days before/after event to analyze
        
    Returns:
        DataFrame with analysis windows
    """
    windows = []
    
    for _, event in events_df.iterrows():
        event_date = event['date']
        
        window_data = {
            'event_date': event_date,
            'event': event['event'],
            'event_type': event['event_type'],
            'window_start': event_date - pd.Timedelta(days=window_days),
            'window_end': event_date + pd.Timedelta(days=window_days),
            'pre_event_start': event_date - pd.Timedelta(days=window_days),
            'pre_event_end': event_date - pd.Timedelta(days=1),
            'post_event_start': event_date + pd.Timedelta(days=1),
            'post_event_end': event_date + pd.Timedelta(days=window_days)
        }
        windows.append(window_data)
    
    return pd.DataFrame(windows)

def main():
    """Example usage of TimelineEvents."""
    # Create basic timeline
    timeline = TimelineEvents()
    print("Basic timeline events:")
    print(timeline.get_events_df())
    
    # Create extended timeline
    extended_timeline = create_extended_timeline()
    print(f"\nExtended timeline has {len(extended_timeline.events)} events")
    
    # Save timeline
    extended_timeline.save_timeline('extended_cultural_timeline')
    
    # Create analysis windows
    events_df = extended_timeline.get_events_df()
    windows_df = analyze_event_correlation_windows(events_df)
    
    # Save analysis windows
    windows_filepath = RAW_DATA_DIR / "event_analysis_windows.csv"
    windows_df.to_csv(windows_filepath, index=False)
    print(f"Analysis windows saved to {windows_filepath}")

if __name__ == "__main__":
    main()
