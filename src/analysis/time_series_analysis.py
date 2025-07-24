"""
Time series analysis for tracking misogyny trends over time.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import ANALYSIS_SETTINGS, FIGURES_DIR
from data_collection.timeline_events import TimelineEvents

class TimeSeriesAnalyzer:
    """Analyze temporal trends in misogynistic content."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.granularity = ANALYSIS_SETTINGS['time_granularity']
        self.normalization_method = ANALYSIS_SETTINGS['normalization_method']
        self.timeline_events = TimelineEvents()
    
    def prepare_time_series(self, 
                           df: pd.DataFrame,
                           date_column: str = 'created_utc',
                           misogyny_column: str = 'is_misogynistic',
                           community_column: str = None) -> pd.DataFrame:
        """
        Prepare data for time series analysis.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            misogyny_column: Name of misogyny indicator column
            community_column: Optional community/platform column
            
        Returns:
            Time series DataFrame
        """
        # Convert date column to datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Set up grouping columns
        group_cols = [pd.Grouper(key=date_column, freq=self._get_frequency())]
        if community_column:
            group_cols.append(community_column)
        
        # Aggregate data
        aggregated = df.groupby(group_cols).agg({
            misogyny_column: ['sum', 'count', 'mean'],
            'score': ['mean', 'std'] if 'score' in df.columns else 'mean'
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = [
            'date' if col[0] == date_column else 
            ('community' if col[0] == community_column else f"{col[0]}_{col[1]}")
            for col in aggregated.columns
        ]
        
        # Rename aggregated columns
        column_mapping = {
            f'{misogyny_column}_sum': 'misogynistic_count',
            f'{misogyny_column}_count': 'total_count',
            f'{misogyny_column}_mean': 'misogyny_rate'
        }
        aggregated = aggregated.rename(columns=column_mapping)
        
        # Apply normalization
        if self.normalization_method == 'total_comments':
            aggregated['normalized_misogyny'] = (
                aggregated['misogynistic_count'] / aggregated['total_count']
            )
        elif self.normalization_method == 'total_words':
            # Would need word count data
            aggregated['normalized_misogyny'] = aggregated['misogyny_rate']
        else:
            aggregated['normalized_misogyny'] = aggregated['misogynistic_count']
        
        return aggregated
    
    def _get_frequency(self) -> str:
        """Get pandas frequency string based on granularity."""
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M'
        }
        return freq_map.get(self.granularity, 'M')
    
    def calculate_trend(self, 
                       time_series: pd.DataFrame,
                       value_column: str = 'normalized_misogyny',
                       date_column: str = 'date') -> Dict:
        """
        Calculate trend statistics.
        
        Args:
            time_series: Time series DataFrame
            value_column: Column with values to analyze
            date_column: Date column
            
        Returns:
            Dictionary with trend statistics
        """
        # Convert dates to numeric for regression
        time_series = time_series.dropna(subset=[value_column])
        x = (time_series[date_column] - time_series[date_column].min()).dt.days
        y = time_series[value_column]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate percentage change
        start_val = y.iloc[0] if len(y) > 0 else 0
        end_val = y.iloc[-1] if len(y) > 0 else 0
        pct_change = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
        
        return {
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
            'trend_strength': abs(r_value),
            'percentage_change': pct_change,
            'is_significant': p_value < 0.05
        }
    
    def analyze_event_correlation(self, 
                                 time_series: pd.DataFrame,
                                 value_column: str = 'normalized_misogyny',
                                 date_column: str = 'date',
                                 window_days: int = 30) -> pd.DataFrame:
        """
        Analyze correlation between events and misogyny trends.
        
        Args:
            time_series: Time series DataFrame
            value_column: Column with values to analyze
            date_column: Date column
            window_days: Days before/after event to analyze
            
        Returns:
            DataFrame with event correlation analysis
        """
        events_df = self.timeline_events.get_events_df()
        correlation_results = []
        
        for _, event in events_df.iterrows():
            event_date = event['date']
            
            # Define windows
            pre_start = event_date - pd.Timedelta(days=window_days)
            pre_end = event_date - pd.Timedelta(days=1)
            post_start = event_date + pd.Timedelta(days=1)
            post_end = event_date + pd.Timedelta(days=window_days)
            
            # Filter data for windows
            pre_data = time_series[
                (time_series[date_column] >= pre_start) & 
                (time_series[date_column] <= pre_end)
            ][value_column]
            
            post_data = time_series[
                (time_series[date_column] >= post_start) & 
                (time_series[date_column] <= post_end)
            ][value_column]
            
            if len(pre_data) > 0 and len(post_data) > 0:
                # Calculate statistics
                pre_mean = pre_data.mean()
                post_mean = post_data.mean()
                
                # Statistical test
                try:
                    t_stat, p_value = stats.ttest_ind(pre_data, post_data)
                    effect_size = (post_mean - pre_mean) / np.sqrt(
                        ((len(pre_data) - 1) * pre_data.var() + 
                         (len(post_data) - 1) * post_data.var()) /
                        (len(pre_data) + len(post_data) - 2)
                    )
                except:
                    t_stat, p_value, effect_size = 0, 1, 0
                
                correlation_results.append({
                    'event_date': event_date,
                    'event': event['event'],
                    'event_type': event['event_type'],
                    'pre_mean': pre_mean,
                    'post_mean': post_mean,
                    'change': post_mean - pre_mean,
                    'percent_change': ((post_mean - pre_mean) / pre_mean * 100) if pre_mean != 0 else 0,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'is_significant': p_value < 0.05,
                    'sample_size_pre': len(pre_data),
                    'sample_size_post': len(post_data)
                })
        
        return pd.DataFrame(correlation_results)
    
    def compare_communities(self, 
                           df: pd.DataFrame,
                           community_column: str,
                           date_column: str = 'created_utc',
                           misogyny_column: str = 'is_misogynistic') -> pd.DataFrame:
        """
        Compare misogyny trends across different communities.
        
        Args:
            df: Input DataFrame
            community_column: Column identifying communities
            date_column: Date column
            misogyny_column: Misogyny indicator column
            
        Returns:
            DataFrame with community comparison
        """
        community_results = []
        
        for community in df[community_column].unique():
            community_data = df[df[community_column] == community]
            
            if len(community_data) < ANALYSIS_SETTINGS['min_community_size']:
                continue
            
            # Prepare time series for this community
            ts = self.prepare_time_series(
                community_data, 
                date_column, 
                misogyny_column
            )
            
            # Calculate trend
            trend_stats = self.calculate_trend(ts)
            
            # Basic statistics
            total_comments = len(community_data)
            misogynistic_comments = community_data[misogyny_column].sum()
            misogyny_rate = misogynistic_comments / total_comments
            
            community_results.append({
                'community': community,
                'total_comments': total_comments,
                'misogynistic_comments': misogynistic_comments,
                'misogyny_rate': misogyny_rate,
                'trend_slope': trend_stats['slope'],
                'trend_r_squared': trend_stats['r_squared'],
                'trend_p_value': trend_stats['p_value'],
                'trend_direction': trend_stats['trend_direction'],
                'percentage_change': trend_stats['percentage_change'],
                'is_trending_significant': trend_stats['is_significant']
            })
        
        return pd.DataFrame(community_results).sort_values('misogyny_rate', ascending=False)
    
    def seasonal_analysis(self, 
                         time_series: pd.DataFrame,
                         value_column: str = 'normalized_misogyny',
                         date_column: str = 'date') -> Dict:
        """
        Analyze seasonal patterns in the data.
        
        Args:
            time_series: Time series DataFrame
            value_column: Column with values to analyze
            date_column: Date column
            
        Returns:
            Dictionary with seasonal analysis
        """
        df = time_series.copy()
        df['month'] = df[date_column].dt.month
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['hour'] = df[date_column].dt.hour if self.granularity == 'hourly' else None
        
        # Monthly patterns
        monthly_stats = df.groupby('month')[value_column].agg(['mean', 'std', 'count'])
        
        # Day of week patterns
        dow_stats = df.groupby('day_of_week')[value_column].agg(['mean', 'std', 'count'])
        
        # Statistical tests for seasonality
        month_f_stat, month_p_value = stats.f_oneway(*[
            df[df['month'] == month][value_column].dropna() 
            for month in df['month'].unique()
        ])
        
        dow_f_stat, dow_p_value = stats.f_oneway(*[
            df[df['day_of_week'] == dow][value_column].dropna() 
            for dow in df['day_of_week'].unique()
        ])
        
        return {
            'monthly_patterns': monthly_stats.to_dict(),
            'day_of_week_patterns': dow_stats.to_dict(),
            'monthly_seasonality_p_value': month_p_value,
            'dow_seasonality_p_value': dow_p_value,
            'has_monthly_seasonality': month_p_value < 0.05,
            'has_dow_seasonality': dow_p_value < 0.05
        }
    
    def generate_summary_report(self, 
                               time_series: pd.DataFrame,
                               community_comparison: pd.DataFrame = None,
                               event_correlation: pd.DataFrame = None) -> str:
        """
        Generate a summary report of the analysis.
        
        Args:
            time_series: Time series data
            community_comparison: Community comparison results
            event_correlation: Event correlation results
            
        Returns:
            Summary report as string
        """
        report = ["=== MISOGYNY TRACKING ANALYSIS REPORT ===\n"]
        
        # Overall trend analysis
        trend_stats = self.calculate_trend(time_series)
        report.append("OVERALL TREND:")
        report.append(f"- Direction: {trend_stats['trend_direction']}")
        report.append(f"- Slope: {trend_stats['slope']:.6f}")
        report.append(f"- R-squared: {trend_stats['r_squared']:.3f}")
        report.append(f"- Statistical significance: {'Yes' if trend_stats['is_significant'] else 'No'}")
        report.append(f"- Percentage change: {trend_stats['percentage_change']:.1f}%\n")
        
        # Community comparison
        if community_comparison is not None:
            report.append("COMMUNITY COMPARISON:")
            top_communities = community_comparison.head(5)
            for _, row in top_communities.iterrows():
                report.append(f"- {row['community']}: {row['misogyny_rate']:.3f} rate "
                            f"({row['trend_direction']} trend)")
            report.append("")
        
        # Event correlation
        if event_correlation is not None:
            significant_events = event_correlation[event_correlation['is_significant']]
            if len(significant_events) > 0:
                report.append("SIGNIFICANT EVENT CORRELATIONS:")
                for _, row in significant_events.iterrows():
                    report.append(f"- {row['event']}: {row['percent_change']:+.1f}% change "
                                f"(p={row['p_value']:.3f})")
            else:
                report.append("No statistically significant event correlations found.")
            report.append("")
        
        # Data summary
        report.append("DATA SUMMARY:")
        report.append(f"- Analysis period: {time_series['date'].min()} to {time_series['date'].max()}")
        report.append(f"- Time granularity: {self.granularity}")
        report.append(f"- Data points: {len(time_series)}")
        report.append(f"- Average misogyny rate: {time_series['normalized_misogyny'].mean():.3f}")
        
        return "\n".join(report)

def main():
    """Example usage of TimeSeriesAnalyzer."""
    # This would typically use real data
    # For demonstration, create sample data
    
    analyzer = TimeSeriesAnalyzer()
    
    # Create sample time series data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    np.random.seed(42)
    
    # Simulate increasing trend with some noise
    trend = np.linspace(0.1, 0.3, len(dates))
    noise = np.random.normal(0, 0.05, len(dates))
    misogyny_rates = np.maximum(0, trend + noise)
    
    sample_ts = pd.DataFrame({
        'date': dates,
        'normalized_misogyny': misogyny_rates,
        'total_count': np.random.randint(100, 1000, len(dates)),
        'misogynistic_count': (misogyny_rates * np.random.randint(100, 1000, len(dates))).astype(int)
    })
    
    # Analyze trend
    trend_stats = analyzer.calculate_trend(sample_ts)
    print("Trend Analysis:")
    print(f"Direction: {trend_stats['trend_direction']}")
    print(f"R-squared: {trend_stats['r_squared']:.3f}")
    print(f"Significance: {trend_stats['is_significant']}")
    
    # Analyze event correlation
    event_correlation = analyzer.analyze_event_correlation(sample_ts)
    print(f"\nEvent correlation analysis completed for {len(event_correlation)} events")
    
    # Generate report
    report = analyzer.generate_summary_report(sample_ts, event_correlation=event_correlation)
    print(f"\n{report}")

if __name__ == "__main__":
    main()
