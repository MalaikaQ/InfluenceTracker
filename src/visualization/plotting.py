"""
Visualization tools for misogyny tracking analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import VIZ_SETTINGS, FIGURES_DIR
from data_collection.timeline_events import TimelineEvents

# Set up plotting style
plt.style.use(VIZ_SETTINGS['style'])
sns.set_palette(VIZ_SETTINGS['color_palette'])

class MisogynyVisualizer:
    """Create visualizations for misogyny tracking analysis."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.figure_size = VIZ_SETTINGS['figure_size']
        self.dpi = VIZ_SETTINGS['dpi']
        self.save_formats = VIZ_SETTINGS['save_formats']
        self.timeline_events = TimelineEvents()
    
    def plot_time_series(self, 
                        time_series: pd.DataFrame,
                        value_column: str = 'normalized_misogyny',
                        date_column: str = 'date',
                        title: str = 'Misogyny Trends Over Time',
                        include_events: bool = True,
                        save_name: str = None) -> go.Figure:
        """
        Create an interactive time series plot.
        
        Args:
            time_series: Time series DataFrame
            value_column: Column with values to plot
            date_column: Date column
            title: Plot title
            include_events: Whether to include event markers
            save_name: Name to save the plot
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Main time series line
        fig.add_trace(go.Scatter(
            x=time_series[date_column],
            y=time_series[value_column],
            mode='lines+markers',
            name='Misogyny Rate',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>Misogyny Rate:</b> %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add trend line
        x_numeric = (time_series[date_column] - time_series[date_column].min()).dt.days
        z = np.polyfit(x_numeric, time_series[value_column], 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=time_series[date_column],
            y=p(x_numeric),
            mode='lines',
            name='Trend Line',
            line=dict(color='blue', width=2, dash='dash'),
            hovertemplate='<b>Trend:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Add event markers
        if include_events:
            events_df = self.timeline_events.create_event_markers(time_series[date_column])
            
            for _, event in events_df.iterrows():
                # Find the corresponding y-value for the event date
                closest_date_idx = np.argmin(np.abs(
                    (time_series[date_column] - event['date']).dt.days
                ))
                y_val = time_series.iloc[closest_date_idx][value_column]
                
                fig.add_trace(go.Scatter(
                    x=[event['date']],
                    y=[y_val],
                    mode='markers',
                    name=f"Event: {event['event'][:30]}...",
                    marker=dict(
                        size=12,
                        symbol='star',
                        color='orange',
                        line=dict(width=2, color='black')
                    ),
                    hovertemplate=f'<b>Event:</b> {event["event"]}<br>' +
                                 f'<b>Date:</b> {event["date"].strftime("%Y-%m-%d")}<br>' +
                                 '<extra></extra>',
                    showlegend=False
                ))
        
        # Customize layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title='Date',
            yaxis_title='Misogyny Rate',
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=12),
            width=1000,
            height=600
        )
        
        if save_name:
            self._save_figure(fig, save_name, is_plotly=True)
        
        return fig
    
    def plot_community_comparison(self, 
                                 community_data: pd.DataFrame,
                                 title: str = 'Misogyny Rates by Community',
                                 save_name: str = None) -> go.Figure:
        """
        Create a bar chart comparing communities.
        
        Args:
            community_data: DataFrame with community comparison data
            title: Plot title
            save_name: Name to save the plot
            
        Returns:
            Plotly figure
        """
        # Sort by misogyny rate
        community_data = community_data.sort_values('misogyny_rate', ascending=True)
        
        # Create color mapping based on trend direction
        colors = ['red' if trend == 'increasing' else 'blue' 
                 for trend in community_data['trend_direction']]
        
        fig = go.Figure(data=[
            go.Bar(
                y=community_data['community'],
                x=community_data['misogyny_rate'],
                orientation='h',
                marker_color=colors,
                text=[f"{rate:.3f}" for rate in community_data['misogyny_rate']],
                textposition='auto',
                hovertemplate='<b>Community:</b> %{y}<br>' +
                             '<b>Misogyny Rate:</b> %{x:.3f}<br>' +
                             '<b>Total Comments:</b> %{customdata[0]}<br>' +
                             '<b>Trend:</b> %{customdata[1]}<br>' +
                             '<extra></extra>',
                customdata=list(zip(
                    community_data['total_comments'],
                    community_data['trend_direction']
                ))
            )
        ])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title='Misogyny Rate',
            yaxis_title='Community',
            template='plotly_white',
            height=max(400, len(community_data) * 30),
            width=800,
            font=dict(size=12)
        )
        
        if save_name:
            self._save_figure(fig, save_name, is_plotly=True)
        
        return fig
    
    def plot_event_correlation(self, 
                              event_correlation: pd.DataFrame,
                              title: str = 'Event Impact on Misogyny',
                              save_name: str = None) -> go.Figure:
        """
        Create a visualization of event correlations.
        
        Args:
            event_correlation: DataFrame with event correlation data
            title: Plot title
            save_name: Name to save the plot
            
        Returns:
            Plotly figure
        """
        # Filter for significant events
        significant_events = event_correlation[event_correlation['is_significant']]
        
        # Create bubble chart
        fig = go.Figure()
        
        # Add all events (transparent)
        fig.add_trace(go.Scatter(
            x=event_correlation['event_date'],
            y=event_correlation['percent_change'],
            mode='markers',
            name='All Events',
            marker=dict(
                size=abs(event_correlation['effect_size']) * 20 + 5,
                color='lightgray',
                opacity=0.5,
                line=dict(width=1, color='gray')
            ),
            text=event_correlation['event'],
            hovertemplate='<b>Event:</b> %{text}<br>' +
                         '<b>Date:</b> %{x}<br>' +
                         '<b>Change:</b> %{y:.1f}%<br>' +
                         '<b>P-value:</b> %{customdata:.3f}<br>' +
                         '<extra></extra>',
            customdata=event_correlation['p_value']
        ))
        
        # Add significant events (highlighted)
        if len(significant_events) > 0:
            fig.add_trace(go.Scatter(
                x=significant_events['event_date'],
                y=significant_events['percent_change'],
                mode='markers',
                name='Significant Events',
                marker=dict(
                    size=abs(significant_events['effect_size']) * 20 + 10,
                    color=np.where(significant_events['percent_change'] > 0, 'red', 'blue'),
                    opacity=0.8,
                    line=dict(width=2, color='black')
                ),
                text=significant_events['event'],
                hovertemplate='<b>Event:</b> %{text}<br>' +
                             '<b>Date:</b> %{x}<br>' +
                             '<b>Change:</b> %{y:.1f}%<br>' +
                             '<b>P-value:</b> %{customdata:.3f}<br>' +
                             '<extra></extra>',
                customdata=significant_events['p_value']
            ))
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title='Event Date',
            yaxis_title='Percent Change in Misogyny',
            template='plotly_white',
            width=1000,
            height=600,
            font=dict(size=12)
        )
        
        if save_name:
            self._save_figure(fig, save_name, is_plotly=True)
        
        return fig
    
    def create_heatmap(self, 
                      pivot_data: pd.DataFrame,
                      title: str = 'Misogyny Heatmap',
                      save_name: str = None) -> plt.Figure:
        """
        Create a heatmap visualization.
        
        Args:
            pivot_data: Pivoted DataFrame for heatmap
            title: Plot title
            save_name: Name to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='Reds',
            center=pivot_data.mean().mean(),
            ax=ax,
            cbar_kws={'label': 'Misogyny Rate'}
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Community', fontsize=12)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name, is_plotly=False)
        
        return fig
    
    def plot_word_frequency(self, 
                           word_freq: Dict[str, int],
                           title: str = 'Most Common Words in Misogynistic Content',
                           top_n: int = 20,
                           save_name: str = None) -> go.Figure:
        """
        Create a word frequency bar chart.
        
        Args:
            word_freq: Dictionary of word frequencies
            title: Plot title
            top_n: Number of top words to show
            save_name: Name to save the plot
            
        Returns:
            Plotly figure
        """
        # Get top N words
        top_words = dict(list(word_freq.items())[:top_n])
        words = list(top_words.keys())
        frequencies = list(top_words.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=frequencies,
                y=words,
                orientation='h',
                marker_color='darkred',
                text=frequencies,
                textposition='auto',
                hovertemplate='<b>Word:</b> %{y}<br>' +
                             '<b>Frequency:</b> %{x}<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title='Frequency',
            yaxis_title='Words',
            template='plotly_white',
            height=max(400, len(words) * 25),
            width=800,
            font=dict(size=12)
        )
        
        # Reverse y-axis to show highest frequency at top
        fig.update_yaxis(categoryorder='total ascending')
        
        if save_name:
            self._save_figure(fig, save_name, is_plotly=True)
        
        return fig
    
    def create_dashboard(self, 
                        time_series: pd.DataFrame,
                        community_data: pd.DataFrame = None,
                        event_correlation: pd.DataFrame = None,
                        word_freq: Dict[str, int] = None) -> go.Figure:
        """
        Create a comprehensive dashboard.
        
        Args:
            time_series: Time series data
            community_data: Community comparison data
            event_correlation: Event correlation data
            word_freq: Word frequency data
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Misogyny Trends Over Time',
                'Community Comparison',
                'Event Impact',
                'Top Misogynistic Terms'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Time series plot
        fig.add_trace(
            go.Scatter(
                x=time_series['date'],
                y=time_series['normalized_misogyny'],
                mode='lines',
                name='Misogyny Rate',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Community comparison
        if community_data is not None:
            top_communities = community_data.head(10)
            fig.add_trace(
                go.Bar(
                    y=top_communities['community'],
                    x=top_communities['misogyny_rate'],
                    orientation='h',
                    name='Community Rates',
                    marker_color='orange'
                ),
                row=1, col=2
            )
        
        # Event correlation
        if event_correlation is not None:
            significant_events = event_correlation[event_correlation['is_significant']]
            if len(significant_events) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=significant_events['event_date'],
                        y=significant_events['percent_change'],
                        mode='markers',
                        name='Event Impact',
                        marker=dict(size=8, color='blue')
                    ),
                    row=2, col=1
                )
        
        # Word frequency
        if word_freq is not None:
            top_words = dict(list(word_freq.items())[:10])
            fig.add_trace(
                go.Bar(
                    x=list(top_words.values()),
                    y=list(top_words.keys()),
                    orientation='h',
                    name='Word Frequency',
                    marker_color='purple'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Misogyny Tracking Dashboard",
            showlegend=False,
            height=800,
            width=1200,
            template='plotly_white'
        )
        
        return fig
    
    def _save_figure(self, 
                    fig, 
                    filename: str, 
                    is_plotly: bool = True):
        """
        Save figure in multiple formats.
        
        Args:
            fig: Figure object
            filename: Base filename
            is_plotly: Whether the figure is a Plotly figure
        """
        for fmt in self.save_formats:
            filepath = FIGURES_DIR / f"{filename}.{fmt}"
            
            if is_plotly:
                if fmt == 'png':
                    fig.write_image(str(filepath), width=1000, height=600)
                elif fmt == 'pdf':
                    fig.write_image(str(filepath))
                elif fmt == 'html':
                    fig.write_html(str(filepath))
            else:
                # Matplotlib figure
                fig.savefig(str(filepath), dpi=self.dpi, bbox_inches='tight')
        
        print(f"Figure saved as {filename} in {len(self.save_formats)} formats")

def create_sample_visualizations():
    """Create sample visualizations with synthetic data."""
    visualizer = MisogynyVisualizer()
    
    # Create sample time series data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    np.random.seed(42)
    
    # Simulate data with trend and events
    base_trend = np.linspace(0.1, 0.25, len(dates))
    seasonal = 0.02 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 0.02, len(dates))
    
    sample_ts = pd.DataFrame({
        'date': dates,
        'normalized_misogyny': np.maximum(0, base_trend + seasonal + noise),
        'total_count': np.random.randint(500, 2000, len(dates))
    })
    
    # Sample community data
    communities = ['r/MensRights', 'r/TheRedPill', 'r/Feminism', 'r/gaming', 'r/politics']
    community_data = pd.DataFrame({
        'community': communities,
        'misogyny_rate': [0.45, 0.52, 0.08, 0.18, 0.15],
        'total_comments': [1500, 800, 1200, 3000, 2500],
        'trend_direction': ['increasing', 'increasing', 'decreasing', 'stable', 'increasing']
    })
    
    # Sample word frequency
    word_freq = {
        'women': 1250, 'female': 980, 'girls': 750, 'hypergamy': 450,
        'chad': 380, 'beta': 320, 'alpha': 280, 'feminism': 250,
        'dating': 220, 'marriage': 180
    }
    
    # Create visualizations
    print("Creating sample visualizations...")
    
    # Time series plot
    ts_fig = visualizer.plot_time_series(sample_ts, save_name='sample_time_series')
    
    # Community comparison
    comm_fig = visualizer.plot_community_comparison(community_data, 
                                                   save_name='sample_community_comparison')
    
    # Word frequency
    word_fig = visualizer.plot_word_frequency(word_freq, save_name='sample_word_frequency')
    
    # Dashboard
    dashboard_fig = visualizer.create_dashboard(sample_ts, community_data, 
                                              word_freq=word_freq)
    visualizer._save_figure(dashboard_fig, 'sample_dashboard', is_plotly=True)
    
    print("Sample visualizations created successfully!")
    
    return {
        'time_series': ts_fig,
        'community_comparison': comm_fig,
        'word_frequency': word_fig,
        'dashboard': dashboard_fig
    }

def main():
    """Example usage of MisogynyVisualizer."""
    create_sample_visualizations()

if __name__ == "__main__":
    main()
