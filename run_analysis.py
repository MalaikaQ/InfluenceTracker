#!/usr/bin/env python3
"""
Misogyny Tracking Analysis - Main Execution Script

This script provides multiple ways to run the analysis:
1. Quick demo with synthetic data
2. Full analysis with real API data
3. Interactive data exploration

Usage:
    python run_analysis.py --mode demo
    python run_analysis.py --mode full
    python run_analysis.py --mode interactive
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.config import load_config
    from src.data_collection.reddit_scraper import RedditScraper
    from src.data_collection.youtube_scraper import YouTubeScraper
    from src.analysis.misogyny_detector import MisogynyDetector
    from src.analysis.time_series_analysis import TimeSeriesAnalyzer
    from src.visualization.plotting import MisogynyVisualizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed and the project structure is correct.")
    sys.exit(1)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_synthetic_data(n_samples=1000):
    """Create synthetic data for demonstration purposes."""
    print("Creating synthetic data for demonstration...")
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years of data
    
    # Generate random dates
    dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Create synthetic text data with varying levels of misogynistic content
    neutral_texts = [
        "Great discussion about technology trends",
        "Looking forward to the conference next week",
        "Thanks for sharing this interesting article",
        "What do you think about the new policy?",
        "Having a productive day at work"
    ]
    
    mild_misogynistic_texts = [
        "Women just don't understand technology",
        "She probably got the job because of diversity quotas",
        "Typical female behavior in the workplace",
        "Women are too emotional for leadership roles",
        "She's probably just trying to get attention"
    ]
    
    strong_misogynistic_texts = [
        "Women belong in the kitchen not in tech",
        "All women are manipulative and can't be trusted",
        "Female CEOs always ruin companies",
        "Women only care about looks and money",
        "Feminism has destroyed modern society"
    ]
    
    # Generate synthetic data
    data = []
    for i, date in enumerate(dates):
        # Vary the probability of misogynistic content over time
        # Simulate an increase in recent years
        time_factor = (date - start_date).days / 365  # Years since start
        misogyny_prob = 0.1 + 0.2 * (time_factor / 3)  # Increase over time
        
        if np.random.random() < misogyny_prob:
            if np.random.random() < 0.7:
                text = random.choice(mild_misogynistic_texts)
                misogyny_score = np.random.uniform(0.3, 0.7)
            else:
                text = random.choice(strong_misogynistic_texts)
                misogyny_score = np.random.uniform(0.7, 0.95)
        else:
            text = random.choice(neutral_texts)
            misogyny_score = np.random.uniform(0.0, 0.3)
        
        data.append({
            'text': text,
            'timestamp': date,
            'misogyny_score': misogyny_score,
            'score': np.random.uniform(1, 5),  # Add generic score column
            'is_misogynistic': 1 if misogyny_score > 0.5 else 0,  # Binary misogyny indicator
            'platform': random.choice(['reddit', 'youtube', 'twitter']),
            'community': random.choice([
                'technology', 'gaming', 'politics', 'dating', 
                'fitness', 'career', 'relationships', 'general'
            ]),
            'created_utc': date  # Add the expected date column name
        })
    
    return pd.DataFrame(data)

def run_demo_analysis():
    """Run analysis with synthetic data for demonstration."""
    print("=== MISOGYNY TRACKING ANALYSIS - DEMO MODE ===\n")
    
    try:
        # Create synthetic data
        df = create_synthetic_data(2000)
        print(f"Generated {len(df)} synthetic data points")
        
        # Initialize analyzer and visualizer
        analyzer = TimeSeriesAnalyzer()
        visualizer = MisogynyVisualizer()
        
        # Perform time series analysis
        print("\n1. Analyzing time series trends...")
        
        # Prepare time series data
        ts_data = analyzer.prepare_time_series(
            df,
            date_column='created_utc',
            misogyny_column='is_misogynistic'
        )
        
        # Calculate trend
        trends = analyzer.calculate_trend(
            ts_data,
            value_column='normalized_misogyny'
        )
        
        print(f"Overall trend slope: {trends['slope']:.4f}")
        print(f"Trend significance (p-value): {trends['p_value']:.4f}")
        
        # Analyze by platform
        print("\n2. Analyzing by platform...")
        platform_stats = df.groupby('platform')['misogyny_score'].agg([
            'mean', 'std', 'count'
        ]).round(3)
        print(platform_stats)
        
        # Analyze by community
        print("\n3. Analyzing by community...")
        community_stats = df.groupby('community')['misogyny_score'].agg([
            'mean', 'std', 'count'
        ]).round(3).sort_values('mean', ascending=False)
        print(community_stats)
        
        # Create visualizations
        print("\n4. Creating visualizations...")
        
        # Prepare data for visualization
        time_series_data = df.groupby(df['timestamp'].dt.date).agg({
            'misogyny_score': 'mean'
        }).reset_index()
        time_series_data.columns = ['date', 'normalized_misogyny']
        
        # Community data for visualization
        community_viz_data = community_stats.reset_index()
        community_viz_data.columns = ['community', 'misogyny_rate', 'std_dev', 'total_comments']
        community_viz_data['trend_direction'] = ['increasing' if rate > 0.3 else 'stable' for rate in community_viz_data['misogyny_rate']]
        
        # Time series plot
        try:
            fig1 = visualizer.plot_time_series(
                time_series_data, 
                date_column='date',
                value_column='normalized_misogyny',
                save_name='demo_time_series'
            )
            print("   ✓ Time series plot created")
        except Exception as e:
            print(f"   ✗ Error creating time series plot: {e}")
        
        # Platform comparison (convert to community comparison format)
        try:
            fig2 = visualizer.plot_community_comparison(
                community_viz_data,
                save_name='demo_community_comparison'
            )
            print("   ✓ Community comparison plot created")
        except Exception as e:
            print(f"   ✗ Error creating community comparison plot: {e}")
        
        print("\nDemo analysis complete!")
        print("\nKey Findings from Synthetic Data:")
        print("- Misogynistic content shows increasing trend over time")
        print("- Different platforms show varying levels of misogynistic content")
        print("- Certain communities have higher concentrations of problematic content")
        
        return df, trends
        
    except Exception as e:
        print(f"Error during demo analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_full_analysis():
    """Run full analysis with real API data."""
    print("=== MISOGYNY TRACKING ANALYSIS - FULL MODE ===\n")
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize data collectors
        reddit_scraper = RedditScraper()
        youtube_scraper = YouTubeScraper()
        
        # Initialize analysis components
        detector = MisogynyDetector()
        analyzer = TimeSeriesAnalyzer()
        visualizer = MisogynyVisualizer()
        
        print("1. Collecting Reddit data...")
        reddit_data = reddit_scraper.collect_recent_data(limit=1000)
        
        print("2. Collecting YouTube data...")
        youtube_data = youtube_scraper.collect_recent_data(limit=1000)
        
        # Combine data
        all_data = pd.concat([reddit_data, youtube_data], ignore_index=True)
        print(f"Total data points collected: {len(all_data)}")
        
        print("\n3. Training misogyny detection model...")
        detector.train_classifier(all_data['text'])
        
        print("4. Analyzing misogyny in collected data...")
        misogyny_scores = detector.predict_misogyny(all_data['text'])
        all_data['misogyny_score'] = misogyny_scores
        
        print("5. Performing time series analysis...")
        trends = analyzer.analyze_trends(
            all_data,
            timestamp_col='timestamp',
            misogyny_col='misogyny_score'
        )
        
        print("6. Creating comprehensive visualizations...")
        # Generate all visualization reports
        visualizer.create_comprehensive_report(all_data)
        
        print("Full analysis complete!")
        return all_data, trends
        
    except Exception as e:
        print(f"Error during full analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_interactive_mode():
    """Run interactive data exploration mode."""
    print("=== MISOGYNY TRACKING ANALYSIS - INTERACTIVE MODE ===\n")
    print("Starting Jupyter Lab for interactive analysis...")
    print("Open the notebook: notebooks/misogyny_tracking_analysis.ipynb")
    
    # Launch Jupyter Lab
    os.system("/opt/anaconda3/bin/jupyter lab notebooks/misogyny_tracking_analysis.ipynb")

def main():
    parser = argparse.ArgumentParser(
        description="Misogyny Tracking Analysis Tool"
    )
    parser.add_argument(
        '--mode',
        choices=['demo', 'full', 'interactive'],
        default='demo',
        help='Analysis mode: demo (synthetic data), full (real APIs), or interactive (Jupyter)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'demo':
            run_demo_analysis()
        elif args.mode == 'full':
            run_full_analysis()
        elif args.mode == 'interactive':
            run_interactive_mode()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your configuration and try again.")

if __name__ == "__main__":
    main()
