#!/usr/bin/env python3
"""
Main analysis script that runs the complete misogyny tracking pipeline.
"""

import sys
from pathlib import Path
import pandas as pd
import datetime

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from utils.config import *
from utils.text_processing import TextProcessor
from data_collection.reddit_scraper import RedditScraper
from data_collection.youtube_scraper import YouTubeScraper
from data_collection.timeline_events import create_extended_timeline
from analysis.misogyny_detector import MisogynyDetector, create_synthetic_training_data
from analysis.time_series_analysis import TimeSeriesAnalyzer
from visualization.plotting import MisogynyVisualizer

def main():
    """Run the complete analysis pipeline."""
    print("🚀 Starting Misogyny Tracking Analysis Pipeline")
    print(f"📅 Analysis started at: {datetime.datetime.now()}")
    
    # Initialize components
    print("\n🔧 Initializing components...")
    text_processor = TextProcessor()
    misogyny_detector = MisogynyDetector()
    time_series_analyzer = TimeSeriesAnalyzer()
    visualizer = MisogynyVisualizer()
    timeline_events = create_extended_timeline()
    
    # For demonstration, create synthetic data
    # In production, you would collect real data here
    print("\n📊 Creating synthetic data for demonstration...")
    
    # This would be replaced with real data collection:
    # reddit_scraper = RedditScraper()
    # youtube_scraper = YouTubeScraper()
    # reddit_data = reddit_scraper.collect_by_category('mens_rights')
    # youtube_data = youtube_scraper.collect_influencer_data('red_pill_influencers')
    
    print("   Note: In production, replace this with real API data collection")
    
    # Train misogyny detection model
    print("\n🤖 Training misogyny detection model...")
    training_data = create_synthetic_training_data()
    training_results = misogyny_detector.train_classifier(training_data)
    print(f"   Model accuracy: {training_results['test_accuracy']:.3f}")
    
    # Save trained model
    misogyny_detector.save_model('production_misogyny_detector')
    
    # Create timeline analysis
    print("\n📅 Analyzing cultural timeline...")
    timeline_df = timeline_events.get_events_df()
    timeline_events.save_timeline('extended_cultural_timeline')
    print(f"   Timeline events: {len(timeline_df)}")
    
    # Generate sample visualizations
    print("\n📈 Creating sample visualizations...")
    sample_figures = visualizer.create_sample_visualizations()
    
    print(f"\n✅ Analysis pipeline completed successfully!")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print(f"📊 Figures saved to: {FIGURES_DIR}")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Configure API credentials in .env file")
    print(f"   2. Run the Jupyter notebook: notebooks/misogyny_tracking_analysis.ipynb")
    print(f"   3. Collect real data using the scraper modules")
    print(f"   4. Adapt the analysis for your specific research questions")

if __name__ == "__main__":
    main()
