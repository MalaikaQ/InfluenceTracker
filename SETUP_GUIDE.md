# Misogyny Tracking Analysis - Setup Guide

## Project Overview
This project tracks misogynistic language in online communities and analyzes its correlation with the rise of red-pill influencers like Andrew Tate.

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API credentials
# - Reddit API: https://www.reddit.com/prefs/apps
# - YouTube API: https://console.developers.google.com
# - Twitter API (optional): https://developer.twitter.com
```

### 3. Run Analysis
```bash
# Option 1: Use Jupyter Notebook (Recommended)
jupyter lab notebooks/misogyny_tracking_analysis.ipynb

# Option 2: Run individual scripts
python src/data_collection/reddit_scraper.py
python src/analysis/misogyny_detector.py
python src/visualization/plotting.py
```

## Project Structure
```
├── src/                    # Source code
│   ├── data_collection/    # API scrapers
│   ├── analysis/          # Misogyny detection & time series
│   ├── visualization/     # Plotting and dashboards
│   └── utils/            # Configuration and text processing
├── data/                  # Data storage
│   ├── raw/              # Raw collected data
│   ├── processed/        # Cleaned and analyzed data
│   └── lexicons/         # Misogyny term lexicons
├── notebooks/            # Jupyter notebooks
├── results/              # Analysis outputs
│   ├── figures/          # Generated plots
│   └── reports/          # Analysis reports
└── requirements.txt      # Python dependencies
```

## Key Features

### Data Collection
- **Reddit Scraper**: Collect comments from target subreddits
- **YouTube Scraper**: Gather comments from influencer videos
- **Timeline Events**: Track cultural milestones and events

### Analysis Pipeline
- **Text Processing**: Clean and normalize social media text
- **Hybrid Misogyny Detection**: Lexicon + Machine Learning approach
- **Time Series Analysis**: Track trends and correlations
- **Community Comparison**: Compare platforms and communities

### Visualization
- **Interactive Time Series**: Trends with event overlays
- **Community Heatmaps**: Compare misogyny rates across platforms
- **Event Correlation**: Impact analysis of key events
- **Comprehensive Dashboard**: All insights in one view

## Research Questions Addressed

1. **Has misogynistic language in online communities increased over time?**
   - Time series analysis with trend detection
   - Statistical significance testing

2. **Is there correlation with red-pill influencer rise?**
   - Event correlation analysis around key dates
   - Before/after impact measurement

3. **Which communities are most affected?**
   - Cross-platform comparison
   - Community-specific trend analysis

## Methodology

### 1. Data Collection
- Reddit API for subreddit comments
- YouTube API for video comments
- Cultural timeline of key events

### 2. Misogyny Detection
- Curated lexicon of misogynistic terms
- Machine learning classifier for context
- Hybrid scoring system

### 3. Analysis
- Monthly aggregation and normalization
- Linear trend analysis
- Event correlation testing
- Cross-community comparison

### 4. Visualization
- Interactive Plotly dashboards
- Publication-ready matplotlib figures
- Statistical summaries and reports

## Sample Results

The analysis provides insights such as:
- Overall trend direction and magnitude
- Statistical significance of trends
- Impact of specific events (bans, arrests, viral content)
- Platform/community risk assessment
- Intervention recommendations

## Ethics and Limitations

### Ethical Considerations
- All data collection follows platform ToS
- Personal information is anonymized
- Research aims to reduce online harm

### Limitations
- Synthetic data used in demonstration
- Limited to English language content
- Subjective nature of misogyny classification
- Platform API limitations

## Extensions and Future Work

1. **Expand Data Sources**: Twitter, TikTok, Discord
2. **Advanced NLP**: Transformer models, sentiment analysis
3. **Real-time Monitoring**: Live dashboard updates
4. **Intervention Testing**: A/B test counter-messaging
5. **Cross-cultural Analysis**: Multi-language support

## Support

For questions or issues:
1. Check the documentation in each module
2. Review the Jupyter notebook examples
3. Examine the synthetic data generation for reference
4. Adapt API credentials and rate limiting as needed

## License

This project is for educational and research purposes. Please ensure compliance with platform Terms of Service and ethical research standards when collecting real data.
