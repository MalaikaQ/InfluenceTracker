# InfluenceTracker# Tracking Misogyny in Online Communities: A Longitudinal Analysis

## Project Overview
This project analyzes the correlation between misogynistic language in online communities and the rise of red-pill influencers like Andrew Tate.

## Core Research Questions
1. Has misogynistic language in online communities increased over time?
2. Is there a measurable spike in misogynistic content correlated with the rise of Andrew Tate and similar "red-pilled" influencers?
3. Which types of communities (e.g., subreddit categories, YouTube channels) are most affected?

## Data Sources
- **Reddit**: Comments from various subreddits (r/MensRights, r/Incels, r/Feminism, r/Gaming, etc.)
- **YouTube**: Comments from Andrew Tate and related influencer videos
- **Cultural Timeline**: Key events and milestones for influencers

## Project Structure
```
├── src/
│   ├── data_collection/
│   │   ├── reddit_scraper.py
│   │   ├── youtube_scraper.py
│   │   └── timeline_events.py
│   ├── analysis/
│   │   ├── misogyny_detector.py
│   │   ├── time_series_analysis.py
│   │   └── community_comparison.py
│   ├── visualization/
│   │   ├── plotting.py
│   │   └── dashboard.py
│   └── utils/
│       ├── text_processing.py
│       └── config.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── lexicons/
├── notebooks/
├── results/
│   ├── figures/
│   └── reports/
└── requirements.txt
```

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Configure API keys in `src/utils/config.py`
3. Run data collection scripts
4. Execute analysis notebooks

## Methodology
1. **Data Collection + Cleaning**: API scrapers, text normalization
2. **Misogyny Detection**: Hybrid approach using regex and ML classifiers
3. **Time Series Analysis**: Frequency analysis with cultural event overlays
4. **Community Comparison**: Cross-platform and cross-community analysis
5. **Visualization**: Interactive dashboards and publication-ready plots

## Ethics and Considerations
- All data collection follows platform ToS and ethical guidelines
- Personal information is anonymized
- Transparent methodology for misogyny detection
- Proper normalization to account for platform growth
