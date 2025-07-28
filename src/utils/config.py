"""
Configuration settings for the misogyny tracking project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LEXICONS_DIR = DATA_DIR / "lexicons"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LEXICONS_DIR, 
                  RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID', 'your_reddit_client_id'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET', 'your_reddit_client_secret'),
    'user_agent': os.getenv('REDDIT_USER_AGENT', 'MisogynyTracker/1.0'),
    'username': os.getenv('REDDIT_USERNAME', 'your_username'),
    'password': os.getenv('REDDIT_PASSWORD', 'your_password')
}

YOUTUBE_CONFIG = {
    'api_key': os.getenv('YOUTUBE_API_KEY', 'your_youtube_api_key'),
    'quota_limit': 10000  # Daily quota limit
}

TWITTER_CONFIG = {
    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN', 'your_twitter_bearer_token'),
    'api_key': os.getenv('TWITTER_API_KEY', 'your_twitter_api_key'),
    'api_secret': os.getenv('TWITTER_API_SECRET', 'your_twitter_api_secret'),
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN', 'your_twitter_access_token'),
    'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET', 'your_twitter_access_token_secret')
}

# Target communities for data collection
REDDIT_COMMUNITIES = {
    'mens_rights': ['MensRights', 'MGTOW', 'TheRedPill'],
    'incel_related': ['IncelsWithoutHate', 'ForeverAlone'],
    'feminist': ['Feminism', 'TwoXChromosomes', 'AskFeminists'],
    'gaming': ['gaming', 'pcmasterrace', 'PS4', 'xbox'],
    'general': ['AskReddit', 'politics', 'worldnews', 'unpopularopinion'],
    'discussion': ['changemyview', 'relationship_advice', 'dating_advice']
}

# YouTube channels and influencers to track
YOUTUBE_TARGETS = {
    'red_pill_influencers': [
        'UCbsSYTwOqUBo4YOj6wdz_6w',  # Andrew Tate (example)
        'UCKp8fzSNYSYpa3Drqh1jGLg',  # Fresh & Fit (example)
        'UC9-y-6csu5WGm29I7JiwpnA'   # Sneako (example)
    ],
    'feminist_creators': [
        'UC5fdssPqmmGhkhsJi4VcckA',  # ContraPoints (example)
        'UCyVWrcNWWAnb2hZIvGrkY5Q'   # Lindsay Ellis (example)
    ],
    'mainstream': [
        'UCsT0YIqwnpJCM-mx7-gSA4Q',  # TEDx Talks (example)
        'UC-lHJZR3Gqxm24_Vd_AJ5Yw'   # PewDiePie (example)
    ]
}

# Timeline of key events
CULTURAL_TIMELINE = {
    '2022-08-19': 'Andrew Tate banned from Facebook, Instagram, TikTok',
    '2022-08-29': 'Andrew Tate banned from YouTube',
    '2022-12-29': 'Andrew Tate arrested in Romania',
    '2023-03-31': 'Andrew Tate released to house arrest',
    '2021-01-01': 'Rise of red-pill content on TikTok',
    '2022-06-01': 'Peak Andrew Tate TikTok presence'
}

# Text processing settings
TEXT_PROCESSING = {
    'min_comment_length': 10,
    'max_comment_length': 1000,
    'languages': ['en'],
    'remove_deleted': True,
    'remove_bot_comments': True
}

# Misogyny detection settings
MISOGYNY_DETECTION = {
    'confidence_threshold': 0.7,
    'use_lexicon': True,
    'use_ml_classifier': True,
    'lexicon_weight': 0.3,
    'ml_weight': 0.7
}

# Analysis settings
ANALYSIS_SETTINGS = {
    'time_granularity': 'monthly',  # daily, weekly, monthly
    'normalization_method': 'total_comments',  # total_comments, total_words, none
    'min_community_size': 100,  # Minimum number of comments for analysis
    'correlation_window': 30  # Days for correlation analysis
}

# Visualization settings
VIZ_SETTINGS = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'save_formats': ['png', 'pdf']
}

def load_config():
    """
    Load and return the complete configuration dictionary.
    
    Returns:
        dict: Complete configuration settings
    """
    return {
        'reddit': REDDIT_CONFIG,
        'youtube': YOUTUBE_CONFIG,
        'twitter': TWITTER_CONFIG,
        'communities': {
            'reddit': REDDIT_COMMUNITIES,
            'youtube': YOUTUBE_TARGETS
        },
        'cultural_timeline': CULTURAL_TIMELINE,
        'analysis': ANALYSIS_SETTINGS,
        'visualization': VIZ_SETTINGS,
        'paths': {
            'project_root': PROJECT_ROOT,
            'data_dir': DATA_DIR,
            'raw_data_dir': RAW_DATA_DIR,
            'processed_data_dir': PROCESSED_DATA_DIR,
            'results_dir': RESULTS_DIR,
            'figures_dir': FIGURES_DIR,
            'reports_dir': REPORTS_DIR
        }
    }
