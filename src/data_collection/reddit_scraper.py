"""
Reddit data collection using PRAW (Python Reddit API Wrapper).
"""
import praw
import pandas as pd
import datetime
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import REDDIT_CONFIG, REDDIT_COMMUNITIES, RAW_DATA_DIR
from utils.text_processing import TextProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditScraper:
    """Reddit data collection and processing."""
    
    def __init__(self):
        """Initialize Reddit API connection."""
        try:
            self.reddit = praw.Reddit(
                client_id=REDDIT_CONFIG['client_id'],
                client_secret=REDDIT_CONFIG['client_secret'],
                user_agent=REDDIT_CONFIG['user_agent'],
                username=REDDIT_CONFIG['username'],
                password=REDDIT_CONFIG['password']
            )
            logger.info("Reddit API connection established")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API: {e}")
            self.reddit = None
            
        self.text_processor = TextProcessor()
    
    def collect_subreddit_comments(self, 
                                 subreddit_name: str,
                                 limit: int = 1000,
                                 time_filter: str = 'all',
                                 sort_by: str = 'hot') -> pd.DataFrame:
        """
        Collect comments from a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            limit: Maximum number of submissions to process
            time_filter: Time filter ('all', 'year', 'month', 'week', 'day')
            sort_by: Sort method ('hot', 'new', 'top', 'rising')
            
        Returns:
            DataFrame with comment data
        """
        if not self.reddit:
            logger.error("Reddit API not initialized")
            return pd.DataFrame()
        
        comments_data = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            logger.info(f"Collecting comments from r/{subreddit_name}")
            
            # Get submissions based on sort method
            if sort_by == 'hot':
                submissions = subreddit.hot(limit=limit)
            elif sort_by == 'new':
                submissions = subreddit.new(limit=limit)
            elif sort_by == 'top':
                submissions = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort_by == 'rising':
                submissions = subreddit.rising(limit=limit)
            else:
                submissions = subreddit.hot(limit=limit)
            
            for submission in submissions:
                # Expand all comments
                submission.comments.replace_more(limit=None)
                
                for comment in submission.comments.list():
                    if comment.body in ['[deleted]', '[removed]']:
                        continue
                        
                    comment_data = {
                        'comment_id': comment.id,
                        'submission_id': submission.id,
                        'subreddit': subreddit_name,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'body': comment.body,
                        'score': comment.score,
                        'created_utc': datetime.datetime.fromtimestamp(comment.created_utc),
                        'is_submitter': comment.is_submitter,
                        'submission_title': submission.title,
                        'submission_score': submission.score,
                        'submission_created_utc': datetime.datetime.fromtimestamp(submission.created_utc),
                        'num_comments': submission.num_comments
                    }
                    comments_data.append(comment_data)
                
                # Rate limiting
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error collecting from r/{subreddit_name}: {e}")
        
        df = pd.DataFrame(comments_data)
        logger.info(f"Collected {len(df)} comments from r/{subreddit_name}")
        
        return df
    
    def collect_multiple_subreddits(self, 
                                  subreddit_list: List[str],
                                  limit_per_sub: int = 500) -> pd.DataFrame:
        """
        Collect comments from multiple subreddits.
        
        Args:
            subreddit_list: List of subreddit names
            limit_per_sub: Limit per subreddit
            
        Returns:
            Combined DataFrame
        """
        all_comments = []
        
        for subreddit in subreddit_list:
            df = self.collect_subreddit_comments(subreddit, limit=limit_per_sub)
            all_comments.append(df)
            
            # Rate limiting between subreddits
            time.sleep(1)
        
        if all_comments:
            return pd.concat(all_comments, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def collect_by_category(self, 
                           category: str,
                           limit_per_sub: int = 500) -> pd.DataFrame:
        """
        Collect comments from subreddits in a specific category.
        
        Args:
            category: Category name from REDDIT_COMMUNITIES
            limit_per_sub: Limit per subreddit
            
        Returns:
            DataFrame with category label
        """
        if category not in REDDIT_COMMUNITIES:
            logger.error(f"Category '{category}' not found in configuration")
            return pd.DataFrame()
        
        subreddits = REDDIT_COMMUNITIES[category]
        df = self.collect_multiple_subreddits(subreddits, limit_per_sub)
        
        if not df.empty:
            df['category'] = category
        
        return df
    
    def search_comments(self, 
                       query: str,
                       subreddit_name: str = None,
                       limit: int = 1000,
                       sort: str = 'relevance') -> pd.DataFrame:
        """
        Search for specific comments using Reddit's search.
        
        Args:
            query: Search query
            subreddit_name: Specific subreddit to search (None for all)
            limit: Maximum results
            sort: Sort method ('relevance', 'hot', 'top', 'new', 'comments')
            
        Returns:
            DataFrame with search results
        """
        if not self.reddit:
            logger.error("Reddit API not initialized")
            return pd.DataFrame()
        
        comments_data = []
        
        try:
            if subreddit_name:
                subreddit = self.reddit.subreddit(subreddit_name)
            else:
                subreddit = self.reddit.subreddit('all')
            
            # Search submissions
            for submission in subreddit.search(query, sort=sort, limit=limit):
                submission.comments.replace_more(limit=0)  # Limited expansion for search
                
                for comment in submission.comments.list()[:50]:  # Limit comments per post
                    if query.lower() in comment.body.lower():
                        comment_data = {
                            'comment_id': comment.id,
                            'submission_id': submission.id,
                            'subreddit': submission.subreddit.display_name,
                            'author': str(comment.author) if comment.author else '[deleted]',
                            'body': comment.body,
                            'score': comment.score,
                            'created_utc': datetime.datetime.fromtimestamp(comment.created_utc),
                            'search_query': query,
                            'submission_title': submission.title
                        }
                        comments_data.append(comment_data)
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
        
        return pd.DataFrame(comments_data)
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV."""
        if df.empty:
            logger.warning("No data to save")
            return
        
        filepath = RAW_DATA_DIR / f"{filename}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} records to {filepath}")
    
    def collect_time_series_data(self, 
                                subreddit_name: str,
                                start_date: datetime.datetime,
                                end_date: datetime.datetime,
                                interval_days: int = 7) -> pd.DataFrame:
        """
        Collect data in time intervals for time series analysis.
        
        Args:
            subreddit_name: Subreddit name
            start_date: Start date
            end_date: End date
            interval_days: Days per interval
            
        Returns:
            DataFrame with time series data
        """
        all_data = []
        current_date = start_date
        
        while current_date < end_date:
            next_date = current_date + datetime.timedelta(days=interval_days)
            
            # Use pushshift API or similar for historical data
            # For now, collect recent data as example
            df = self.collect_subreddit_comments(subreddit_name, limit=200)
            
            if not df.empty:
                # Filter by date range (this is simplified)
                df['collection_period_start'] = current_date
                df['collection_period_end'] = next_date
                all_data.append(df)
            
            current_date = next_date
            time.sleep(2)  # Longer delay for time series collection
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

def main():
    """Example usage of RedditScraper."""
    scraper = RedditScraper()
    
    # Collect from red-pill communities
    print("Collecting from red-pill communities...")
    redpill_data = scraper.collect_by_category('mens_rights', limit_per_sub=100)
    if not redpill_data.empty:
        scraper.save_data(redpill_data, 'reddit_redpill_sample')
    
    # Collect from feminist communities for comparison
    print("Collecting from feminist communities...")
    feminist_data = scraper.collect_by_category('feminist', limit_per_sub=100)
    if not feminist_data.empty:
        scraper.save_data(feminist_data, 'reddit_feminist_sample')
    
    # Search for specific terms
    print("Searching for Andrew Tate mentions...")
    tate_data = scraper.search_comments('Andrew Tate', limit=100)
    if not tate_data.empty:
        scraper.save_data(tate_data, 'reddit_andrew_tate_mentions')

if __name__ == "__main__":
    main()
