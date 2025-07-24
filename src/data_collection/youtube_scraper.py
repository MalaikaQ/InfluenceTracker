"""
YouTube data collection using YouTube Data API.
"""
import os
import pandas as pd
import datetime
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path
import sys
import requests
import json

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import YOUTUBE_CONFIG, YOUTUBE_TARGETS, RAW_DATA_DIR
from utils.text_processing import TextProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeScraper:
    """YouTube data collection using the YouTube Data API."""
    
    def __init__(self):
        """Initialize YouTube API connection."""
        self.api_key = YOUTUBE_CONFIG['api_key']
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.quota_used = 0
        self.quota_limit = YOUTUBE_CONFIG['quota_limit']
        self.text_processor = TextProcessor()
        
        if self.api_key == 'your_youtube_api_key':
            logger.warning("YouTube API key not configured. Set YOUTUBE_API_KEY environment variable.")
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make a request to YouTube API.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            API response or None if error
        """
        params['key'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Estimate quota usage (simplified)
            self.quota_used += 1
            if self.quota_used >= self.quota_limit:
                logger.warning("API quota limit reached")
                return None
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_channel_videos(self, 
                          channel_id: str,
                          max_results: int = 50,
                          published_after: str = None) -> pd.DataFrame:
        """
        Get videos from a specific channel.
        
        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos
            published_after: RFC 3339 formatted date-time string
            
        Returns:
            DataFrame with video information
        """
        videos_data = []
        page_token = None
        
        while len(videos_data) < max_results:
            params = {
                'part': 'snippet,statistics',
                'channelId': channel_id,
                'maxResults': min(50, max_results - len(videos_data)),
                'order': 'date',
                'type': 'video'
            }
            
            if published_after:
                params['publishedAfter'] = published_after
            if page_token:
                params['pageToken'] = page_token
            
            response = self._make_request('search', params)
            if not response:
                break
            
            for item in response.get('items', []):
                video_data = {
                    'video_id': item['id']['videoId'],
                    'channel_id': channel_id,
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'published_at': item['snippet']['publishedAt'],
                    'channel_title': item['snippet']['channelTitle'],
                    'thumbnail_url': item['snippet']['thumbnails']['default']['url']
                }
                videos_data.append(video_data)
            
            page_token = response.get('nextPageToken')
            if not page_token:
                break
            
            time.sleep(0.1)  # Rate limiting
        
        df = pd.DataFrame(videos_data)
        logger.info(f"Collected {len(df)} videos from channel {channel_id}")
        
        return df
    
    def get_video_comments(self, 
                          video_id: str,
                          max_results: int = 100) -> pd.DataFrame:
        """
        Get comments for a specific video.
        
        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments
            
        Returns:
            DataFrame with comment data
        """
        comments_data = []
        page_token = None
        
        while len(comments_data) < max_results:
            params = {
                'part': 'snippet,replies',
                'videoId': video_id,
                'maxResults': min(100, max_results - len(comments_data)),
                'order': 'time'
            }
            
            if page_token:
                params['pageToken'] = page_token
            
            response = self._make_request('commentThreads', params)
            if not response:
                break
            
            for item in response.get('items', []):
                top_comment = item['snippet']['topLevelComment']['snippet']
                
                comment_data = {
                    'comment_id': item['snippet']['topLevelComment']['id'],
                    'video_id': video_id,
                    'author': top_comment['authorDisplayName'],
                    'author_channel_id': top_comment.get('authorChannelId', {}).get('value', ''),
                    'text': top_comment['textDisplay'],
                    'like_count': top_comment['likeCount'],
                    'published_at': top_comment['publishedAt'],
                    'updated_at': top_comment['updatedAt'],
                    'reply_count': item['snippet']['totalReplyCount'],
                    'is_reply': False
                }
                comments_data.append(comment_data)
                
                # Get replies if they exist
                if 'replies' in item and len(comments_data) < max_results:
                    for reply in item['replies']['comments']:
                        reply_snippet = reply['snippet']
                        reply_data = {
                            'comment_id': reply['id'],
                            'video_id': video_id,
                            'author': reply_snippet['authorDisplayName'],
                            'author_channel_id': reply_snippet.get('authorChannelId', {}).get('value', ''),
                            'text': reply_snippet['textDisplay'],
                            'like_count': reply_snippet['likeCount'],
                            'published_at': reply_snippet['publishedAt'],
                            'updated_at': reply_snippet['updatedAt'],
                            'reply_count': 0,
                            'is_reply': True,
                            'parent_comment_id': item['snippet']['topLevelComment']['id']
                        }
                        comments_data.append(reply_data)
            
            page_token = response.get('nextPageToken')
            if not page_token:
                break
            
            time.sleep(0.1)
        
        df = pd.DataFrame(comments_data)
        logger.info(f"Collected {len(df)} comments from video {video_id}")
        
        return df
    
    def get_video_details(self, video_ids: List[str]) -> pd.DataFrame:
        """
        Get detailed statistics for videos.
        
        Args:
            video_ids: List of video IDs
            
        Returns:
            DataFrame with video statistics
        """
        all_details = []
        
        # Process in batches of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            params = {
                'part': 'statistics,snippet',
                'id': ','.join(batch)
            }
            
            response = self._make_request('videos', params)
            if not response:
                continue
            
            for item in response.get('items', []):
                details = {
                    'video_id': item['id'],
                    'view_count': int(item['statistics'].get('viewCount', 0)),
                    'like_count': int(item['statistics'].get('likeCount', 0)),
                    'comment_count': int(item['statistics'].get('commentCount', 0)),
                    'duration': item['snippet'].get('duration', ''),
                    'category_id': item['snippet'].get('categoryId', ''),
                    'tags': ','.join(item['snippet'].get('tags', []))
                }
                all_details.append(details)
            
            time.sleep(0.1)
        
        return pd.DataFrame(all_details)
    
    def collect_influencer_data(self, 
                               influencer_category: str,
                               videos_per_channel: int = 20,
                               comments_per_video: int = 100) -> pd.DataFrame:
        """
        Collect data from influencer channels.
        
        Args:
            influencer_category: Category from YOUTUBE_TARGETS
            videos_per_channel: Videos to collect per channel
            comments_per_video: Comments to collect per video
            
        Returns:
            Combined DataFrame with all data
        """
        if influencer_category not in YOUTUBE_TARGETS:
            logger.error(f"Category '{influencer_category}' not found")
            return pd.DataFrame()
        
        all_comments = []
        channel_ids = YOUTUBE_TARGETS[influencer_category]
        
        for channel_id in channel_ids:
            logger.info(f"Processing channel: {channel_id}")
            
            # Get videos from channel
            videos_df = self.get_channel_videos(channel_id, videos_per_channel)
            
            if videos_df.empty:
                continue
            
            # Get comments for each video
            for _, video in videos_df.iterrows():
                comments_df = self.get_video_comments(video['video_id'], comments_per_video)
                
                if not comments_df.empty:
                    # Add video metadata to comments
                    comments_df['video_title'] = video['title']
                    comments_df['channel_title'] = video['channel_title']
                    comments_df['video_published_at'] = video['published_at']
                    comments_df['influencer_category'] = influencer_category
                    all_comments.append(comments_df)
                
                time.sleep(0.5)  # Rate limiting
        
        if all_comments:
            return pd.concat(all_comments, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def search_videos(self, 
                     query: str,
                     max_results: int = 50,
                     published_after: str = None) -> pd.DataFrame:
        """
        Search for videos by query.
        
        Args:
            query: Search query
            max_results: Maximum results
            published_after: Date filter
            
        Returns:
            DataFrame with search results
        """
        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'maxResults': min(50, max_results),
            'order': 'relevance'
        }
        
        if published_after:
            params['publishedAfter'] = published_after
        
        response = self._make_request('search', params)
        if not response:
            return pd.DataFrame()
        
        videos_data = []
        for item in response.get('items', []):
            video_data = {
                'video_id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'channel_id': item['snippet']['channelId'],
                'channel_title': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'search_query': query
            }
            videos_data.append(video_data)
        
        return pd.DataFrame(videos_data)
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV."""
        if df.empty:
            logger.warning("No data to save")
            return
        
        filepath = RAW_DATA_DIR / f"{filename}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} records to {filepath}")

def main():
    """Example usage of YouTubeScraper."""
    scraper = YouTubeScraper()
    
    if scraper.api_key == 'your_youtube_api_key':
        print("Please configure YouTube API key in environment variables")
        return
    
    # Collect from red-pill influencers
    print("Collecting from red-pill influencers...")
    redpill_data = scraper.collect_influencer_data('red_pill_influencers', 
                                                   videos_per_channel=10, 
                                                   comments_per_video=50)
    if not redpill_data.empty:
        scraper.save_data(redpill_data, 'youtube_redpill_comments')
    
    # Search for Andrew Tate videos
    print("Searching for Andrew Tate videos...")
    tate_videos = scraper.search_videos('Andrew Tate', max_results=20)
    if not tate_videos.empty:
        scraper.save_data(tate_videos, 'youtube_andrew_tate_videos')
        
        # Get comments for these videos
        all_comments = []
        for _, video in tate_videos.iterrows():
            comments = scraper.get_video_comments(video['video_id'], 100)
            if not comments.empty:
                comments['video_title'] = video['title']
                comments['search_query'] = 'Andrew Tate'
                all_comments.append(comments)
        
        if all_comments:
            all_comments_df = pd.concat(all_comments, ignore_index=True)
            scraper.save_data(all_comments_df, 'youtube_andrew_tate_comments')

if __name__ == "__main__":
    main()
