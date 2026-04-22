"""
Social Media Sentiment Extraction Module
Extracts and analyzes sentiment from social media for election prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import re


@dataclass
class SentimentData:
    """Container for sentiment data"""
    booth_id: str
    ward_id: str
    district: str
    party_sentiments: Dict[str, float]  # party -> sentiment score (-1 to 1)
    party_mentions: Dict[str, int]      # party -> mention count
    overall_sentiment: float
    timestamp: datetime
    source: str  # twitter, facebook, news


class SentimentExtractor:
    """
    Extract sentiment from social media sources.
    In production, this would connect to Twitter API, Facebook Graph API, etc.
    """
    
    def __init__(self, config):
        self.config = config
        self.sentiment_model = None
        self._load_sentiment_model()
    
    def _load_sentiment_model(self):
        """Load pretrained sentiment analysis model"""
        try:
            from transformers import pipeline
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Warning: Could not load sentiment model: {e}")
            print("Using rule-based sentiment analysis as fallback")
            self.sentiment_model = None
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of a single text.
        Returns score from -1 (negative) to 1 (positive)
        """
        if self.sentiment_model:
            try:
                result = self.sentiment_model(text[:512])[0]
                # Convert 1-5 star rating to -1 to 1 scale
                label = result['label']
                if 'star' in label.lower():
                    stars = int(label[0])
                    return (stars - 3) / 2  # Map 1-5 to -1 to 1
                else:
                    # POSITIVE/NEGATIVE model
                    score = result['score']
                    return score if result['label'] == 'POSITIVE' else -score
            except:
                pass
        
        # Fallback: Simple keyword-based sentiment
        return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> float:
        """Simple rule-based sentiment analysis"""
        positive_words = [
            'good', 'great', 'excellent', 'win', 'victory', 'support',
            'development', 'progress', 'success', 'popular', 'best',
            # Tamil positive sentiment
            'நல்ல', 'வெற்றி', 'வளர்ச்சி', 'முன்னேற்றம்', 'சிறந்த', 'ஆதரவு',
        ]
        negative_words = [
            'bad', 'worst', 'fail', 'corrupt', 'scam', 'loss',
            'against', 'protest', 'scandal', 'poor', 'terrible',
            # Tamil negative sentiment
            'மோசம்', 'தோல்வி', 'ஊழல்', 'மோசடி', 'எதிர்ப்பு', 'போராட்டம்',
        ]
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def extract_party_mentions(self, text: str) -> Dict[str, bool]:
        """Extract which parties are mentioned in text"""
        from config import SENTIMENT_KEYWORDS
        
        mentions = {}
        text_lower = text.lower()
        
        for party, keywords in SENTIMENT_KEYWORDS.items():
            if party == 'general':
                continue
            mentions[party] = any(kw.lower() in text_lower for kw in keywords)
        
        return mentions
    
    def aggregate_sentiment_by_region(
        self, 
        sentiments: List[SentimentData]
    ) -> pd.DataFrame:
        """Aggregate sentiment data by booth/ward"""
        records = []
        for s in sentiments:
            record = {
                'booth_id': s.booth_id,
                'ward_id': s.ward_id,
                'district': s.district,
                'overall_sentiment': s.overall_sentiment,
                'source': s.source,
                'timestamp': s.timestamp
            }
            for party, score in s.party_sentiments.items():
                record[f'{party}_sentiment'] = score
            for party, count in s.party_mentions.items():
                record[f'{party}_mentions'] = count
            records.append(record)
        
        return pd.DataFrame(records)


class MockSentimentGenerator:
    """
    Generate mock sentiment data for demonstration.
    In production, replace with actual API calls.
    """
    
    def __init__(self, config):
        self.config = config
        self.parties = config.parties
        self.districts = config.districts
    
    def generate_booth_sentiments(
        self,
        num_booths: int = 1000,
        days_back: int = 30
    ) -> np.ndarray:
        """
        Generate mock sentiment features for booths.
        Returns array of shape (num_booths, num_sentiment_features)
        
        Features per booth:
        - Per party (4 parties):
            - avg_sentiment (-1 to 1)
            - sentiment_std
            - mention_count (normalized)
            - sentiment_trend (change over time)
        - Overall:
            - total_engagement (normalized)
            - sentiment_volatility
        Total: 4*4 + 2 = 18 features
        """
        np.random.seed(42)
        
        num_features = len(self.parties) * 4 + 2  # 18 features
        sentiments = np.zeros((num_booths, num_features))
        
        for i in range(num_booths):
            # Simulate regional bias (some booths favor certain parties)
            base_bias = np.random.dirichlet(np.ones(len(self.parties)))
            
            idx = 0
            for j, party in enumerate(self.parties):
                # Average sentiment for this party in this booth
                # Biased towards the booth's leaning
                base_sentiment = (base_bias[j] - 0.25) * 2  # -0.5 to 1.5, centered around bias
                avg_sentiment = np.clip(
                    base_sentiment + np.random.normal(0, 0.2), -1, 1
                )
                sentiments[i, idx] = avg_sentiment
                idx += 1
                
                # Sentiment standard deviation
                sentiments[i, idx] = np.random.uniform(0.1, 0.5)
                idx += 1
                
                # Normalized mention count (higher for stronger presence)
                mention_base = base_bias[j] * 100
                sentiments[i, idx] = np.clip(
                    (mention_base + np.random.exponential(20)) / 200, 0, 1
                )
                idx += 1
                
                # Sentiment trend (positive = improving sentiment)
                sentiments[i, idx] = np.random.normal(0, 0.1)
                idx += 1
            
            # Total engagement (normalized 0-1)
            sentiments[i, idx] = np.random.beta(2, 5)
            idx += 1
            
            # Sentiment volatility
            sentiments[i, idx] = np.random.uniform(0.05, 0.3)
        
        return sentiments.astype(np.float32)
    
    def generate_labels(
        self,
        num_booths: int = 1000,
        sentiment_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate mock election outcome labels.
        Labels are influenced by sentiment if provided.
        Returns array of shape (num_booths,) with party indices
        """
        np.random.seed(43)
        
        labels = np.zeros(num_booths, dtype=np.int64)
        
        for i in range(num_booths):
            if sentiment_features is not None:
                # Use sentiment to influence labels
                # Extract average sentiment per party (indices 0, 4, 8, 12)
                party_sentiments = sentiment_features[i, [0, 4, 8, 12]]
                
                # Convert to probabilities with softmax-like function
                exp_sentiments = np.exp(party_sentiments * 2)
                probs = exp_sentiments / exp_sentiments.sum()
                
                # Add some noise
                probs = probs * 0.7 + np.random.dirichlet(np.ones(4)) * 0.3
                probs = probs / probs.sum()
            else:
                # Tamil Nadu prior: DMK alliance leads, AIADMK NDA second,
                # TVK a distant third, OTHERS/NTK/independents minimal.
                probs = np.array([0.48, 0.34, 0.12, 0.06])
                probs = probs + np.random.normal(0, 0.05, 4)
                probs = np.clip(probs, 0.01, 1)
                probs = probs / probs.sum()
            
            labels[i] = np.random.choice(len(self.parties), p=probs)
        
        return labels


def get_sentiment_feature_names() -> List[str]:
    """Get names of sentiment features"""
    from config import PARTIES

    feature_names = []
    for party in PARTIES:
        feature_names.extend([
            f"{party}_avg_sentiment",
            f"{party}_sentiment_std",
            f"{party}_mention_count",
            f"{party}_sentiment_trend",
        ])
    feature_names.extend(["total_engagement", "sentiment_volatility"])
    return feature_names
