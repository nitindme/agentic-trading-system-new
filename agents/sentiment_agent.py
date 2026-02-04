"""
Sentiment Analysis Agent
Uses FinBERT and other financial sentiment models to analyze news and social media
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel


class SentimentScore(BaseModel):
    """Structured sentiment output"""
    score: float  # -1 to 1 (negative to positive)
    confidence: float  # 0 to 1
    label: str  # "positive", "negative", "neutral"
    reasoning: List[str]
    sources: List[Dict]  # List of {name, url, title, sentiment}
    timestamp: datetime


class NewsSentimentAgent:
    """
    Analyzes financial news using FinBERT
    Domain-specific sentiment model trained on financial text
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "auto"
    ):
        """
        Initialize FinBERT model
        
        Args:
            model_name: HuggingFace model identifier
            device: "cuda", "cpu", or "auto"
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading FinBERT on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Create pipeline for easier inference
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        # Label mapping
        self.label_mapping = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0
        }
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze single text for financial sentiment
        
        Args:
            text: News headline or article text
            
        Returns:
            Dict with score, label, and confidence
        """
        # Truncate if too long (FinBERT max length is 512 tokens)
        result = self.sentiment_pipeline(text[:512])[0]
        
        label = result['label'].lower()
        confidence = result['score']
        score = self.label_mapping.get(label, 0.0)
        
        return {
            "score": score,
            "confidence": confidence,
            "label": label,
            "text": text[:200],  # Store longer snippet for better context
            "full_text": text  # Store full text for display
        }
    
    def analyze_news_batch(
        self,
        news_items: List[Dict],
        weight_by_recency: bool = True
    ) -> SentimentScore:
        """
        Analyze multiple news articles and aggregate sentiment
        
        Args:
            news_items: List of dicts with 'title', 'description', 'publishedAt'
            weight_by_recency: Give more weight to recent news
            
        Returns:
            Aggregated SentimentScore
        """
        if not news_items:
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                label="neutral",
                reasoning=["No news data available for this symbol"],
                sources=["yfinance API", "Market data provider"],
                timestamp=datetime.now()
            )
        
        print(f"Analyzing {len(news_items)} news items...")
        
        sentiments = []
        weights = []
        sources = []
        
        for item in news_items:
            # Combine title and description for better context
            title = item.get('title', '')
            description = item.get('description', '')
            text = f"{title} {description}"
            
            if not text.strip():
                continue
            
            # Analyze sentiment
            result = self.analyze_text(text)
            sentiments.append(result)
            
            # Calculate recency weight (exponential decay)
            if weight_by_recency and 'publishedAt' in item:
                try:
                    pub_date = datetime.fromisoformat(item['publishedAt'].replace('Z', '+00:00'))
                    hours_ago = (datetime.now() - pub_date).total_seconds() / 3600
                    weight = np.exp(-hours_ago / 24)  # Decay with 24-hour half-life
                except:
                    weight = 1.0
            else:
                weight = 1.0
            
            weights.append(weight)
            
            # Extract source info with URL
            source_name = 'Unknown'
            source_url = item.get('link', '') or item.get('url', '')
            
            if 'publisher' in item:
                source_name = item['publisher']
            elif 'source' in item:
                source = item['source']
                if isinstance(source, dict):
                    source_name = source.get('name', 'Unknown')
                elif isinstance(source, str):
                    source_name = source
            elif source_url:
                # Extract domain from link as fallback
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(source_url).netloc
                    source_name = domain.replace('www.', '')
                except:
                    pass
            
            # Determine sentiment label for this article
            article_sentiment = "neutral"
            if result['score'] > 0.15:
                article_sentiment = "positive"
            elif result['score'] < -0.15:
                article_sentiment = "negative"
            
            sources.append({
                'name': source_name,
                'url': source_url,
                'title': title[:150] if title else 'No title',
                'sentiment': article_sentiment,
                'score': result['score']
            })
        
        if not sentiments:
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                label="neutral",
                reasoning=["No valid news text found"],
                sources=[],
                timestamp=datetime.now()
            )
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        scores = np.array([s['score'] for s in sentiments])
        confidences = np.array([s['confidence'] for s in sentiments])
        
        avg_score = np.average(scores, weights=weights)
        avg_confidence = np.average(confidences, weights=weights)
        
        # Determine label
        if avg_score > 0.15:
            label = "positive"
        elif avg_score < -0.15:
            label = "negative"
        else:
            label = "neutral"
        
        # Generate reasoning
        reasoning = self._generate_reasoning(sentiments, avg_score)
        
        # Sort sources by absolute score to get most impactful articles first
        sorted_sources = sorted(sources, key=lambda x: abs(x.get('score', 0)), reverse=True)
        
        print(f"Sentiment analysis complete: {len(sorted_sources)} sources, confidence: {avg_confidence:.2%}")
        
        return SentimentScore(
            score=float(avg_score),
            confidence=float(avg_confidence),
            label=label,
            reasoning=reasoning,
            sources=sorted_sources[:10],  # Top 10 most impactful articles
            timestamp=datetime.now()
        )
    
    def _generate_reasoning(
        self,
        sentiments: List[Dict],
        avg_score: float
    ) -> List[str]:
        """Generate human-readable reasoning"""
        reasoning = []
        
        # Count sentiment distribution
        positive = sum(1 for s in sentiments if s['score'] > 0.15)
        negative = sum(1 for s in sentiments if s['score'] < -0.15)
        neutral = len(sentiments) - positive - negative
        
        reasoning.append(
            f"Analyzed {len(sentiments)} news articles: "
            f"{positive} positive, {neutral} neutral, {negative} negative"
        )
        
        # Overall sentiment
        if avg_score > 0.3:
            reasoning.append("Strong positive sentiment in recent news")
        elif avg_score > 0.15:
            reasoning.append("Moderately positive news sentiment")
        elif avg_score < -0.3:
            reasoning.append("Strong negative sentiment in recent news")
        elif avg_score < -0.15:
            reasoning.append("Moderately negative news sentiment")
        else:
            reasoning.append("Neutral or mixed sentiment in news")
        
        # Highlight key articles with full text
        top_positive = max(sentiments, key=lambda x: x['score'], default=None)
        top_negative = min(sentiments, key=lambda x: x['score'], default=None)
        
        if top_positive and top_positive['score'] > 0.3:
            # Use full_text if available, otherwise use text
            article_text = top_positive.get('full_text', top_positive.get('text', ''))[:250]
            reasoning.append(f"📈 Most positive: \"{article_text}\"")
        
        if top_negative and top_negative['score'] < -0.3:
            article_text = top_negative.get('full_text', top_negative.get('text', ''))[:250]
            reasoning.append(f"📉 Most negative: \"{article_text}\"")
        
        # Add confidence statement
        high_conf = [s for s in sentiments if s['confidence'] > 0.9]
        if high_conf:
            reasoning.append(f"High confidence analysis on {len(high_conf)} of {len(sentiments)} articles")
        
        return reasoning


class SocialSentimentAgent:
    """
    Analyzes social media sentiment (Twitter, Reddit, StockTwits)
    Detects hype vs organic sentiment
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"):
        """Initialize social media sentiment model"""
        print(f"Loading social sentiment model...")
        
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Hype detection keywords
        self.hype_keywords = [
            "moon", "rocket", "lambo", "to the moon", "diamond hands",
            "hodl", "yolo", "apes", "squeeze", "short squeeze"
        ]
    
    def analyze_social_batch(
        self,
        posts: List[str],
        detect_hype: bool = True
    ) -> SentimentScore:
        """
        Analyze social media posts
        
        Args:
            posts: List of social media text posts
            detect_hype: Flag suspicious hype patterns
            
        Returns:
            SentimentScore with hype detection
        """
        if not posts:
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                label="neutral",
                reasoning=["No social media data available"],
                sources=["social"],
                timestamp=datetime.now()
            )
        
        sentiments = []
        hype_count = 0
        
        for post in posts:
            if not post.strip():
                continue
            
            # Analyze sentiment
            try:
                result = self.sentiment_pipeline(post[:512])[0]
                
                # Map labels
                label = result['label'].lower()
                if 'positive' in label:
                    score = 1.0
                elif 'negative' in label:
                    score = -1.0
                else:
                    score = 0.0
                
                sentiments.append({
                    'score': score,
                    'confidence': result['score']
                })
                
                # Check for hype
                if detect_hype:
                    post_lower = post.lower()
                    if any(keyword in post_lower for keyword in self.hype_keywords):
                        hype_count += 1
            
            except Exception as e:
                print(f"Error analyzing post: {e}")
                continue
        
        if not sentiments:
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                label="neutral",
                reasoning=["Could not analyze social posts"],
                sources=["social"],
                timestamp=datetime.now()
            )
        
        # Calculate averages
        avg_score = np.mean([s['score'] for s in sentiments])
        avg_confidence = np.mean([s['confidence'] for s in sentiments])
        
        # Adjust confidence based on hype detection
        hype_ratio = hype_count / len(posts) if posts else 0
        if hype_ratio > 0.3:  # More than 30% hype posts
            avg_confidence *= 0.7  # Reduce confidence
        
        # Determine label
        if avg_score > 0.2:
            label = "positive"
        elif avg_score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        
        # Generate reasoning
        reasoning = [
            f"Analyzed {len(sentiments)} social media posts",
            f"Average sentiment: {label}"
        ]
        
        if hype_ratio > 0.3:
            reasoning.append(
                f"⚠️ High hype detected ({hype_count}/{len(posts)} posts) - "
                "sentiment may be artificially inflated"
            )
        
        return SentimentScore(
            score=float(avg_score),
            confidence=float(avg_confidence),
            label=label,
            reasoning=reasoning,
            sources=["Twitter", "Reddit", "StockTwits"],
            timestamp=datetime.now()
        )


class EnsembleSentimentAgent:
    """
    Combines news and social sentiment with configurable weights
    """
    
    def __init__(
        self,
        news_weight: float = 0.6,
        social_weight: float = 0.4
    ):
        """
        Initialize ensemble agent
        
        Args:
            news_weight: Weight for news sentiment (default 0.6)
            social_weight: Weight for social sentiment (default 0.4)
        """
        self.news_agent = NewsSentimentAgent()
        self.social_agent = SocialSentimentAgent()
        
        self.news_weight = news_weight
        self.social_weight = social_weight
        
        # Normalize weights
        total = news_weight + social_weight
        self.news_weight /= total
        self.social_weight /= total
    
    def analyze_combined(
        self,
        news_items: List[Dict],
        social_posts: List[str]
    ) -> SentimentScore:
        """
        Combine news and social sentiment
        
        Args:
            news_items: List of news articles
            social_posts: List of social media posts
            
        Returns:
            Combined SentimentScore
        """
        # Analyze both sources
        news_sentiment = self.news_agent.analyze_news_batch(news_items)
        social_sentiment = self.social_agent.analyze_social_batch(social_posts)
        
        # Weighted combination
        combined_score = (
            news_sentiment.score * self.news_weight +
            social_sentiment.score * self.social_weight
        )
        
        combined_confidence = (
            news_sentiment.confidence * self.news_weight +
            social_sentiment.confidence * self.social_weight
        )
        
        # Determine label
        if combined_score > 0.2:
            label = "positive"
        elif combined_score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        
        # Combine reasoning
        reasoning = [
            f"Combined sentiment analysis (News: {self.news_weight:.1%}, Social: {self.social_weight:.1%})",
            f"News sentiment: {news_sentiment.label} ({news_sentiment.score:.2f})",
            f"Social sentiment: {social_sentiment.label} ({social_sentiment.score:.2f})",
        ]
        reasoning.extend(news_sentiment.reasoning[:2])
        reasoning.extend(social_sentiment.reasoning[:2])
        
        return SentimentScore(
            score=float(combined_score),
            confidence=float(combined_confidence),
            label=label,
            reasoning=reasoning,
            sources=news_sentiment.sources + social_sentiment.sources,
            timestamp=datetime.now()
        )


# Example usage
if __name__ == "__main__":
    # Test news sentiment
    news_agent = NewsSentimentAgent()
    
    sample_news = [
        {
            "title": "Apple Reports Record Q4 Earnings, Beats Expectations",
            "description": "Tech giant exceeds analyst predictions with strong iPhone sales",
            "publishedAt": "2024-01-20T10:00:00Z",
            "source": {"name": "Bloomberg"}
        },
        {
            "title": "Supply Chain Concerns Mount for Apple Suppliers",
            "description": "Analysts warn of potential disruptions in Q1 2024",
            "publishedAt": "2024-01-21T14:30:00Z",
            "source": {"name": "Reuters"}
        }
    ]
    
    result = news_agent.analyze_news_batch(sample_news)
    print("\n=== News Sentiment Analysis ===")
    print(f"Score: {result.score:.3f}")
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Sources: {', '.join(result.sources)}")
    print("\nReasoning:")
    for reason in result.reasoning:
        print(f"  • {reason}")
