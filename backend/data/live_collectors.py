"""
Live API collectors for Tamil Nadu election social and sentiment data.

Optional and safe-by-default:
- If API keys are missing or API calls fail, callers can fallback to mock
  generators. Live collectors return empty DataFrames rather than raising.
- Party labels align with config.PARTIES: DMK_ALLIANCE, AIADMK_NDA, TVK, OTHERS.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from config import PARTIES, SENTIMENT_KEYWORDS
from .sentiment_extractor import SentimentExtractor


@dataclass
class ApiContext:
    x_bearer_token: str = ""
    youtube_api_key: str = ""
    news_api_key: str = ""


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE entries from an env file into process env if unset."""
    import os

    if not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_context() -> ApiContext:
    import os

    return ApiContext(
        x_bearer_token=os.getenv("X_BEARER_TOKEN", "").strip(),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", "").strip(),
        news_api_key=os.getenv("NEWS_API_KEY", "").strip(),
    )


def _party_queries() -> Dict[str, str]:
    queries: Dict[str, str] = {}
    # OTHERS is skipped for live queries: it's a residual bucket and direct
    # search would collect too much unrelated noise (independents, minor
    # parties each need their own narrow query if enriched later).
    for party in ["DMK_ALLIANCE", "AIADMK_NDA", "TVK", "NTK"]:
        keywords = SENTIMENT_KEYWORDS.get(party, [])[:5]
        quoted = [f'"{kw}"' for kw in keywords if kw]
        queries[party] = " OR ".join(quoted) if quoted else party
    return queries


def _score_text(sentiment_extractor: SentimentExtractor, text: str) -> float:
    if not text:
        return 0.0
    return float(sentiment_extractor.analyze_text(text))


def _safe_get(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Optional[dict]:
    try:
        res = requests.get(url, headers=headers, params=params, timeout=timeout)
        if res.status_code != 200:
            return None
        return res.json()
    except Exception:
        return None


def fetch_news_records(
    news_api_key: str,
    sentiment_extractor: SentimentExtractor,
    *,
    from_date: str,
    to_date: Optional[str] = None,
    page_size: int = 50,
) -> List[dict]:
    if not news_api_key:
        return []

    records: List[dict] = []
    endpoint = "https://newsapi.org/v2/everything"
    today = datetime.utcnow().date().isoformat()
    queries = _party_queries()

    for party, query in queries.items():
        payload = _safe_get(
            endpoint,
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "from": from_date,
                "to": to_date or today,
                "pageSize": page_size,
                "apiKey": news_api_key,
            },
        )
        if not payload:
            continue

        for article in payload.get("articles", []):
            title = article.get("title") or ""
            desc = article.get("description") or ""
            text = f"{title}. {desc}".strip()
            published = article.get("publishedAt", "")[:10]
            date_month = published[:7] if len(published) >= 7 else today[:7]
            source_name = (article.get("source") or {}).get("name") or "News"
            records.append(
                {
                    "date_month": date_month,
                    "platform": "News",
                    "party": party,
                    "identifier": source_name,
                    "engagement_volume": 1,
                    "sentiment_score": _score_text(sentiment_extractor, text),
                }
            )

    return records


def fetch_youtube_records(
    youtube_api_key: str,
    sentiment_extractor: SentimentExtractor,
    *,
    max_results: int = 30,
) -> List[dict]:
    if not youtube_api_key:
        return []

    records: List[dict] = []
    search_url = "https://www.googleapis.com/youtube/v3/search"
    videos_url = "https://www.googleapis.com/youtube/v3/videos"
    queries = _party_queries()

    for party, query in queries.items():
        search_payload = _safe_get(
            search_url,
            params={
                "key": youtube_api_key,
                "part": "snippet",
                "q": f"Tamil Nadu election {query}",
                "type": "video",
                "maxResults": max_results,
                "order": "date",
            },
        )
        if not search_payload:
            continue

        ids = [((item.get("id") or {}).get("videoId")) for item in search_payload.get("items", [])]
        ids = [video_id for video_id in ids if video_id]
        if not ids:
            continue

        stats_payload = _safe_get(
            videos_url,
            params={
                "key": youtube_api_key,
                "part": "statistics,snippet",
                "id": ",".join(ids),
            },
        )
        if not stats_payload:
            continue

        for item in stats_payload.get("items", []):
            snippet = item.get("snippet") or {}
            stats = item.get("statistics") or {}
            title = snippet.get("title") or ""
            published = (snippet.get("publishedAt") or "")[:10]
            date_month = published[:7] if len(published) >= 7 else datetime.utcnow().date().isoformat()[:7]
            views = int(stats.get("viewCount", 0) or 0)
            likes = int(stats.get("likeCount", 0) or 0)
            comments = int(stats.get("commentCount", 0) or 0)
            engagement = views + likes * 3 + comments * 5

            records.append(
                {
                    "date_month": date_month,
                    "platform": "YouTube",
                    "party": party,
                    "identifier": snippet.get("channelTitle") or "YouTube",
                    "engagement_volume": int(engagement),
                    "sentiment_score": _score_text(sentiment_extractor, title),
                }
            )

    return records


def fetch_x_records(
    x_bearer_token: str,
    sentiment_extractor: SentimentExtractor,
    *,
    max_results: int = 100,
) -> List[dict]:
    if not x_bearer_token:
        return []

    records: List[dict] = []
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {x_bearer_token}"}
    queries = _party_queries()

    for party, query in queries.items():
        payload = _safe_get(
            url,
            headers=headers,
            params={
                "query": f"({query}) (lang:en OR lang:ta) -is:retweet",
                "max_results": min(max_results, 100),
                "tweet.fields": "created_at,public_metrics,text",
            },
        )
        if not payload:
            continue

        for tweet in payload.get("data", []):
            metrics = tweet.get("public_metrics") or {}
            likes = int(metrics.get("like_count", 0) or 0)
            replies = int(metrics.get("reply_count", 0) or 0)
            reposts = int(metrics.get("retweet_count", 0) or 0)
            quotes = int(metrics.get("quote_count", 0) or 0)
            engagement = likes + replies * 3 + reposts * 4 + quotes * 2

            created = (tweet.get("created_at") or "")[:10]
            date_month = created[:7] if len(created) >= 7 else datetime.utcnow().date().isoformat()[:7]
            text = tweet.get("text") or ""

            records.append(
                {
                    "date_month": date_month,
                    "platform": "Twitter/X",
                    "party": party,
                    "identifier": "X search",
                    "engagement_volume": int(engagement),
                    "sentiment_score": _score_text(sentiment_extractor, text),
                }
            )

    return records


def create_social_media_details_live(
    sentiment_extractor: SentimentExtractor,
    *,
    from_date: str = "2025-01-01",
    to_date: Optional[str] = None,
) -> pd.DataFrame:
    """Build live social-media detail rows for Tamil Nadu election sentiment."""
    ctx = get_api_context()

    rows: List[dict] = []
    rows.extend(fetch_x_records(ctx.x_bearer_token, sentiment_extractor))
    rows.extend(fetch_youtube_records(ctx.youtube_api_key, sentiment_extractor))
    rows.extend(
        fetch_news_records(ctx.news_api_key, sentiment_extractor, from_date=from_date, to_date=to_date)
    )

    if not rows:
        return pd.DataFrame(columns=[
            "date_month", "platform", "party", "identifier",
            "engagement_volume", "sentiment_score",
        ])

    df = pd.DataFrame(rows)
    df["engagement_volume"] = df["engagement_volume"].astype(int)
    df["sentiment_score"] = df["sentiment_score"].astype(float)
    return df


def _percentage_split(scores: Iterable[float]) -> tuple[int, int, int]:
    scores_list = list(scores)
    if not scores_list:
        return 0, 0, 100
    total = len(scores_list)
    pos = sum(1 for s in scores_list if s > 0.1)
    neg = sum(1 for s in scores_list if s < -0.1)
    neu = total - pos - neg
    return (
        int(round((pos / total) * 100)),
        int(round((neg / total) * 100)),
        int(round((neu / total) * 100)),
    )


# Tamil Nadu 2026 issue field schema. Scores in [-1, 1], positive = helps
# the alliance. These are rough priors used when live data is absent.
_TN_ISSUE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "DMK_ALLIANCE": {
        "governance_score": 0.55,
        "welfare_scheme_score": 0.65,          # Kalaignar Magalir Urimai Thogai, etc.
        "anti_incumbency_score": -0.30,
        "ls2024_momentum": 0.40,
        "youth_engagement_score": 0.35,
        "neet_language_stance_score": 0.50,    # anti-NEET, anti-Hindi imposition
        "katchatheevu_fishermen_score": 0.05,
        "corruption_perception_score": -0.20,
        "ai_campaign_score": 0.35,
        "ground_campaign_score": 0.70,
        "celebrity_endorsement": 0.20,
        "poll_seats_low": 130, "poll_seats_mid": 155, "poll_seats_high": 175,
        "poll_vote_share": 42.0,
    },
    "AIADMK_NDA": {
        "governance_score": 0.30,
        "welfare_scheme_score": 0.40,
        "anti_incumbency_score": 0.25,
        "ls2024_momentum": -0.15,
        "youth_engagement_score": 0.15,
        "neet_language_stance_score": -0.10,
        "katchatheevu_fishermen_score": 0.20,
        "corruption_perception_score": 0.05,
        "ai_campaign_score": 0.25,
        "ground_campaign_score": 0.55,
        "celebrity_endorsement": 0.10,
        "poll_seats_low": 45, "poll_seats_mid": 65, "poll_seats_high": 85,
        "poll_vote_share": 30.0,
    },
    "TVK": {
        "governance_score": 0.00,
        "welfare_scheme_score": 0.00,
        "anti_incumbency_score": 0.15,
        "ls2024_momentum": 0.00,
        "youth_engagement_score": 0.60,        # Vijay's core strength
        "neet_language_stance_score": 0.20,
        "katchatheevu_fishermen_score": 0.10,
        "corruption_perception_score": 0.25,   # "clean slate" perception
        "ai_campaign_score": 0.40,
        "ground_campaign_score": 0.35,
        "celebrity_endorsement": 0.70,
        "poll_seats_low": 0, "poll_seats_mid": 8, "poll_seats_high": 20,
        "poll_vote_share": 10.0,
    },
    "NTK": {
        "governance_score": 0.00,
        "welfare_scheme_score": 0.00,
        "anti_incumbency_score": 0.20,
        "ls2024_momentum": -0.02,
        "youth_engagement_score": 0.45,
        "neet_language_stance_score": 0.55,   # strong Tamil-nationalist stance
        "katchatheevu_fishermen_score": 0.40, # fishermen rights central plank
        "corruption_perception_score": 0.30,  # "clean" reputation
        "ai_campaign_score": 0.10,
        "ground_campaign_score": 0.45,
        "celebrity_endorsement": 0.20,
        "poll_seats_low": 0, "poll_seats_mid": 0, "poll_seats_high": 2,
        "poll_vote_share": 4.0,
    },
    "OTHERS": {
        "governance_score": 0.00,
        "welfare_scheme_score": 0.00,
        "anti_incumbency_score": 0.05,
        "ls2024_momentum": -0.05,
        "youth_engagement_score": 0.10,
        "neet_language_stance_score": 0.10,
        "katchatheevu_fishermen_score": 0.00,
        "corruption_perception_score": 0.05,
        "ai_campaign_score": 0.05,
        "ground_campaign_score": 0.20,
        "celebrity_endorsement": 0.05,
        "poll_seats_low": 0, "poll_seats_mid": 2, "poll_seats_high": 6,
        "poll_vote_share": 4.0,
    },
}


def create_sentiment_data_live(social_df: pd.DataFrame) -> pd.DataFrame:
    """Per-alliance sentiment + issue summary for Tamil Nadu 2026."""
    rows: List[dict] = []
    for party in PARTIES:
        part_df = social_df[social_df["party"] == party] if not social_df.empty else pd.DataFrame()

        news_df = part_df[part_df["platform"] == "News"] if not part_df.empty else pd.DataFrame()
        x_df = part_df[part_df["platform"] == "Twitter/X"] if not part_df.empty else pd.DataFrame()
        yt_df = part_df[part_df["platform"] == "YouTube"] if not part_df.empty else pd.DataFrame()

        sentiment_values = part_df["sentiment_score"].tolist() if not part_df.empty else []
        pos_pct, neg_pct, neu_pct = _percentage_split(sentiment_values)

        defaults = _TN_ISSUE_DEFAULTS.get(party, _TN_ISSUE_DEFAULTS["OTHERS"])

        row = {
            "party": party,
            "news_sentiment": float(news_df["sentiment_score"].mean()) if not news_df.empty else 0.0,
            "twitter_mentions": int(len(x_df)),
            "facebook_engagement": 0,
            "instagram_posts": 0,
            "linkedin_articles": int(len(news_df)),
            "youtube_views": int(yt_df["engagement_volume"].sum()) if not yt_df.empty else 0,
            "positive_mentions_pct": pos_pct,
            "negative_mentions_pct": neg_pct,
            "neutral_mentions_pct": neu_pct,
            "final_sentiment_score": float(part_df["sentiment_score"].mean()) if not part_df.empty else 0.0,
            **defaults,
        }
        rows.append(row)

    return pd.DataFrame(rows)
