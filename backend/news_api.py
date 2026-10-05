import requests
from configs.logging_config import logger

class NewsAPIClient:
    """Fetches live news headlines via NewsAPI with automatic fallback mock articles."""

    def __init__(self, api_key: str = "", endpoint: str = "https://newsapi.org/v2/top-headlines"):
        self.api_key = api_key
        self.endpoint = endpoint

    def fetch_top_headlines(self, query: str = "technology", country: str = "us") -> list:
        """Fetches live headlines or returns fallback mock news list."""
        if self.api_key:
            try:
                params = {
                    "q": query,
                    "country": country,
                    "apiKey": self.api_key,
                    "pageSize": 10
                }
                res = requests.get(self.endpoint, params=params, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    articles = data.get("articles", [])
                    logger.info(f"Fetched {len(articles)} live articles from NewsAPI.")
                    return [
                        {
                            "title": a.get("title", ""),
                            "description": a.get("description", ""),
                            "content": a.get("content", a.get("description", "")),
                            "url": a.get("url", ""),
                            "urlToImage": a.get("urlToImage", ""),
                            "publishedAt": a.get("publishedAt", "")
                        } for a in articles
                    ]
            except Exception as e:
                logger.warning(f"NewsAPI request failed: {e}. Falling back to demo articles.")

        # Fallback: no API key configured, or the live request failed.
        # Reuses the project's single shared demo corpus (datasets/demo_data.py)
        # instead of maintaining a second, separate hardcoded article list.
        from datasets.demo_data import DEMO_ARTICLES, generate_demo_images

        img_map = generate_demo_images()
        fallback = []
        for a in DEMO_ARTICLES[:6]:
            image_key = "doctored" if a["label"] == 1 else "authentic"
            fallback.append({
                "title": a["title"],
                "description": a["text"][:140],
                "content": a["text"],
                "url": f"https://example.com/news/{a['id']}",
                "urlToImage": img_map[image_key],
                "publishedAt": f"{a['published_date']}T00:00:00Z",
            })
        return fallback
