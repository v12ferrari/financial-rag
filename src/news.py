"""뉴스 수집."""
import requests
from bs4 import BeautifulSoup


def fetch_news(keyword: str, n: int = 10) -> list[dict]:
    """Google News RSS에서 기사 수집."""
    url = f"https://news.google.com/rss/search?q={keyword}&hl=ko&gl=KR&ceid=KR:ko"
    res = requests.get(url, timeout=10)
    soup = BeautifulSoup(res.text, "xml")

    items = []
    for it in soup.select("item")[:n]:
        items.append({
            "title": it.title.text,
            "date": it.pubDate.text if it.pubDate else "",
            "link": it.link.text if it.link else "",
            "desc": it.description.text if it.description else "",
        })
    return items