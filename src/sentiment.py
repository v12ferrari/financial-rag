"""뉴스 감성분석: 주가 영향도 스코어링."""
import re
from statistics import mean

from langchain_ollama import ChatOllama

from src.config import LLM_FAST, LLM_HEAVY

PROMPT = """You are a financial analyst scoring news impact on a stock price.

Score the headline's likely impact on {ticker}'s share price:
  -2 = clearly negative (losses, lawsuits, downgrades, strikes, regulatory pressure)
  -1 = mildly negative
   0 = neutral / no clear directional impact / mere reporting
  +1 = mildly positive
  +2 = clearly positive (earnings beat, major contract, upgrade)

Domain rules:
- Capacity expansion, accelerated production timelines = POSITIVE (demand signal)
- Foreign media criticism, analyst warnings without hard data = NEUTRAL (noise)
- Explanatory articles ("why X is difficult") = NEUTRAL (reporting, not news)
- Earnings beat / surprise = +2, not +1

Be strict. Most headlines are 0. Reserve +2/-2 for unambiguous cases.
Uncertainty, disputes, and internal conflict are NOT positive.

Headline: {title}
Context: {desc}

Respond with ONLY the integer. No explanation."""


def _parse(text: str) -> int:
    m = re.search(r"-?[0-2]", text)
    return int(m.group()) if m else 0


def score_news(news: list[dict], ticker: str = "Samsung Electronics",
               heavy: bool = False) -> list[dict]:
    model = LLM_HEAVY if heavy else LLM_FAST
    llm = ChatOllama(model=model, temperature=0)

    out = []
    for n in news:
        resp = llm.invoke(PROMPT.format(ticker=ticker, title=n["title"], 
                                desc=n.get("desc", "")[:300]))
        out.append({**n, "score": _parse(resp.content)})
    return out

def summarize(scored: list[dict]) -> dict:
    scores = [s["score"] for s in scored]
    return {
        "n": len(scores),
        "mean": round(mean(scores), 2) if scores else 0,
        "pos": sum(1 for s in scores if s > 0),
        "neg": sum(1 for s in scores if s < 0),
        "neu": sum(1 for s in scores if s == 0),
    }


