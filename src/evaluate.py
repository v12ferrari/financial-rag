"""감성분석 정확도 평가."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.eval_set import EVAL_SET
from src.sentiment import score_news


def evaluate(heavy: bool = False, verbose: bool = True) -> dict:
    news = [{"title": t} for t, _ in EVAL_SET]
    gold = [g for _, g in EVAL_SET]

    pred = [s["score"] for s in score_news(news, heavy=heavy)]

    exact = sum(p == g for p, g in zip(pred, gold))
    direction = sum(
        (p > 0 and g > 0) or (p < 0 and g < 0) or (p == 0 and g == 0)
        for p, g in zip(pred, gold)
    )
    mae = sum(abs(p - g) for p, g in zip(pred, gold)) / len(gold)

    if verbose:
        for (title, g), p in zip(EVAL_SET, pred):
            mark = "✅" if p == g else "❌"
            print(f"{mark} gold:{g:+d} pred:{p:+d} | {title[:45]}")

    return {
        "model": "70B" if heavy else "8B",
        "n": len(gold),
        "exact": f"{exact}/{len(gold)} ({exact/len(gold)*100:.0f}%)",
        "direction": f"{direction}/{len(gold)} ({direction/len(gold)*100:.0f}%)",
        "mae": round(mae, 2),
    }