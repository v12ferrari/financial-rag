"""여러 문서를 동일 질문으로 비교."""
from src.chain import ask
from src.config import DATA_DIR

DOCS = {
    "samsung": DATA_DIR / "삼성전자1분기.pdf",
    "sk_hynix": DATA_DIR / "sk하이닉스1분기.pdf",
}


def compare(question: str, heavy: bool = True) -> None:
    for name, path in DOCS.items():
        ans = ask(name, question, pdf_path=path, heavy=heavy)
        print(f"=== {name} ===")
        print(ans.text)
        pages = ", ".join(f"p.{p}" for p, _ in ans.sources)
        print(f"({pages})\n")