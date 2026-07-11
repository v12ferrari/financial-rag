"""RAG 체인: 검색 → 프롬프트 → LLM → 답변(+출처)."""
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config import LLM_FAST, LLM_HEAVY, TOP_K
from src.vectorstore import get_store

PROMPT = ChatPromptTemplate.from_template(
    """You are a financial analyst. Answer using ONLY the context below.
If the answer is not in the context, say exactly: "Not found in the document."
Be concise. Cite specific numbers when present.

Context:
{context}

Question: {question}

Answer:"""
)


@dataclass
class Answer:
    text: str
    sources: list  # [(page, snippet), ...]


def get_llm(heavy: bool = True) -> ChatOllama:
    return ChatOllama(model=LLM_HEAVY if heavy else LLM_FAST, temperature=0)


def ask(collection: str, question: str, pdf_path=None, heavy: bool = True, k: int = TOP_K) -> Answer:
    if pdf_path is not None:
        from src.retriever import build_hybrid
        retriever = build_hybrid(collection, pdf_path, k=k)
        docs = retriever.invoke(question)
    else:
        docs = get_store(collection).similarity_search(question, k=k)

    if not docs:
        return Answer("No documents retrieved.", [])

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    llm = get_llm(heavy)
    resp = llm.invoke(PROMPT.format(context=context, question=question))

    sources = [(d.metadata.get("page", "?"), d.page_content[:100]) for d in docs[:3]]
    return Answer(resp.content, sources)


def show(ans: Answer) -> None:
    print(ans.text)
    print("\n" + "-" * 50)
    for page, snippet in ans.sources:
        print(f"[p.{page}] {snippet.strip()[:80]}...")