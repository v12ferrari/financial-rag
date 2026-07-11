"""하이브리드 검색: BM25(키워드) + 벡터(의미) 앙상블."""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever



from src.config import TOP_K
from src.vectorstore import get_store, load_and_split


def build_hybrid(collection: str, pdf_path, k: int = TOP_K):
    """BM25 + 벡터 검색을 결합한 retriever 반환."""
    chunks = load_and_split(pdf_path)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    vector = get_store(collection).as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.5, 0.5],
    )


"""하이브리드 검색: BM25(키워드) + 벡터(의미) 앙상블."""
from functools import lru_cache

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from src.config import TOP_K
from src.vectorstore import get_store, load_and_split


@lru_cache(maxsize=8)
def _cached_chunks(pdf_path: str):
    return tuple(load_and_split(pdf_path))


def build_hybrid(collection: str, pdf_path, k: int = TOP_K):
    chunks = list(_cached_chunks(str(pdf_path)))

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    vector = get_store(collection).as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])