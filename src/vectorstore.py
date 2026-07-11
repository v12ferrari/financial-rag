"""문서 로드 → 청크 분할 → 벡터DB 저장/조회."""
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


def load_and_split(pdf_path: str | Path) -> list:
    """PDF를 읽어 청크 리스트로 반환."""
    pages = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    print(f"[load] {Path(pdf_path).name}: {len(pages)}p -> {len(chunks)} chunks")
    return chunks


def build_store(collection: str, pdf_path: str | Path, batch_size: int = 50) -> Chroma:
    """PDF를 임베딩해 컬렉션에 저장. 컬렉션명으로 문서 간 격리."""
    chunks = load_and_split(pdf_path)

    # 출처 추적용 메타데이터
    for c in chunks:
        c.metadata["collection"] = collection

    store = Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )

    for i in range(0, len(chunks), batch_size):
        store.add_documents(chunks[i : i + batch_size])
        print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)}")
        time.sleep(0.3)

    print(f"[build] '{collection}' done")
    return store


def get_store(collection: str) -> Chroma:
    """이미 만들어둔 컬렉션 불러오기 (재임베딩 없음)."""
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )