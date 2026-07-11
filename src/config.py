"""프로젝트 전역 설정."""
from pathlib import Path

# 경로
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"

# 모델
LLM_HEAVY = "llama3.3:70b"      # 정밀 분석용 (~5.7 tok/s)
LLM_FAST = "llama3.1:8b"        # 대량 처리용
EMBED_MODEL = "nomic-embed-text"

# RAG 파라미터
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200
TOP_K = 4