# rag-study/day7_mini/indexer.py

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
import pickle
import os

from recipes_data import RECIPES

# ============================
# 설정
# ============================
CHROMA_PATH = "./mini_chroma_db"
BM25_PATH = "./bm25_index.pkl"
COLLECTION_NAME = "recipes"
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"


# ============================
# 토큰화 (BM25용)
# ============================
def tokenize(text):
    """간단 토큰화 — 실무에서는 형태소 분석기 권장"""
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


# ============================
# 청크 + 메타데이터 준비
# ============================
def prepare_documents(recipes):
    """레시피 → (청크 텍스트, 메타데이터, ID) 분리"""
    documents = []
    metadatas = []
    ids = []
    
    for recipe in recipes:
        # 청크 텍스트: 이름 + 설명
        # (이름을 본문에 포함하면 임베딩 + BM25 둘 다에 도움)
        text = f"{recipe['name']}: {recipe['description']}"
        
        # 메타데이터: 검색 필터에 쓸 정보
        metadata = {
            "name": recipe["name"],
            "category": recipe["category"],
            "difficulty": recipe["difficulty"],
            "time_minutes": recipe["time_minutes"],
            "main_ingredient": recipe["main_ingredient"]
        }
        
        documents.append(text)
        metadatas.append(metadata)
        ids.append(recipe["id"])
    
    return documents, metadatas, ids


# ============================
# Dense 인덱스 (ChromaDB)
# ============================
def build_dense_index(documents, metadatas, ids):
    print("\n[1/2] Dense 인덱스 구축 중...")
    
    # 임베딩 모델 로드
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # ChromaDB 클라이언트 (영구 저장)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # 기존 컬렉션 삭제 (다시 빌드하기 위함)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  기존 컬렉션 삭제됨")
    except:
        pass
    
    # 컬렉션 생성 + HNSW 파라미터 명시
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",       # Day 2: 텍스트는 cosine
            "hnsw:M": 16,
            "hnsw:construction_ef": 100,
            "hnsw:search_ef": 30,
        }
    )
    
    # 임베딩 생성
    print(f"  {len(documents)}개 청크 임베딩 중...")
    embeddings = model.encode(documents, show_progress_bar=False).tolist()
    
    # DB에 추가
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"  완료: {collection.count()}개 청크 저장")
    return collection


# ============================
# BM25 인덱스
# ============================
def build_bm25_index(documents, ids):
    print("\n[2/2] BM25 인덱스 구축 중...")
    
    # 토큰화
    tokenized = [tokenize(doc) for doc in documents]
    
    # BM25 인덱스 생성
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    
    # pickle로 저장 (BM25 객체 + 토큰화된 문서 + ids 함께)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "tokenized": tokenized,
            "ids": ids,
            "documents": documents
        }, f)
    
    print(f"  완료: {len(documents)}개 청크 BM25 인덱싱")


# ============================
# 메인
# ============================
def main():
    print(f"=== 레시피 RAG 인덱싱 시작 ===")
    print(f"대상: {len(RECIPES)}개 레시피")
    
    # 데이터 준비
    documents, metadatas, ids = prepare_documents(RECIPES)
    
    # 인덱스 구축
    build_dense_index(documents, metadatas, ids)
    build_bm25_index(documents, ids)
    
    print("\n=== 인덱싱 완료 ===")
    print(f"  Dense DB: {CHROMA_PATH}")
    print(f"  BM25 Index: {BM25_PATH}")


if __name__ == "__main__":
    main()