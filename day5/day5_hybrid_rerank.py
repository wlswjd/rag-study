from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import re
import time

# ===========================
# 1. 데이터 준비
# ===========================
documents = [
    "강아지 사료 추천해주세요. 7살 푸들에게 좋은 사료가 뭔가요",
    "개 먹이 좋은 거 알려주세요",
    "반려견 영양식 종류와 선택 방법",
    "푸들 사료 브랜드 추천",
    "골든리트리버 식단 관리 방법",
    "노령견 글루코사민 함유 사료 효과",
    "강아지 사료 보관과 유통기한 관리",
    "고양이 사료 브랜드 추천",
    "고양이 간식 종류",
    "자동차 엔진오일 5000km마다 교체",
    "타이어 공기압 점검 방법",
    "강아지가 토할 때 응급처치",
    "강아지 산책 주의사항",
    "오늘 서울 날씨 맑음",
    "주말 여행지 추천",
]

# ===========================
# 2. 모델 로드
# ===========================
print("Cross-encoder 모델 로드 중...")
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
reranker = CrossEncoder("Dongjin-kr/ko-reranker")
print("로딩 완료\n")

# ===========================
# 3. 인덱스 구축
# ===========================
def simple_tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

#bm25
tokenized_docs = [simple_tokenize(doc) for doc in documents]
bm25 = BM250kapi(tokenized_docs, k1=1.5, b=0.75)

#dense
client = chromadb.Client()
collection = client.create_collection(
    name="day5_rerank",
    metadata = {"hnsw:space": "cosine"}
)
embeddings = embedder.encode(documents).tolist()
collection.add(
    embeddings = embeddings,
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# ===========================
# 4. 검색 함수
# ===========================
def search_dense(query, top_n=10):
    q_emb = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_n)
    return [(int(doc_id.split('_')[1]), rank + 1)
            for rank, doc_id in enumerate(results['ids'][0])]

def search_bm25(query, top_n=10):
    tokens = simple_tokenize(query)
    scores = bm25.get_scores(tokens)
    scored = [(i, s) for i, s in enumerate(scores) if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(idx, rank+1) for rank, (idx, _) in enumerate(scored[:top_n])]

def hybrid_search(query, top_n=10, k_rrf=60):
    """Day 4와 동일 — RRF로 Dense + BM25 결합"""
    dense = search_dense(query, top_n)
    bm25_r = search_bm25(query, top_n)
    
    rrf_scores = {}
    for idx, rank in dense:
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank)
    for idx, rank in bm25_r:
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank)
    
    final = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in final[:top_n]]

def hybrid_with_rerank(query, top_n=10, top_k=5):
    """Hybrid로 후보 N개 → Cross-encoder로 K개 재정렬"""
    # 1단계: Hybrid로 후보 가져오기
    candidates_idx = hybrid_search(query, top_n=top_n)
    candidates_docs = [documents[i] for i in candidates_idx]
    
    # 2단계: Cross-encoder로 재정렬
    pairs = [(query, doc) for doc in candidates_docs]
    rerank_scores = reranker.predict(pairs)
    
    # 점수순 정렬
    reranked = sorted(
        zip(candidates_idx, candidates_docs, rerank_scores),
        key=lambda x: x[2],
        reverse=True
    )
    return reranked[:top_k]

# ===========================
# 5. 비교 출력
# ===========================
def compare(query):
    print(f"\n{'=' * 70}")
    print(f"[질문] {query}")
    print(f"{'=' * 70}")
    
    # Hybrid만
    hybrid_idx = hybrid_search(query, top_n=5)
    print("\n[Hybrid 단독 (Top 5)]")
    for rank, idx in enumerate(hybrid_idx, 1):
        print(f"  {rank}. {documents[idx]}")
    
    # Hybrid + Re-rank
    print("\n[Hybrid + Re-ranking (N=10 → K=5)]")
    final = hybrid_with_rerank(query, top_n=10, top_k=5)
    for rank, (idx, doc, score) in enumerate(final, 1):
        print(f"  {rank}. (rerank={score:+.3f}) {doc}")

# ===========================
# 6. 테스트
# ===========================
compare("강아지 사료 추천")
compare("노령견 관절 건강")
compare("강아지 응급 상황")