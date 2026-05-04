from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
import re

# 1. 데이터 준비
documents = [
    "강아지 사료 추천해주세요. 7살 푸들에게 좋은 사료가 뭔가요",
    "개 먹이 좋은 거 알려주세요",
    "반려견 영양식 종류와 선택 방법",
    "푸들 사료 브랜드 추천",
    "골든리트리버 식단 관리",
    "고양이 사료 브랜드 추천",
    "고양이 간식 종류",
    "고양이 모래 비교",
    "자동차 엔진오일 5000km마다 교체",
    "타이어 공기압 점검 방법",
    "에피소드 47에서 출연자가 한 말",
    "EP.47 마지막 회 분석",
    "글루코사민 함유 사료의 효과",
    "최예나 출연 영상 모음",
    "예나가 나온 그 회차",
]

# 2. 토큰화
def simple_tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

tokenized_docs = [simple_tokenize(doc) for doc in documents]

# 3. BM25 인덱스
bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)

# 4. Dense 인덱스 (ChromaDB)
model = SentenceTransformer("jhgan/ko-sroberta-multitask")
client = chromadb.Client()
collection = client.create_collection(
    name="hybrid_test",
    metadata={"hnsw:space": "cosine"}
)
embeddings = model.encode(documents).tolist()
collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# 5. 검색 함수들
def search_dense(query, top_k=10):
    """Dense 검색 — 결과를 (doc_idx, 순위) 리스트로 반환"""
    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    
    ranked = []
    for rank, doc_id in enumerate(results['ids'][0]):
        idx = int(doc_id.split('_')[1])
        ranked.append((idx, rank + 1))  # 순위는 1부터 시작
    return ranked

def search_bm25_ranked(query, top_k=10):
    """BM25 검색 — 결과를 (doc_idx, 순위) 리스트로 반환"""
    tokens = simple_tokenize(query)
    scores = bm25.get_scores(tokens)
    
    # 점수 0 이상만, 점수순 정렬
    scored = [(i, s) for i, s in enumerate(scores) if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    ranked = [(idx, rank + 1) for rank, (idx, _) in enumerate(scored[:top_k])]
    return ranked

def hybrid_search(query, top_k=5, k_rrf=60):
    """RRF로 두 검색을 결합"""
    dense_results = search_dense(query, top_k=10)
    bm25_results = search_bm25_ranked(query, top_k=10)
    
    # RRF 점수 계산
    rrf_scores = {}
    
    for idx, rank in dense_results:
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank)
    
    for idx, rank in bm25_results:
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank)
    
    # 점수순 정렬
    final = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return final[:top_k], dense_results, bm25_results

# 6. 비교 출력
def compare_search(query):
    final, dense, bm25_r = hybrid_search(query, top_k=5)
    
    print(f"\n{'=' * 70}")
    print(f"[질문] {query}")
    print(f"{'=' * 70}")
    
    print("\n[Dense 검색 상위 5]")
    for idx, rank in dense[:5]:
        print(f"  {rank}위: {documents[idx]}")
    
    print("\n[BM25 검색 상위 5]")
    if not bm25_r:
        print("  매칭 결과 없음")
    for idx, rank in bm25_r[:5]:
        print(f"  {rank}위: {documents[idx]}")
    
    print("\n[Hybrid (RRF) 최종 상위 5]")
    for i, (idx, score) in enumerate(final):
        print(f"  {i+1}위: RRF={score:.5f} | {documents[idx]}")

# 테스트
compare_search("EP.47")           # BM25에 유리
compare_search("개 먹이")          # Dense에 유리 (동의어)
compare_search("글루코사민 사료")   # 둘 다 유리
compare_search("예나")             # 부분 매칭 (애매한 케이스)