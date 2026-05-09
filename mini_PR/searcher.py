# rag-study/day7_mini/searcher.py

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
import re

# ============================
# 설정
# ============================
CHROMA_PATH = "./mini_chroma_db"
BM25_PATH = "./bm25_index.pkl"
COLLECTION_NAME = "recipes"
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
RERANKER_MODEL = "Dongjin-kr/ko-reranker"

# 검색 파라미터 (Day 5에서 결정한 값)
TOP_N = 10        # 1단계 후보
TOP_K = 3         # 최종 결과
RRF_K = 60        # RRF 상수


# ============================
# 토큰화 (BM25용, indexer.py와 동일해야 함)
# ============================
def tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


# ============================
# 인덱스 로딩
# ============================
class RecipeSearcher:
    def __init__(self):
        print("[검색기 초기화 중...]")
        
        # 임베딩 모델
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        # Cross-encoder
        self.reranker = CrossEncoder(RERANKER_MODEL)
        
        # ChromaDB 로드
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_collection(COLLECTION_NAME)
        
        # BM25 로드
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.tokenized = data["tokenized"]
            self.ids = data["ids"]
            self.documents = data["documents"]
        
        # ID → 인덱스 매핑 (BM25는 인덱스 기반이라 변환 필요)
        self.id_to_idx = {id_: i for i, id_ in enumerate(self.ids)}
        self.idx_to_id = {i: id_ for i, id_ in enumerate(self.ids)}
        
        print("[초기화 완료]\n")
    
    
    # ============================
    # 1. Dense 검색
    # ============================
    def _dense_search(self, query, top_n, where=None):
        """Dense 검색 — (id, 순위) 리스트 반환"""
        q_emb = self.embedder.encode(query).tolist()
        
        kwargs = {
            "query_embeddings": [q_emb],
            "n_results": top_n
        }
        if where:
            kwargs["where"] = where
        
        results = self.collection.query(**kwargs)
        
        # 결과가 비었을 수도 있음 (필터로 다 걸러진 경우)
        if not results['ids'][0]:
            return []
        
        return [(doc_id, rank + 1) 
                for rank, doc_id in enumerate(results['ids'][0])]
    
    
    # ============================
    # 2. BM25 검색
    # ============================
    def _bm25_search(self, query, top_n, allowed_ids=None):
        """BM25 검색 — (id, 순위) 리스트 반환
        
        allowed_ids: 메타데이터 필터 통과한 ID 집합 (있으면 그 안에서만)
        """
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        
        # 점수 0 이상만 + (필터 있으면) allowed_ids 안에 있는 것만
        scored = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            doc_id = self.idx_to_id[idx]
            if allowed_ids is not None and doc_id not in allowed_ids:
                continue
            scored.append((doc_id, score))
        
        # 점수순 정렬
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [(doc_id, rank + 1) for rank, (doc_id, _) in enumerate(scored[:top_n])]
    
    
    # ============================
    # 3. RRF 결합
    # ============================
    def _rrf_combine(self, dense_results, bm25_results):
        """두 검색 결과를 RRF로 결합"""
        rrf_scores = {}
        
        for doc_id, rank in dense_results:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)
        
        for doc_id, rank in bm25_results:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)
        
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_ids]
    
    
    # ============================
    # 4. Re-ranking
    # ============================
    def _rerank(self, query, candidate_ids, top_k):
        """Cross-encoder로 재정렬"""
        # ID로 텍스트 가져오기
        candidates_text = []
        for doc_id in candidate_ids:
            idx = self.id_to_idx[doc_id]
            candidates_text.append(self.documents[idx])
        
        # Cross-encoder 점수
        pairs = [(query, text) for text in candidates_text]
        scores = self.reranker.predict(pairs)
        
        # 점수순 정렬 (id, text, 점수)
        scored = list(zip(candidate_ids, candidates_text, scores))
        scored.sort(key=lambda x: x[2], reverse=True)
        
        return scored[:top_k]
    
    
    # ============================
    # 5. 메타데이터로 필터된 ID 가져오기
    # ============================
    def _get_filtered_ids(self, where):
        """where 조건에 맞는 ID 집합 반환 (BM25용)"""
        if not where:
            return None
        
        # ChromaDB get으로 메타데이터 필터링만 적용
        results = self.collection.get(where=where)
        return set(results['ids'])
    
    
    # ============================
    # 통합 검색 (메인 인터페이스)
    # ============================
    def search(self, query, where=None, top_k=TOP_K, verbose=False):
        """Hybrid + Re-rank + 메타데이터 필터를 통합한 검색"""
        
        # 0. 메타데이터 필터로 후보 ID 사전 추출 (BM25용)
        allowed_ids = self._get_filtered_ids(where)
        
        # 1. Dense 검색 (where는 ChromaDB가 직접 처리)
        dense_results = self._dense_search(query, top_n=TOP_N, where=where)
        
        # 2. BM25 검색 (수동으로 allowed_ids 필터링)
        bm25_results = self._bm25_search(query, top_n=TOP_N, allowed_ids=allowed_ids)
        
        if verbose:
            print(f"  Dense 결과 수: {len(dense_results)}")
            print(f"  BM25 결과 수: {len(bm25_results)}")
        
        # 검색 결과가 없으면 빈 리스트 반환
        if not dense_results and not bm25_results:
            return []
        
        # 3. RRF 결합
        merged_ids = self._rrf_combine(dense_results, bm25_results)
        
        if verbose:
            print(f"  RRF 결합 후 후보: {len(merged_ids)}개")
        
        # 4. Re-ranking
        final = self._rerank(query, merged_ids, top_k)
        
        # 5. 결과 정리 (메타데이터 포함)
        results = []
        for doc_id, text, score in final:
            meta = self.collection.get(ids=[doc_id])['metadatas'][0]
            results.append({
                "id": doc_id,
                "text": text,
                "rerank_score": float(score),
                "metadata": meta
            })
        
        return results