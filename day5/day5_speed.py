from sentence_transformers import SentenceTransformer, CrossEncoder
import time

embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
reranker = CrossEncoder("Dongjin-kr/ko-reranker")

# 가짜 후보 데이터 (실제 청크라고 가정)
candidates = [f"이것은 {i}번째 후보 문서입니다. 강아지 사료에 대한 내용입니다." 
              for i in range(100)]

query = "강아지 사료 추천"

# Bi-encoder 속도 측정 (참고용)
print("[Bi-encoder 임베딩 속도]")
for n in [10, 30, 100]:
    # 워밍업
    embedder.encode(candidates[:n])
    # 측정
    start = time.time()
    embedder.encode(candidates[:n])
    elapsed = (time.time() - start) * 1000
    print(f"  {n}개 임베딩: {elapsed:.1f}ms")

print("\n[Cross-encoder Re-ranking 속도]")
for n in [10, 30, 50, 100]:
    pairs = [(query, doc) for doc in candidates[:n]]
    # 워밍업
    reranker.predict(pairs)
    # 측정
    start = time.time()
    reranker.predict(pairs)
    elapsed = (time.time() - start) * 1000
    print(f"  {n}개 재정렬: {elapsed:.1f}ms")