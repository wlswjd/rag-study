from sentence_transformers import SentenceTransformer
import time

print("Cross-encoder 모델 로드 중...")
start = time.time()
reranker = SentenceTransformer("Dongjin-kr/ko-reranker")
print(f"로딩완료: {time.time() - start:.1f}")

#같은 질문에 대한 다양한 후보 문서들
query = "강아지 사료 추천"

candidates = [
    "푸들에게 좋은 사료 종류와 추천",          # 매우 관련 있음
    "강아지가 토할 때 응급처치 방법",            # 강아지 관련이지만 사료 아님
    "고양이 사료 브랜드 비교",                  # 사료 관련이지만 강아지 아님
    "자동차 엔진오일 5000km마다 교체",         # 완전 무관
    "노령견을 위한 글루코사민 함유 사료",       # 매우 관련 있음
    "오늘 서울 날씨는 맑음",                    # 완전 무관
    "강아지 사료 보관 방법과 유통기한",          # 관련 있음
]

#cross-encoder 입력 형식 : 질문+후보 리스트
pairs = [(query, doc) for doc in candidates]

#점수 계산
print(f"[질문] {query}\n")
print("점수 게산중 ...")
start = time.time()
scores = reranker.predict(pairs)
elapsed = (time.time() - start) * 1000
print(f"소요시간: {elapsed:.1f}ms ({len(pairs)}개 후보)\n")

#점수순으로 정렬
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=true)

print("="*50)
print(f"{'순위':<4} {'점수':>10}  문서")
print("=" * 60)
for rank, (doc, score) in enumerate(ranked, 1):
    print(f"{rank:<4} {score:>10.4f}  {doc}")