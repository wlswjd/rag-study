from rank_bm25 import BM25Okapi
import re

# 같은 도메인 데이터로 비교 (Day 2와 유사)
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

# 간단한 토큰화 (한국어 단어 + 공백 분리)
def simple_tokenize(text):
    # 한글, 영문, 숫자만 남기고 공백 분리
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

# 모든 문서를 토큰화
tokenized_docs = [simple_tokenize(doc) for doc in documents]

print("토큰화 결과 예시:")
print(f"원문: {documents[0]}")
print(f"토큰: {tokenized_docs[0]}\n")

# BM25 인덱스 구축
bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)

# 검색 함수
def search_bm25(query, top_k=5):
    tokens = simple_tokenize(query)
    scores = bm25.get_scores(tokens)
    
    # 점수순 정렬
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    
    print(f"\n[질문] {query}")
    print(f"[토큰화] {tokens}")
    print("-" * 60)
    for i, (idx, score) in enumerate(ranked[:top_k]):
        print(f"{i+1}. 점수={score:.4f} | {documents[idx]}")
    return ranked[:top_k]

# 테스트 1: 정확한 단어 매칭
search_bm25("EP.47")

# 테스트 2: 고유명사
search_bm25("최예나")

# 테스트 3: 일반 키워드
search_bm25("글루코사민 사료")

# 테스트 4: 동의어 (BM25가 약한 영역)
search_bm25("개 먹이")