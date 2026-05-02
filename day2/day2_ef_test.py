import chromadb
from sentence_transformers import SentenceTransformer
import time

model = SentenceTransformer("jhgan/ko-sroberta-multitask")
client = chromadb.Client()

# 더 많은 데이터로 테스트
documents = [
    f"문서 {i}: " + text
    for i, text in enumerate([
        "강아지 사료 추천해주세요",
        "개 먹이 좋은 거 알려주세요",
        "반려견 영양식 종류",
        "푸들 사료 브랜드",
        "골든리트리버 식단",
        "고양이 사료 브랜드",
        "고양이 간식 추천",
        "고양이 모래 종류",
        "자동차 엔진오일 교체",
        "타이어 공기압 점검",
        "자동차 워셔액 보충",
        "엔진 경고등 의미",
        "오늘 서울 날씨",
        "부산 주말 날씨",
        "주말 여행지 추천",
        "제주도 맛집",
        "서울 카페 추천",
        "수영 배우는 법",
        "헬스장 추천 운동",
        "다이어트 식단",
    ])
]

embeddings = model.encode(documents).tolist()

# search_ef 값을 바꿔가며 같은 검색 실행
ef_values = [5, 10, 50, 100]
query = "강아지 먹이"
query_embedding = model.encode(query).tolist()

for ef in ef_values:
    # 매번 새 컬렉션 (파라미터 적용 위해)
    name = f"test_ef_{ef}"
    try:
        client.delete_collection(name)
    except:
        pass
    
    collection = client.create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:search_ef": ef,
        }
    )
    collection.add(
        embeddings=embeddings,
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    
    # 검색 시간 측정
    start = time.time()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    elapsed = (time.time() - start) * 1000
    
    print(f"\n[search_ef = {ef}] 검색 시간: {elapsed:.2f}ms")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"  {i+1}. 거리={distance:.4f} | {doc}")