import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jhgan/ko-sroberta-multitask")
client = chromadb.Client()

collection = client.create_collection(
    name="day2_hnsw",
    metadata={
        "hnsw:space": "cosine",         #거리함수(코사인)
        "hnsw:m": 16,                   #각 노드의 이웃 수
        "hnsw:ef_construction": 100,    #인덱스 구축 시 탐색 범위
        "hnsw:ef_search": 100,          #검색 시 탐색 범위
    }
)

documents = [
    "강아지 사료 추천해주세요",
    "개 먹이 좋은 거 알려주세요",
    "반려견 영양식 종류",
    "고양이 사료 브랜드",
    "고양이 간식 추천",
    "자동차 엔진오일 교체",
    "타이어 공기압 점검",
    "오늘 날씨 어때요",
    "주말 여행지 추천",
]

embeddings = model.encode(documents).tolist()
collection.add(
    embeddings=embeddings,
    documents=documents,
    ids = [f"doc_{i}" for i in range(len(documents))]
)

#코사인 거리 기반 검색
query ="강아지 사료"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings = [query_embedding],
    n_results = 3,
)

print(f"[질문] {query}")
print(f"[거리 함수] 코사인 거리 (1 - 코사인 유사도)")
print("-" * 50)
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    similarity = 1 - distance  # 코사인 거리 → 코사인 유사도 변환
    print(f"{i+1}. 거리={distance:.4f} 유사도={similarity:.4f} | {doc}")