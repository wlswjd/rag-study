import chromadb
from sentence_transformers import SentenceTransformer

#1. 임베딩 모델 로드
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

#2. chromadb 클라이언트 생성
client = chromadb.Client()

#3. 털렉션 생성(db의 테이블 개념)
collection = client.create_collection(name="day2_test")

#데이터 준비
documents = [
    "강아지 사료 추천해주세요. 7살 푸들에게 좋은 사료가 뭔가요?",
    "고양이가 갑자기 밥을 안 먹어요. 어떻게 해야 하나요?",
    "자동차 엔진오일은 보통 5,000km마다 교체합니다.",
    "타이어 공기압은 한 달에 한 번 점검하는 게 좋습니다.",
    "오늘 서울 날씨는 맑고 기온은 23도입니다.",
    "부산 해운대 근처 맛집 추천해주세요.",
    "푸들은 활동량이 많아서 매일 산책이 필요합니다.",
    "고양이 모래는 벤토나이트와 두부 모래가 가장 많이 쓰입니다.",
]

#임베딩 생성
embeddings = model.encode(documents).tolist()

#chromadb에 추가
collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=[f"doc_{i}" for i  in range(len(documents))]
)

print(f"저장된 문서 수 : {collection.count()}")

#질문 검색
query = "강아지 키우는 법 알려주세요"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print(f"\n[질문] {query}")
print("-"*30)
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"{i+1}, (거리: {distance:.4f}) {doc}")