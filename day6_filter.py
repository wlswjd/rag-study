import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jhgan/ko-sroberta-multitask")
client = chromadb.Client()

# 청크 + 메타데이터 (마인크래프트 가상 데이터)
documents = [
    "1.20 패치에서 새로운 고고학 시스템이 추가되었습니다",
    "1.19 패치는 야생 업데이트로 알려져 있습니다",
    "1.18 패치에서 동굴 생성 방식이 크게 바뀌었습니다",
    "구리 곡괭이는 1.21에서 추가될 예정인 도구입니다",
    "다이아몬드 곡괭이는 흑요석을 캘 수 있는 유일한 도구입니다",
    "철 곡괭이는 다이아몬드를 캘 수 있는 가장 약한 도구입니다",
    "워든은 어둠의 깊은 곳에서 등장하는 강력한 몹입니다",
    "엔더 드래곤은 엔드 차원의 보스입니다",
    "유저 꿀팁: 침대 위에서 몹이 스폰되지 않습니다",
    "유저 꿀팁: 사탕수수는 모래에서만 자랍니다",
]

metadatas = [
    {"category": "patch_notes", "version": "1.20", "source": "official_wiki"},
    {"category": "patch_notes", "version": "1.19", "source": "official_wiki"},
    {"category": "patch_notes", "version": "1.18", "source": "official_wiki"},
    {"category": "items", "version": "1.21", "source": "official_wiki"},
    {"category": "items", "version": "1.0", "source": "official_wiki"},
    {"category": "items", "version": "1.0", "source": "official_wiki"},
    {"category": "mobs", "version": "1.19", "source": "official_wiki"},
    {"category": "mobs", "version": "1.0", "source": "official_wiki"},
    {"category": "tips", "version": "unknown", "source": "community"},
    {"category": "tips", "version": "unknown", "source": "community"},
]

# 컬렉션 생성 + 데이터 추가
collection = client.create_collection(
    name="day6_filter",
    metadata={"hnsw:space": "cosine"}
)

embeddings = model.encode(documents).tolist()
collection.add(
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# 검색 함수
def search(query, where=None, n_results=5, label=""):
    q_emb = model.encode(query).tolist()
    
    kwargs = {
        "query_embeddings": [q_emb],
        "n_results": n_results
    }
    if where:
        kwargs["where"] = where
    
    results = collection.query(**kwargs)
    
    print(f"\n[{label}] {query}")
    if where:
        print(f"  필터: {where}")
    print("-" * 60)
    
    if not results['documents'][0]:
        print("  검색 결과 없음")
        return
    
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"  {i+1}. (거리={dist:.3f}) [{meta['category']}/{meta['version']}] {doc}")

# ===========================
# 비교 실험
# ===========================

# 1. 일반 검색
search("1.20 패치 내용", label="필터 없음")

# 2. 카테고리 필터
search("패치 내용", where={"category": "patch_notes"}, label="패치노트만")

# 3. 버전 필터
search("도구", where={"version": "1.21"}, label="1.21 버전만")

# 4. 출처 필터
search("꿀팁", where={"source": "community"}, label="커뮤니티만")

# 5. AND 조건 (여러 조건 동시)
search(
    "도구",
    where={
        "$and": [
            {"category": "items"},
            {"source": "official_wiki"}
        ]
    },
    label="공식 위키 + 아이템"
)

# 6. $in 연산자 (여러 값 중 하나)
search(
    "패치",
    where={"version": {"$in": ["1.19", "1.20"]}},
    label="1.19 또는 1.20"
)