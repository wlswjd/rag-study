import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jhgan/ko-sroberta-multitask")
client = chromadb.Client()

# 청크 (일부 메타데이터 누락된 상황 시뮬레이션)
data = [
    {
        "text": "EP.47에서 유재석과 박진영이 만났습니다",
        "metadata": {"video_id": "abc123", "upload_date": 20240315}
    },
    {
        "text": "EP.48 게스트는 이서진이었습니다",
        "metadata": {"video_id": "def456", "upload_date": 20240322}
    },
    {
        "text": "EP.49 핑계고 100회 특집 방송",
        "metadata": {"video_id": "ghi789", "upload_date": 0}
    },
    {
        "text": "EP.50 김원희 출연 회차",
        "metadata": {"video_id": "jkl012", "upload_date": 20240412}
    },
    {
        "text": "EP.51 연말 특집 미공개 자료",
        "metadata": {"video_id": "mno345", "upload_date": 0}
    },
]

# 안전한 메타데이터 처리 함수
def safe_metadata(meta):
    return {
        "video_id": meta.get("video_id", "unknown"),
        "upload_date": meta.get("upload_date", 0)
    }

# 정수 → 보기 좋은 날짜 문자열로 변환 (← 이 함수가 빠져 있었습니다)
def format_date(d):
    if d == 0:
        return "(날짜 누락)"
    s = str(d)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

# 컬렉션 생성 + 데이터 추가
collection = client.create_collection(
    name="day6_missing",
    metadata={"hnsw:space": "cosine"}
)

documents = [d["text"] for d in data]
metadatas = [safe_metadata(d["metadata"]) for d in data]
embeddings = model.encode(documents).tolist()

collection.add(
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# 검색 함수
def search(query, where=None, label=""):
    q_emb = model.encode(query).tolist()
    kwargs = {"query_embeddings": [q_emb], "n_results": 10}
    if where:
        kwargs["where"] = where
    
    results = collection.query(**kwargs)
    
    print(f"\n[{label}]")
    if where:
        print(f"  필터: {where}")
    print("-" * 60)
    
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        date_str = format_date(meta['upload_date'])
        print(f"  [{date_str}] {doc}")

# ===========================
# 누락 처리 실험
# ===========================

search("핑계고 영상", label="전체")
search("핑계고 영상", where={"upload_date": {"$ne": 0}}, label="날짜 있는 것만")
search("핑계고 영상", where={"upload_date": 0}, label="날짜 누락된 것만")
search(
    "핑계고 영상",
    where={
        "$and": [
            {"upload_date": {"$gte": 20240401}},
            {"upload_date": {"$ne": 0}}
        ]
    },
    label="2024-04-01 이후"
)