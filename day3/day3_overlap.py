import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# "관절 건강 글루코사민"이 청크 경계에 걸리도록 의도적으로 구성
text = "푸들 사료는 연령이 중요합니다. 7살 이상의 노령견은 관절 건강을 위해 글루코사민이 함유된 사료를 선택해야 합니다. 단백질 함량도 확인하세요."

# Overlap 0
def chunk_no_overlap(text, size=50):
    return [text[i:i+size] for i in range(0, len(text), size)]

# Overlap 20
def chunk_with_overlap(text, size=50, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

chunks_no = chunk_no_overlap(text, 50)
chunks_yes = chunk_with_overlap(text, 50, 20)

print("[Overlap 없음]")
for i, c in enumerate(chunks_no):
    print(f"  청크 {i+1}: {c}")

print("\n[Overlap 20]")
for i, c in enumerate(chunks_yes):
    print(f"  청크 {i+1}: {c}")

# 두 방식으로 만든 청크를 각각 ChromaDB에 넣고 검색 비교
client = chromadb.Client()

def test_search(chunks, name, query):
    coll = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    embs = model.encode(chunks).tolist()
    coll.add(embeddings=embs, documents=chunks, ids=[f"c{i}" for i in range(len(chunks))])
    
    q_emb = model.encode(query).tolist()
    results = coll.query(query_embeddings=[q_emb], n_results=2)
    
    print(f"\n[{name}] 질문: {query}")
    for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"  {i+1}. 거리={dist:.4f} | {doc}")

query = "관절에 좋은 글루코사민"

test_search(chunks_no, "no_overlap", query)
test_search(chunks_yes, "with_overlap", query)