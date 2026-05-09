# rag-study/day7_mini/verify.py

import chromadb
import pickle

# Dense 인덱스 확인
client = chromadb.PersistentClient(path="./mini_chroma_db")
collection = client.get_collection("recipes")

print("=== Dense 인덱스 ===")
print(f"청크 수: {collection.count()}")

# 샘플 데이터 1개 출력
sample = collection.get(ids=["recipe_01"])
print(f"\n샘플 (recipe_01):")
print(f"  텍스트: {sample['documents'][0][:80]}...")
print(f"  메타데이터: {sample['metadatas'][0]}")

# BM25 인덱스 확인
print("\n=== BM25 인덱스 ===")
with open("./bm25_index.pkl", "rb") as f:
    data = pickle.load(f)

print(f"청크 수: {len(data['documents'])}")
print(f"샘플 토큰화: {data['tokenized'][0][:10]}...")
print(f"샘플 ID: {data['ids'][:3]}")