from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("jhgan/ko-sroberta-multitask")

sentences = [
    "강아지 사료 추천해줘",
    "개 먹이 좋은거 알려줘",
    "고양이 간식 추천해줘",
    "자동차 엔진오일 교체 방법",
    "오늘 날씨 어때",
]

#임베딩 설정
embeddings = model.encode(sentences)

print(f"벡터 차원: {embeddings.shape}")
print(f"첫 번째 문장 벡터 앞 10개 : {embeddings[0][:10]}")

#코사인 유사도 계산 함수
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#기존 문장 : 강아지 사료 추천해줘
query = "강아지 사료 추천해줘"
query_emb = model.encode(query)
print("\n[기준] 강아지 사료 추천해줘")
print("-"*30)
for i, sentence in enumerate(sentences[1:], 1):
    score = cosine_similarity(query_emb, embeddings[i])
    print(f"{score:.4f} | {sentence}")
print("-"*30)