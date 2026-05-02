from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 짧은 텍스트
short = "푸들 사료는 연령에 맞게 골라야 합니다. 노령견은 글루코사민이 좋습니다."

# 긴 텍스트 (앞부분은 푸들, 뒷부분은 자동차 - 의도적으로 다른 주제)
puddle_part = "푸들 사료는 연령에 맞게 골라야 합니다. " * 30  # 약 600자
car_part = "자동차 엔진오일은 5000km마다 교체합니다. " * 30  # 약 600자
long_text = puddle_part + car_part  # 약 1200자

# 토큰 수 확인
def count_tokens(text):
    return len(model.tokenize([text])['input_ids'][0])

print(f"짧은 텍스트: {len(short)}자, {count_tokens(short)}토큰")
print(f"긴 텍스트:   {len(long_text)}자, {count_tokens(long_text)}토큰")
print(f"모델 최대 토큰: {model.max_seq_length}\n")

# 임베딩 생성
emb_short = model.encode(short)
emb_long = model.encode(long_text)
emb_puddle_only = model.encode(puddle_part)  # 앞부분만
emb_car_only = model.encode(car_part)        # 뒷부분만

# 유사도 함수
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("=" * 60)
print("긴 텍스트(푸들+자동차)의 임베딩 vs 각 부분의 임베딩")
print("=" * 60)
print(f"긴 텍스트 ↔ 푸들 부분만:    {cos_sim(emb_long, emb_puddle_only):.4f}")
print(f"긴 텍스트 ↔ 자동차 부분만:  {cos_sim(emb_long, emb_car_only):.4f}")
print()
print("→ 두 값 중 어느 쪽이 높은지 보세요")
print("  '긴 텍스트'는 자동차 정보가 임베딩에 거의 안 들어갔음을 의미")