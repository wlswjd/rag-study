from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
import re

# 테스트용 텍스트 (반려동물 가이드 스타일)
text = """푸들 사료를 고를 때는 연령을 먼저 봐야 합니다. 7살 이상의 노령견은 관절 건강을 위해 글루코사민이 함유된 제품을 선택하는 것이 좋습니다. 또한 칼로리는 일반 성견용보다 10~15% 낮춰야 합니다.

고양이 사료의 경우 단백질 함량이 가장 중요합니다. 고양이는 완전 육식동물이기 때문에 식물성 단백질로는 영양 요구를 충족할 수 없습니다. 동물성 단백질이 최소 30% 이상 함유된 제품을 선택하세요.

자동차 엔진오일은 일반적으로 5,000km 또는 6개월마다 교체합니다. 합성유는 1만km까지 사용할 수 있지만 가혹 조건에서는 더 자주 교체해야 합니다."""

print(f"원본 길이: {len(text)}자\n")
print("=" * 60)

# ============================================
#1. fixed-size(단순 글자 수 자르기)
# ============================================
def fixed_chunk(text, size=100):
    return[text[i:i+size] for i in range(0, len(text), size)]

print("\n[1] Fixed-size (size=100, overlap=0)")
print("-" * 60)
chunks = fixed_chunk(text, 100)
for i, c in enumerate(chunks):
    print(f"청크 {i+1} ({len(c)}자): {c[:50]}...")

# ============================================
# 2. Sliding Window (overlap 추가)
# ============================================
def sliding_chunk(text, size=100, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

print("\n[2] Sliding Window (size=100, overlap=20)")
print("-" * 60)
chunks = sliding_chunk(text, 100, 20)
for i, c in enumerate(chunks):
    print(f"청크 {i+1} ({len(c)}자): {c[:50]}...")

# ============================================
# 3. Sentence-based (문장 단위)
# ============================================
def sentence_chunk(text, max_chars=150):
    # 한국어 문장 종결 부호 기반 분리 (간단 버전)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks

print("\n[3] Sentence-based (max=150)")
print("-" * 60)
chunks = sentence_chunk(text, 150)
for i, c in enumerate(chunks):
    print(f"청크 {i+1} ({len(c)}자): {c[:50]}...")

# ============================================
# 4. Recursive (LangChain)
# ============================================
print("\n[4] Recursive (size=150, overlap=30)")
print("-" * 60)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(text)
for i, c in enumerate(chunks):
    print(f"청크 {i+1} ({len(c)}자): {c[:50]}...")