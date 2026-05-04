from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
import re

okt = Okt()

documents = [
    "강아지가 좋아하는 사료를 추천합니다",
    "강아지를 위한 사료 종류",
    "강아지 사료 브랜드 비교",
    "고양이가 좋아하는 간식",
]

# 방법 1: 단순 공백 분리
def simple_tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

# 방법 2: 형태소 분석 (명사만 추출)
def morpheme_tokenize(text):
    return okt.nouns(text)

# 방법 3: 형태소 분석 (어간 추출 — 명사+동사+형용사)
def stem_tokenize(text):
    return [word for word, _ in okt.pos(text, stem=True) 
            if _ in ['Noun', 'Verb', 'Adjective']]

# 비교
print("원문:", documents[0])
print("단순 분리:", simple_tokenize(documents[0]))
print("명사 추출:", morpheme_tokenize(documents[0]))
print("어간 포함:", stem_tokenize(documents[0]))

# 두 방식으로 BM25 인덱스 만들고 비교
docs_simple = [simple_tokenize(d) for d in documents]
docs_morph = [morpheme_tokenize(d) for d in documents]

bm25_simple = BM25Okapi(docs_simple)
bm25_morph = BM25Okapi(docs_morph)

# 같은 질문으로 비교
query = "강아지 사료"

print(f"\n[질문] {query}")
print("\n[단순 분리 BM25]")
scores = bm25_simple.get_scores(simple_tokenize(query))
for i, score in enumerate(scores):
    print(f"  {score:.3f} | {documents[i]}")

print("\n[형태소 분석 BM25]")
scores = bm25_morph.get_scores(morpheme_tokenize(query))
for i, score in enumerate(scores):
    print(f"  {score:.3f} | {documents[i]}")