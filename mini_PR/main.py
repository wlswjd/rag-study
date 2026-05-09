# rag-study/day7_mini/main.py

from searcher import RecipeSearcher


def print_results(query, results, where=None):
    """결과를 보기 좋게 출력"""
    print(f"\n{'=' * 70}")
    print(f"[질문] {query}")
    if where:
        print(f"[필터] {where}")
    print(f"{'=' * 70}")
    
    if not results:
        print("  검색 결과 없음")
        return
    
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"\n{i}. {meta['name']} (rerank={r['rerank_score']:+.3f})")
        print(f"   카테고리: {meta['category']} | "
              f"난이도: {meta['difficulty']} | "
              f"{meta['time_minutes']}분")
        print(f"   {r['text'][:120]}...")


def main():
    searcher = RecipeSearcher()
    
    # =====================================
    # 케이스 1: 정확한 이름 매칭 (BM25 강함)
    # =====================================
    results = searcher.search("김치찌개 만드는 법")
    print_results("김치찌개 만드는 법", results)
    
    # =====================================
    # 케이스 2: 의미 검색 (Dense 강함)
    # =====================================
    results = searcher.search("매콤한 국물 요리")
    print_results("매콤한 국물 요리", results)
    
    # =====================================
    # 케이스 3: Hybrid 효과 (둘 다 활용)
    # =====================================
    results = searcher.search("두부 들어간 한식")
    print_results("두부 들어간 한식", results)
    
    # =====================================
    # 케이스 4: 메타데이터 필터 — 30분 이내 쉬운 요리
    # =====================================
    results = searcher.search(
        "간단한 요리",
        where={
            "$and": [
                {"difficulty": "easy"},
                {"time_minutes": {"$lte": 30}}
            ]
        }
    )
    print_results(
        "간단한 요리",
        results,
        where="easy + 30분 이내"
    )
    
    # =====================================
    # 케이스 5: 카테고리 필터 + 의미 검색
    # =====================================
    results = searcher.search(
        "면 요리",
        where={"category": "중식"}
    )
    print_results("면 요리 (중식만)", results, where="category=중식")
    
    # =====================================
    # 케이스 6: 복잡한 질문 — Re-ranking 효과
    # =====================================
    results = searcher.search("초보자가 도전하기 좋은 양식")
    print_results("초보자가 도전하기 좋은 양식", results)


if __name__ == "__main__":
    main()