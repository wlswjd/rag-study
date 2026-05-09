# 다양성:
#  - 카테고리: 한식 4, 양식 5, 중식 3, 일식 2, 기타 1
#  - 난이도: easy 5, medium 6, hard 3
#  - 시간: 10~90분 분포

# 테스트 가능한 케이스:
#  - 정확한 이름 매칭: "까르보나라" (BM25 강함)
#  - 의미 검색: "매운 국물 요리" → 김치찌개, 마라탕 (Dense 강함)
#  - 메타데이터 필터: "30분 이내", "쉬운 요리"
#  - Re-ranking: 비슷한 후보 중 정밀 선별

RECIPES = [
    {
        "id": "recipe_01",
        "name": "김치찌개",
        "category": "한식",
        "difficulty": "easy",
        "time_minutes": 30,
        "main_ingredient": "김치",
        "description": "김치찌개는 잘 익은 신김치와 돼지고기로 끓이는 한국의 대표적인 찌개 요리입니다. 돼지고기는 목살이나 삼겹살이 적당하며, 두부와 대파를 추가하면 더 풍부한 맛이 납니다. 매콤하고 깊은 맛의 국물 요리로 초보자도 쉽게 만들 수 있습니다."
    },
    {
        "id": "recipe_02",
        "name": "된장찌개",
        "category": "한식",
        "difficulty": "easy",
        "time_minutes": 25,
        "main_ingredient": "된장",
        "description": "된장찌개는 구수한 된장 베이스에 두부, 호박, 양파, 청양고추를 넣고 끓이는 한식 국물 요리입니다. 멸치 다시마 육수를 사용하면 깊은 맛이 납니다. 매일 먹어도 질리지 않는 한국인의 소울푸드입니다."
    },
    {
        "id": "recipe_03",
        "name": "비빔밥",
        "category": "한식",
        "difficulty": "medium",
        "time_minutes": 40,
        "main_ingredient": "밥",
        "description": "비빔밥은 흰 밥 위에 시금치, 콩나물, 고사리 등 여러 가지 나물과 고기, 계란 프라이를 올리고 고추장으로 비벼 먹는 요리입니다. 다양한 재료의 영양과 색감이 조화로워 한식의 대표 요리로 꼽힙니다."
    },
    {
        "id": "recipe_04",
        "name": "파스타 까르보나라",
        "category": "양식",
        "difficulty": "medium",
        "time_minutes": 25,
        "main_ingredient": "파스타",
        "description": "까르보나라는 베이컨, 계란 노른자, 파마산 치즈, 후추로 만드는 이탈리아식 크림 파스타입니다. 진짜 까르보나라에는 생크림이 들어가지 않으며, 계란과 치즈만으로 부드러운 소스를 만듭니다. 빠르고 풍부한 맛의 양식 요리입니다."
    },
    {
        "id": "recipe_05",
        "name": "토마토 파스타",
        "category": "양식",
        "difficulty": "easy",
        "time_minutes": 20,
        "main_ingredient": "파스타",
        "description": "토마토 파스타는 토마토 소스와 마늘, 양파, 바질로 만드는 가장 기본적인 양식 파스타입니다. 캔 토마토를 사용하면 빠르게 만들 수 있고, 바질을 마지막에 넣어 향을 살립니다. 누구나 좋아하는 클래식 요리입니다."
    },
    {
        "id": "recipe_06",
        "name": "마라탕",
        "category": "중식",
        "difficulty": "medium",
        "time_minutes": 35,
        "main_ingredient": "마라소스",
        "description": "마라탕은 사천식 매운 국물 요리로, 마라 소스와 다양한 재료를 끓여 만듭니다. 청경채, 분당면, 새우, 소고기, 두부피 등을 골라 넣을 수 있습니다. 매운맛과 마비되는 듯한 얼얼한 맛이 특징입니다."
    },
    {
        "id": "recipe_07",
        "name": "탕수육",
        "category": "중식",
        "difficulty": "hard",
        "time_minutes": 60,
        "main_ingredient": "돼지고기",
        "description": "탕수육은 돼지고기에 튀김옷을 입혀 두 번 튀기고 새콤달콤한 소스를 부어 먹는 중식 요리입니다. 바삭한 튀김과 달콤한 소스의 조화가 핵심이며, 부먹/찍먹 논쟁의 주인공이기도 합니다. 손이 많이 가는 요리입니다."
    },
    {
        "id": "recipe_08",
        "name": "초밥",
        "category": "일식",
        "difficulty": "hard",
        "time_minutes": 90,
        "main_ingredient": "회",
        "description": "초밥은 식초로 간한 밥 위에 신선한 회를 올려 먹는 일본 요리입니다. 참치, 연어, 광어 등 다양한 회를 사용하며, 와사비와 간장에 살짝 찍어 먹습니다. 신선한 재료가 가장 중요한 요리입니다."
    },
    {
        "id": "recipe_09",
        "name": "라멘",
        "category": "일식",
        "difficulty": "medium",
        "time_minutes": 45,
        "main_ingredient": "면",
        "description": "라멘은 진한 육수에 면을 넣고 차슈, 반숙란, 파, 김 등을 토핑으로 올려 먹는 일본식 면 요리입니다. 돈코츠, 쇼유, 미소 등 다양한 베이스가 있으며, 면의 식감과 국물 조화가 핵심입니다."
    },
    {
        "id": "recipe_10",
        "name": "샐러드",
        "category": "양식",
        "difficulty": "easy",
        "time_minutes": 10,
        "main_ingredient": "채소",
        "description": "샐러드는 신선한 채소에 드레싱을 곁들여 먹는 가벼운 요리입니다. 양상추, 토마토, 오이, 당근 등을 기본으로 하고 견과류나 치즈를 더하면 풍성해집니다. 다이어트에도 좋고 빠르게 준비할 수 있습니다."
    },
    {
        "id": "recipe_11",
        "name": "계란말이",
        "category": "한식",
        "difficulty": "easy",
        "time_minutes": 15,
        "main_ingredient": "계란",
        "description": "계란말이는 계란을 풀어 얇게 부치며 말아내는 반찬입니다. 당근, 파, 햄을 잘게 썰어 넣으면 색감과 맛이 좋아집니다. 도시락 반찬이나 술안주로도 좋은 만능 요리입니다."
    },
    {
        "id": "recipe_12",
        "name": "스테이크",
        "category": "양식",
        "difficulty": "medium",
        "time_minutes": 30,
        "main_ingredient": "소고기",
        "description": "스테이크는 두툼한 소고기를 강한 불에 구워 겉은 바삭하고 속은 부드럽게 익히는 양식 요리입니다. 소금과 후추로만 간하고 버터, 마늘, 로즈마리로 향을 더합니다. 굽기 정도(레어, 미디엄, 웰던)에 따라 식감이 달라집니다."
    },
    {
        "id": "recipe_13",
        "name": "짜장면",
        "category": "중식",
        "difficulty": "medium",
        "time_minutes": 35,
        "main_ingredient": "면",
        "description": "짜장면은 춘장에 양파, 돼지고기, 감자를 볶아 만든 짜장 소스를 면 위에 부어 먹는 한국식 중화요리입니다. 단맛과 짠맛이 조화로우며 한국인이 가장 사랑하는 배달 음식 중 하나입니다."
    },
    {
        "id": "recipe_14",
        "name": "리조또",
        "category": "양식",
        "difficulty": "hard",
        "time_minutes": 50,
        "main_ingredient": "쌀",
        "description": "리조또는 이탈리아의 쌀 요리로, 쌀을 볶다가 육수를 조금씩 부어가며 천천히 끓이는 방식으로 만듭니다. 버섯, 새우, 시금치 등 다양한 재료로 변형 가능하며, 크리미한 식감이 특징입니다. 약 50분간 계속 저어야 하는 인내심이 필요한 요리입니다."
    },
    {
        "id": "recipe_15",
        "name": "샌드위치",
        "category": "양식",
        "difficulty": "easy",
        "time_minutes": 10,
        "main_ingredient": "빵",
        "description": "샌드위치는 빵 사이에 햄, 치즈, 채소 등을 넣어 만드는 간단한 양식 요리입니다. 마요네즈와 머스타드를 발라 풍미를 더하고 다양한 속재료로 변형 가능합니다. 아침 식사나 도시락으로 인기 있습니다."
    },
]