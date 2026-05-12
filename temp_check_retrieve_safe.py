from app.retriever import retrieve
from pprint import pprint

queries = [
    "Có được đặt cọc mua NEO CITY chưa?",
    "Dự án đã mở bán chưa?",
    "Căn 2PN giá bao nhiêu?",
    "Shophouse giá bao nhiêu?",
    "Gia đình trẻ phù hợp sản phẩm nào?",
    "NEO CITY có tiện ích gì?",
    "Khách hàng mục tiêu của NEO CITY là ai?",
    "Mê Linh kết nối sân bay Nội Bài thế nào?",
    "NEO CITY là dự án gì?",
]

def extract_chunks(result):
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("chunks", "results", "retrieved_chunks", "data"):
            value = result.get(key)
            if isinstance(value, list):
                return value
        print("DICT KEYS:", list(result.keys()))
        return []
    print("UNKNOWN RESULT TYPE:", type(result))
    pprint(result)
    return []

for q in queries:
    print("\n" + "=" * 80)
    print("QUERY:", q)

    result = retrieve(q, limit=20, min_score=0.15, top_k=5)
    chunks = extract_chunks(result)

    print("RESULT TYPE:", type(result))
    print("CHUNK COUNT:", len(chunks))

    for i, r in enumerate(chunks, 1):
        if not isinstance(r, dict):
            print(i, "NON-DICT:", type(r), r)
            continue

        print(
            i,
            r.get("id"),
            "section=", r.get("section"),
            "topic=", r.get("topic"),
            "score=", r.get("score"),
            "rerank_score=", r.get("rerank_score"),
        )
