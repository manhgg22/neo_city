from app.retriever import retrieve

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

for q in queries:
    print("\n" + "=" * 80)
    print("QUERY:", q)
    results = retrieve(q, limit=20, min_score=0.15, top_k=5)
    for i, r in enumerate(results, 1):
        print(
            i,
            r.get("id"),
            "section=", r.get("section"),
            "topic=", r.get("topic"),
            "score=", r.get("score"),
            "rerank_score=", r.get("rerank_score"),
        )
