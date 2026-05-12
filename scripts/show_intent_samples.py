from app.intent_classifier import classify

samples = [
    "NEO CITY la du an gi?",
    "Can 2PN gia bao nhieu?",
    "Du an da mo ban chua?",
    "Gia dinh tre phu hop san pham nao?",
    "Vi tri du an cach trung tam Hanoi bao xa?",
    "Chinh sach vay ngan hang nhu the nao?",
    "Co the dat coc ngay khong?",
    "Loi nhuan dau tu co dam bao khong?",
    "NEO Square la gi?",
    "Thi truong bat dong san Me Linh the nao?",
]

for q in samples:
    r = classify(q)
    print(f"Q: {q}")
    print(f"   intent={r.intent}  risk={r.risk_level}  sections={r.target_sections}  legal_only={r.must_use_legal_only}")
    print()