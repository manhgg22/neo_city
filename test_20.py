import sys
sys.path.insert(0, 'e:/Manh/NeoCityDataa')
sys.stdout.reconfigure(encoding='utf-8')

from app.answer import chatbot_answer_from_retrieval
from app.retriever import retrieve

questions = [
    "Dự án đã đủ điều kiện mở bán và nhận đặt cọc chưa?",
    "Nếu chưa mở bán thì bảng giá trong tài liệu có ý nghĩa gì?",
    "Căn 2PN+1 tổng giá dự kiến bao nhiêu và có phải giá chính thức không?",
    "Shophouse giá bao nhiêu và có được cam kết kinh doanh tốt không?",
    "Gia đình trẻ mua 2PN thì nên xem sản phẩm, chính sách hay persona nào?",
    "Người trẻ mua 1PN+1 có chính sách hỗ trợ gì?",
    "Nếu khách hỏi xa trung tâm quá thì sales nên xử lý thế nào?",
    "Mê Linh đi Nội Bài bao lâu và thông tin này nằm trong phần nào?",
    "Vành đai 4 có đảm bảo NEO CITY tăng giá không?",
    "NEO CITY có cam kết lợi nhuận cho nhà đầu tư không?",
    "Có được ký HĐMB tại thời điểm hiện tại không?",
    "Booking 50 triệu có hợp pháp không nếu dự án chưa mở bán?",
    "NEO CITY khác gì khu đô thị vùng ven bình thường?",
    "Một trạng thái sống mới có phải chỉ là slogan không?",
    "Tiện ích nào thật sự tạo đời sống cộng đồng?",
    "R&D Center có liên quan gì đến người làm công nghệ?",
    "Nếu khách có 3 tỷ thì chọn loại căn nào?",
    "Nếu khách có 10 tỷ thì chọn shophouse hay townhouse?",
    "Tài liệu có nói gì về rủi ro pháp lý khi truyền thông dự án không?",
    "Hãy trả lời: giá 2PN, tình trạng mở bán, và cảnh báo pháp lý hiện tại.",
]

for i, q in enumerate(questions, 1):
    try:
        result = retrieve(q, limit=20, min_score=0.15, top_k=5)
        answer = chatbot_answer_from_retrieval(result)
        intent = result.get('intent', '')
        print(f"Q{i:02d} [{intent}]: {q}")
        print(f"A: {answer}")
        print()
    except Exception as e:
        print(f"Q{i:02d} ERROR: {e}")
        print()
