# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Ngô Anh Khoa

**Nhóm:** THE LIEMS

**Ngày:** 20/09/2026

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine

Độ tương tự cosine cao nghĩa là hai vector có hướng gần nhau, tức hai đoạn văn có nội dung/ngữ nghĩa gần nhau trong không gian embedding.

**Ví dụ có độ tương tự cao**

- Câu A: “Người bán phải phản hồi trong 02 ngày lịch.”
- Câu B: “Nhà bán hàng có hai ngày để trả lời khiếu nại.”
- Hai câu diễn đạt cùng một quy định bằng từ ngữ khác nhau.

**Ví dụ có độ tương tự thấp**

- Câu A: “Shopee xử phạt đơn hàng ảo đến 10 triệu đồng.”
- Câu B: “Hôm nay thời tiết tại Hà Nội có mưa.”
- Hai câu khác chủ đề và không chia sẻ cùng ý định.

Cosine được ưu tiên hơn khoảng cách Euclid vì tập trung vào hướng của vector, ít bị ảnh hưởng bởi độ lớn vector hoặc độ dài văn bản. Điều này phù hợp với mục tiêu so sánh ngữ nghĩa của text embeddings.

### Bài toán Chunking

Với tài liệu 10.000 ký tự, `chunk_size=500`, `overlap=50`, bước trượt là `500 - 50 = 450`. Số chunk là `ceil((10000 - 500) / 450) + 1 = 23`.

Nếu tăng overlap lên 100 thì bước trượt còn 400 và số chunk là `ceil(9500 / 400) + 1 = 25`. Overlap lớn hơn giúp giữ ngữ cảnh ở biên chunk nhưng tăng số vector, chi phí embedding và khả năng sinh kết quả gần trùng nhau.

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ

**`SentenceChunker.chunk`:** Dùng regex `(?<=[.!?])(?:[ \t]+|\r?\n+)` để tách sau dấu kết thúc câu nhưng giữ dấu câu trong nội dung. Hàm bỏ đoạn rỗng, chuẩn hóa khoảng trắng và gom tối đa số câu đã cấu hình vào mỗi chunk.

**`RecursiveChunker.chunk` / `_split`:** Thử lần lượt các separator `\n\n`, `\n`, `. `, khoảng trắng và cuối cùng cắt cứng. Base case là đoạn không vượt `chunk_size`; các mảnh nhỏ kề nhau được ghép lại để tránh tạo quá nhiều chunk vụn.

**`HeadingChunker.chunk` (phần cá nhân):** Tách Markdown theo heading, duy trì stack heading cha/con và gắn đường dẫn heading vào từng chunk để giữ ngữ cảnh. Section quá dài được giao cho `RecursiveChunker` với ngân sách còn lại, bảo đảm mỗi chunk không quá 400 ký tự.

### Lớp EmbeddingStore

**`add_documents` + `search`:** Mỗi `Document` được embedding một lần và lưu cùng nội dung, metadata, id và thứ tự thêm. Khi tìm kiếm, query được embedding rồi tính dot product với các vector đã chuẩn hóa, sắp giảm dần và trả về `top_k`.

**`search_with_filter` + `delete_document`:** Lọc metadata trước khi tính similarity để ứng viên không đúng audience không chiếm Top-K. Xóa tất cả record có `metadata.doc_id` tương ứng và trả về `True` chỉ khi thực sự có dữ liệu bị xóa.

### Tác tử KnowledgeBaseAgent

`answer` truy xuất Top-K chunk rồi tạo prompt gồm nguồn đánh số `[1]`, `[2]`, nội dung context và câu hỏi. Prompt yêu cầu chỉ dùng context, nêu rõ khi thiếu thông tin và trích dẫn; `answer_with_filter` bảo đảm Agent dùng đúng tập kết quả đã lọc.

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Kết quả chạy bằng Python 3.11:

```text
============================= test session starts =============================
collected 48 items
tests/test_solution.py ..........................................        [ 87%]
tests/test_heading_chunker.py ......                              [100%]
============================= 48 passed =======================================
```

**Số bài test vượt qua:** 42/42 bài chính thức, cộng 6/6 bài riêng cho `HeadingChunker` (tổng 48/48).

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Đo bằng `text-embedding-3-small` và cosine similarity:

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|---|---|---|---|---:|---|
| 1 | Người mua được trả hàng trong 15 ngày. | Thời hạn yêu cầu hoàn tiền của khách hàng là mười lăm ngày. | Cao | 0.5676 | Có |
| 2 | Người bán phải phản hồi trong 02 ngày lịch. | Nhà bán hàng có hai ngày để trả lời khiếu nại. | Cao | 0.6546 | Có |
| 3 | Sản phẩm lỗi kỹ thuật được bảo hành miễn phí. | Hàng hỏng do nhà sản xuất đủ điều kiện bảo hành không mất phí. | Cao | 0.6802 | Có |
| 4 | Shopee xử phạt đơn hàng ảo đến 10 triệu đồng. | Hôm nay thời tiết tại Hà Nội có mưa. | Thấp | 0.2115 | Có |
| 5 | Hàng hóa phải còn ít nhất 30% hạn sử dụng. | Người mua có thể đổi địa chỉ giao hàng. | Thấp | 0.3495 | Có |

Cặp 5 cao hơn cặp 4 dù khác ý định vì vẫn cùng miền thương mại điện tử và cùng nói về hàng hóa/người mua. Embedding biểu diễn nhiều lớp liên hệ ngữ nghĩa, nên cùng miền từ vựng vẫn có thể làm điểm nền tăng lên.

## 5. Kết quả truy xuất của tôi (HeadingChunker) — Cá nhân (10 điểm)

Backend: OpenAI `text-embedding-3-small`; Agent: `gpt-4o-mini`; 645 chunks; trung bình 327,4 ký tự; tối đa 400.

| # | Câu hỏi | Top-1 Chunk (tóm tắt) | Score | Relevant? | Agent (tóm tắt) |
|---|---|---|---:|---|---|
| 1 | Thời hạn người mua yêu cầu trả hàng? | Mục 3.2: 15 ngày, riêng hàng tươi sống 24 giờ | 0.7324 | Có | Đúng đủ, có `[1]` |
| 2 | Thời hạn người bán phản hồi khiếu nại? | Mục 5: phản hồi trong 02 ngày lịch | 0.6257 | Có | Đúng đủ, có `[1]`, `[2]` |
| 3 | Mức bồi thường đơn hàng ảo? | Đúng tài liệu nhưng chunk Top-1 không chứa 10.000.000 VND | 0.7259 | Không theo key fact | Agent báo thiếu thông tin |
| 4 | Hàng hóa phải còn bao nhiêu hạn sử dụng? | Mục D.2: ít nhất 30% và 30 ngày | 0.8179 | Có | Đúng đủ, có `[1]` |
| 5 | Điều kiện bảo hành miễn phí? | Mục 1: điều kiện bảo hành | 0.7941 | Có | Đúng 3/4 ý, thiếu tem bảo hành |

**Kết quả:** 7/10; Hit@1 = 4/5 (80%); Hit@3 = 4/5 (80%); MRR = 0.8000.

Điều học được từ so sánh nhóm là giữ heading giúp chunk dễ hiểu và xếp hạng tốt ở các câu bám sát cấu trúc mục, nhưng lặp lại heading ở nhiều chunk có thể làm các đoạn cùng section cạnh tranh lẫn nhau. Sentence chunking đạt 9/10 vì giữ nguyên mệnh đề chứa con số tốt hơn trên corpus này, dù đôi khi tạo chunk dài quá 400 ký tự.

## Tự Đánh Giá

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động | 5/5 |
| Hướng tiếp cận | 10/10 |
| Hoàn thiện code | 30/30 |
| Dự đoán độ tương tự | 5/5 |
| Kết quả truy xuất HeadingChunker | 7/10 |
| **Tổng phần cá nhân** | **57/60** |
