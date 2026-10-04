# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Lê Nguyễn Quốc Bảo — 2A202603011  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

Nguồn: `reports/naive_baseline_report.json` và `reports/ragas_report.json` (20 câu hỏi, judge `gpt-4o-mini`).

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.775 | 0.829 | +0.054 |
| Answer Relevancy | 0.668 | 0.819 | **+0.151** |
| Context Precision | 0.925 | 0.950 | +0.025 |
| Context Recall | 0.900 | 0.933 | +0.033 |

**Nhận xét:**
- Cả 4 metric của production đều ≥ 0.75. Answer Relevancy tăng nhiều nhất (+0.15). Lý do: hierarchical retrieval trả về đoạn cha đầy đủ, reranker đẩy đúng đoạn lên top-3, và prompt yêu cầu trả lời ngắn, đúng số liệu.
- Faithfulness (0.829) là metric thấp nhất. 4/5 câu tệ nhất có nguyên nhân ở bước **generation**, không phải retrieval: retrieval đã tốt (precision 0.95 / recall 0.93), nhưng LLM tự suy luận hoặc tự tính toán vượt quá những gì context nói.

**Pipeline production:** hierarchical chunking (parent 2048 / child 256) → M5 enrichment (1 call/chunk) → BM25 (underthesea) + bge-m3 dense → RRF (k=60) → bge-reranker-v2-m3 → top-3 parent → `gpt-4o-mini` (temperature 0).

## Latency breakdown

| Bước | Thời gian |
|------|-----------|
| Chunking (M1, 26 docs) | 41 ms |
| Enrichment (M5, offline) | 423.0 s |
| Indexing BM25 + Dense (offline) | 57.5 s |
| Hybrid search / query | 160 ms |
| Rerank / query (CPU) | **11 343 ms** |
| LLM generation / query | 2 522 ms |

→ Ở thời điểm query, reranker chiếm khoảng 80% latency, do chạy bge-reranker-v2-m3 (568M tham số) trên CPU với 20 candidate. Đây là bottleneck cần tối ưu đầu tiên trước khi đưa lên production.

## Bottom-5 Failures

> Report hiện tại chỉ lưu metric của từng câu, **không lưu câu trả lời**. Vì vậy mục "Got" bên dưới là hành vi suy ra từ metric và context, chưa phải output nguyên văn. Fix #0 cho pipeline: lưu thêm `answer` + `contexts` của từng câu vào `ragas_report.json`.

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép (v2024); lương Senior (P3-P4) 20–35 triệu VNĐ/tháng.
- **Got (suy ra):** Câu trả lời thiếu một nửa hoặc trả lời kiểu "không tìm thấy". RAGAS chấm Answer Relevancy = 0 cho những câu trả lời không cam kết (noncommittal) như vậy.
- **Worst metric:** answer_relevancy = 0.00
- **Error Tree:** Output sai → Context đúng? **Chỉ đúng một phần**: câu hỏi multi-hop cần 2 tài liệu (`nghi_phep_nam_v2024.md` + `bang_luong_2024.md`), nhưng top-3 parent có thể bị 1 tài liệu chiếm hết → Query OK? Không: một query gộp 2 ý nên khó khớp tốt với cả hai tài liệu.
- **Root cause:** Retrieval cho câu hỏi multi-hop. Query không được tách nhỏ, cộng với giới hạn `RERANK_TOP_K=3`.
- **Suggested fix:** Query decomposition (tách thành 2 sub-query, retrieve riêng rồi gộp kết quả), hoặc nâng top-k lên 5 khi phát hiện câu hỏi chứa "và".

### #2
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Trên 50.000.000 VNĐ → Tổng Giám đốc (CEO) phê duyệt.
- **Got (suy ra):** Câu trả lời có claim không khớp với context. Faithfulness = 0 nghĩa là không claim nào được context hỗ trợ.
- **Worst metric:** faithfulness = 0.00
- **Error Tree:** Output sai → Context đúng? Không chắc. `mua_sam.md` lưu thẩm quyền dưới dạng **bảng markdown**, mà `chunk_hierarchical` cắt đoạn con theo số ký tự cố định (256), nên có thể cắt ngang dòng "Trên 50.000.000 VNĐ | CEO" → Query OK? Có (từ khóa "55 triệu" vs "50.000.000" khác cách viết số, BM25 không khớp được).
- **Root cause:** Chunking cắt ngang bảng, cộng với cách chuẩn hóa số khác nhau ("55 triệu" ↔ "55.000.000").
- **Suggested fix:** Dùng `chunk_structure_aware` cho tài liệu có bảng (giữ nguyên section), và chuẩn hóa số tiền ("X triệu" → "X.000.000") trước khi đưa vào BM25.

### #3
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; 2%/tháng × 15tr = 300.000 VNĐ/tháng (pro-rata ≈ 50.000 VNĐ).
- **Got (suy ra):** LLM tự tính ra con số cuối cùng. Phép tính đó không có nguyên văn trong context nên RAGAS coi là claim không được hỗ trợ.
- **Worst metric:** faithfulness = 0.33
- **Error Tree:** Output sai một phần → Context đúng? **Có** (`tam_ung.md`: hạn 15 ngày, phí 2%/tháng) → Query OK? Có → Lỗi nằm ở **generation / reasoning**.
- **Root cause:** Câu hỏi dạng numeric reasoning. LLM suy luận nhiều bước mà không trích dẫn căn cứ, nên faithfulness bị phạt.
- **Suggested fix:** Prompt yêu cầu ghi rõ từng bước: "Theo context: hạn 15 ngày, phí 2%/tháng → quá hạn 5 ngày → ...". Mỗi claim trung gian đều bám vào context nên verify được.

### #4
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** 16–30 ngày → CEO phê duyệt; nghỉ > 14 ngày phải tự đóng bảo hiểm.
- **Got (suy ra):** Đúng ý chính nhưng thêm hoặc trộn các mức phê duyệt khác (trưởng phòng, Giám đốc Nhân sự). Một nửa số claim không khớp.
- **Worst metric:** faithfulness = 0.50
- **Error Tree:** Output sai một phần → Context đúng? **Có** (`nghi_phep_khong_luong.md` liệt kê cả 3 bậc 1–5 / 6–15 / 16–30 ngày) → Query OK? Có → Lỗi ở **generation**: LLM không chọn đúng một bậc.
- **Root cause:** Context chứa nhiều ngưỡng số gần nhau, LLM liệt kê dư hoặc diễn giải sai ngưỡng.
- **Suggested fix:** Prompt: "Chỉ nêu mức áp dụng cho đúng giá trị trong câu hỏi". Thêm few-shot cho dạng câu hỏi ngưỡng / khoảng.

### #5
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 ngày (v2024); v2023 (12 ngày) đã bị thay thế.
- **Got (suy ra):** Câu trả lời đúng, nhưng context lẫn cả bản v2023.
- **Worst metric:** context_precision = 0.50
- **Error Tree:** Output đúng → Context đúng? **Lẫn tạp**: cả `nghi_phep_nam_v2023.md` và `v2024.md` đều được retrieve vì nội dung gần như giống hệt. Chunk v2023 lỗi thời chiếm một trong 2 vị trí đầu → Query OK? Có.
- **Root cause:** Xung đột phiên bản tài liệu (version conflict). Index không có metadata hiệu lực, nên retriever không phân biệt được bản cũ và bản mới.
- **Suggested fix:** Trích `Ngày hiệu lực` / `Phiên bản` vào metadata, rồi filter tài liệu superseded (hoặc chỉ giữ bản mới nhất cho mỗi chủ đề) trước khi rerank.

## Case Study (cho presentation)

**Question chọn phân tích:** #5 — "Nhân viên được nghỉ bao nhiêu ngày phép năm?" (version conflict, lỗi điển hình của RAG trong doanh nghiệp)

**Error Tree walkthrough:**
1. Output đúng? → Đúng (15 ngày), nhờ prompt có dòng "ưu tiên phiên bản mới nhất".
2. Context đúng? → Chỉ một nửa: chunk v2023 (12 ngày) được rank ngang chunk v2024, làm context_precision giảm còn 0.5.
3. Query rewrite OK? → Query rõ ràng, không cần rewrite.
4. Fix ở bước: **Indexing / metadata**. Gắn `effective_date` cho mỗi chunk và filter bản superseded. Không nên để prompt "chữa cháy" lỗi ở tầng retrieval.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Lưu `answer` + `contexts` của từng câu vào report để chẩn đoán dựa trên output thật thay vì suy luận.
- Giảm latency rerank 11.3s → < 1s: rerank 10 candidate thay vì 20, chạy model ở FP16/GPU, hoặc thử `FlashrankReranker`.
- Dùng structure-aware chunking cho tài liệu có bảng (mua sắm, bảng lương), sau đó chạy lại RAGAS để đo Δ faithfulness.
