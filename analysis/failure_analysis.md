# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Lê Nguyễn Quốc Bảo — 2A202603011  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

Nguồn: `reports/naive_baseline_report.json` và `reports/ragas_report.json` (20 câu hỏi, judge `gpt-4o-mini`, `answer_relevancy.strictness = 1`).

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.846 | **0.879** | +0.033 |
| Answer Relevancy | 0.722 | **0.824** | **+0.102** |
| Context Precision | 0.933 | **0.983** | +0.050 |
| Context Recall | 0.900 | **0.933** | +0.033 |

**Nhận xét:**
- Production tốt hơn baseline ở cả 4 metric, và cả 4 đều ≥ 0.75 (Faithfulness ≥ 0.85).
- Answer Relevancy tăng nhiều nhất (+0.10). Lý do: hierarchical retrieval trả về đoạn cha đầy đủ, reranker đưa đúng đoạn lên top-3, và prompt yêu cầu trả lời ngắn, đúng số liệu.
- Context Precision đạt 0.983: reranker gần như luôn đặt chunk đúng ở vị trí #1. Vấn đề còn lại nằm ở **vị trí #2–#3** (nhiễu, hoặc bản chính sách cũ) và ở **bước generation** (các câu numeric reasoning).

**Pipeline production:** hierarchical chunking (parent 2048 / child 256) → M5 enrichment (1 call/chunk) → BM25 (underthesea) + bge-m3 dense → RRF (k=60) → bge-reranker-v2-m3 → top-3 parent → `gpt-4o-mini` (temperature 0).

> **Lưu ý về đo lường:** ở lần chạy đầu, 13/20 câu bị Answer Relevancy = 0 dù câu trả lời đúng. Metric này mặc định gọi LLM với `n=3`, API proxy trả `400 Invalid request`, RAGAS ghi các lần lỗi đó là NaN, rồi `evaluate_ragas` đổi NaN thành 0. Sau khi đặt `strictness=1`, cả baseline và production đều được chấm lại không còn lỗi nào. Bài học: phải kiểm tra log của evaluator trước khi tin vào metric.

## Latency breakdown

| Bước | Thời gian |
|------|-----------|
| Chunking (M1, 26 docs) | 167 ms |
| Enrichment (M5, offline, 1 call/chunk) | 983.9 s |
| Indexing BM25 + Dense (offline) | 63.9 s |
| Hybrid search / query | 162 ms |
| Rerank / query (CPU, 20 candidates) | **11 350 ms** |
| LLM generation / query | 2 021 ms |

→ Ở thời điểm query, reranker chiếm khoảng 83% latency, do chạy bge-reranker-v2-m3 (568M tham số) trên CPU. Đây là bottleneck cần tối ưu đầu tiên.

## Bottom-5 Failures

Số liệu và câu trả lời lấy nguyên văn từ `per_question` trong `reports/ragas_report.json`.

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép (v2024); lương Senior (P3-P4) 20–35 triệu VNĐ/tháng.
- **Got:** "Nhân viên có 9 năm thâm niên sẽ được nghỉ **18 ngày phép năm** (15 ngày cơ bản + 3 ngày cộng thêm). Lương sẽ được tính theo quy định của công ty, không có thông tin cụ thể về mức lương trong ngữ cảnh này."
- **Worst metric:** answer_relevancy = 0.00 (F 0.75, CP 1.0, CR 0.5)
- **Error Tree:** Output sai một nửa → Context đúng? **Thiếu**: top-3 gồm nghỉ phép v2024, nghỉ phép **v2023** (bản cũ) và nghỉ phép không lương; không có `bang_luong_2024.md` → Query OK? **Không**: một query gộp 2 ý (phép + lương), nên tài liệu nghỉ phép chiếm cả 3 slot.
- **Root cause:** Câu hỏi multi-hop nhưng retrieve chỉ bằng một query. Bản v2023 lỗi thời còn chiếm mất một slot. Câu "không có thông tin..." bị RAGAS coi là noncommittal, nên AR = 0.
- **Suggested fix:** Query decomposition ("Senior 9 năm được bao nhiêu ngày phép?" + "Lương Senior bao nhiêu?"), retrieve riêng rồi gộp. Thêm filter bỏ tài liệu superseded (v2023).

### #2
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; 2%/tháng × 15tr = 300.000 VNĐ/tháng (pro-rata ≈ 50.000 VNĐ).
- **Got:** "Nhân viên sẽ bị phạt 600.000 VNĐ. (2% của 15.000.000 VNĐ là 300.000 VNĐ/tháng, sau 20 ngày tương đương 1 tháng)."
- **Worst metric:** faithfulness = 0.33 (AR 0.80, CP 1.0, CR 0.67)
- **Error Tree:** Output **sai** → Context đúng? **Có** (`tam_ung.md`: hạn 15 ngày, phí 2%/tháng) → Query OK? Có → Lỗi ở **generation / reasoning**: LLM tính sai (600k) và tự mâu thuẫn ngay trong câu trả lời (300k/tháng × "1 tháng" = 600k).
- **Root cause:** LLM tính toán nhiều bước mà không có bước trung gian. Nó bỏ qua 15 ngày miễn phạt và áp phí cho toàn bộ 20 ngày.
- **Suggested fix:** Prompt chain-of-thought có cấu trúc: (1) trích nguyên văn quy định, (2) tính số ngày quá hạn, (3) áp công thức. Hoặc giao phần tính cho tool/calculator thay vì để LLM tự tính.

### #3
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** 5–50 triệu → Giám đốc phòng ban (Director); cần xác nhận cấu hình từ CNTT; > 10 triệu cần ít nhất 3 báo giá.
- **Got:** "Giám đốc phòng ban (Director) sẽ phê duyệt đơn hàng 30 triệu VNĐ. Cần có xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất."
- **Worst metric:** faithfulness = 0.50 (AR 0.80, CP 1.0, CR 1.0)
- **Error Tree:** Output **đúng nhưng thiếu** (không nhắc 3 báo giá) → Context đúng? **Có** (`mua_sam.md` ở top-1) → Query OK? Có → Lỗi ở **generation**, cộng với giới hạn của metric: claim "Director phê duyệt đơn 30 triệu" là suy luận từ ngưỡng 5–50 triệu, không có nguyên văn trong context, nên judge chấm là không được hỗ trợ.
- **Root cause:** Câu trả lời không bám vào câu chữ trong context: LLM nêu kết luận mà không dẫn ngưỡng làm căn cứ. Thêm vào đó, prompt "trả lời ngắn gọn" khiến LLM bỏ sót yêu cầu 3 báo giá.
- **Suggested fix:** Prompt yêu cầu dẫn nguyên văn quy định trước khi kết luận ("Theo quy định: 5–50 triệu → Director, nên…") và liệt kê **mọi** điều kiện áp dụng.

### #4
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** "Lương thử việc của nhân viên Junior mức cao nhất là 17.000.000 VNĐ (85% của 20.000.000 VNĐ)."
- **Worst metric:** faithfulness = 0.50 (AR 0.83, CP 1.0, CR 1.0)
- **Error Tree:** Output **đúng** → Context đúng? **Có** (bảng lương + chính sách thử việc ở top-2) → Query OK? Có → **Không có lỗi pipeline**. Điểm thấp là **false negative của metric**: con số 17.000.000 là kết quả tính toán, không xuất hiện nguyên văn trong context.
- **Root cause:** Faithfulness của RAGAS chấm theo từng claim. Claim phái sinh từ phép tính đúng vẫn bị coi là "không được hỗ trợ".
- **Suggested fix:** Không sửa pipeline. Trong câu trả lời, ghi rõ hai dữ kiện gốc ("Junior tối đa 20.000.000; thử việc hưởng 85%") để judge verify được từng bước. Với các câu numeric, đánh giá bổ sung bằng metric exact-match so với ground truth.

### #5
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 (bắt buộc MFA cho email, VPN, hệ thống nội bộ); bản v1.0 cũ không yêu cầu MFA.
- **Got:** "Có, tất cả nhân viên **bắt buộc** kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ."
- **Worst metric:** context_recall = 0.50 (F 1.0, AR 0.85, CP 1.0)
- **Error Tree:** Output **đúng** → Context đúng? **Thiếu**: top-1 là `mat_khau_v2.md`, nhưng slot 2–3 là mua sắm và phân loại dữ liệu (nhiễu). `mat_khau_v1.md` không được retrieve nên không đủ căn cứ cho ý "bản cũ không yêu cầu MFA" → Query OK? Có.
- **Root cause:** Reranker chỉ giữ 1 chunk liên quan, 2 slot còn lại bị lấp bằng nhiễu, vì không có ngưỡng điểm tối thiểu.
- **Suggested fix:** Bỏ các chunk có `rerank_score` dưới ngưỡng thay vì luôn lấy đủ top-3. Với câu hỏi về chính sách, retrieve thêm theo metadata `topic` để lấy cả bản cũ và bản mới khi câu hỏi cần so sánh phiên bản.

## Case Study (cho presentation)

**Question chọn phân tích:** #2 — "Tạm ứng 15 triệu, 20 ngày mới thanh toán, bị phạt bao nhiêu?". Đây là case duy nhất trong bottom-5 mà **retrieval hoàn hảo nhưng câu trả lời sai về nội dung**.

**Error Tree walkthrough:**
1. Output đúng? → **Sai**: 600.000 VNĐ, trong khi đáp án đúng là 300.000 VNĐ/tháng (≈ 50.000 VNĐ pro-rata).
2. Context đúng? → **Đúng**: `tam_ung.md` ở top-1, chứa đủ "15 ngày" và "2%/tháng" (CP = 1.0).
3. Query rewrite OK? → Query rõ ràng, không cần rewrite.
4. Fix ở bước: **Generation**. Thêm reasoning có cấu trúc (trích quy định → số ngày quá hạn → công thức), hoặc dùng tool calculator. Thêm retrieval hay rerank không giúp được case này.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Query decomposition cho câu multi-hop (#1), cộng với filter tài liệu superseded theo `Ngày hiệu lực`.
- Prompt "trích dẫn rồi mới kết luận" cho câu numeric (#2, #3, #4); kỳ vọng faithfulness > 0.9.
- Giảm latency rerank từ 11.3 s xuống < 1 s: rerank 10 candidate, FP16/GPU, hoặc `FlashrankReranker`. Thêm ngưỡng `rerank_score` để bỏ chunk nhiễu (#5).
