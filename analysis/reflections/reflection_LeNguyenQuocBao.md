# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Lê Nguyễn Quốc Bảo — 2A202603011  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 (all-MiniLM-L6-v2) tạo **208 chunks, trung bình 99 ký tự**, so với basic là **51 chunks, trung bình 410 ký tự**. Với tiếng Việt, MiniLM (model tiếng Anh) cho similarity thấp giữa các câu liền kề, nên cắt quá vụn (chunk nhỏ nhất chỉ 6 ký tự). Vì vậy pipeline chọn hierarchical. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | **87 children (≤256 ký tự)** dùng để retrieve chính xác, sau đó trả về parent (≤2048 ký tự) cho LLM. Đây là lý do context_recall đạt 0.933. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 sections, giữ nguyên header trong `metadata["section"]`. Phù hợp với tài liệu có bảng (xem failure #2). |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 (đã tách từ bằng underthesea) bắt được từ khóa chính xác như "MFA", "PVI"; bge-m3 bắt được các câu hỏi diễn đạt khác đi. RRF (k=60) chỉ dùng thứ hạng nên không cần chuẩn hóa thang điểm của 2 retriever. Search chỉ mất **162 ms/query**. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 rerank 20 → top-3, đưa context_precision lên **0.983** (baseline 0.933). Đổi lại latency **11.35 s/query trên CPU**, chiếm khoảng 83% thời gian xử lý một query. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production: F 0.879 / AR 0.824 / CP 0.983 / CR 0.933, cả 4 đều ≥ 0.75. Lỗi còn lại tập trung ở các câu numeric: LLM tính sai (tạm ứng → 600k), hoặc phép tính đúng nhưng judge không verify được (lương thử việc 17tr). |
| Contextual embeddings | M5 | `_enrich_single_call()` | 1 call/chunk trả về JSON gồm summary + 3 câu hỏi HyQA + câu context + metadata. Câu context và câu hỏi được nối vào `enriched_text` trước khi index. Tốn 984 s ở bước offline, đổi lại Answer Relevancy +0.10 so với baseline (0.722 → 0.824). |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

**1. Gọi LLM thất bại dù đã điền API key**
- Error: `openai.APIConnectionError: Connection error.`
- Debug: gọi thẳng bằng `curl` thì endpoint trả 401, tức là mạng vẫn thông. Tiếp đó xem `repr(e.__cause__)` thì ra `UnsupportedProtocol("Request URL is missing an 'http://' or 'https://' protocol.")`. Kiểm tra `env` thì thấy shell đang có sẵn `OPENAI_API_KEY=""` và `OPENAI_BASE_URL=""`.
- Root cause: `load_dotenv()` mặc định **không ghi đè** biến môi trường đã tồn tại, kể cả khi biến đó rỗng.
- Fix: `load_dotenv(override=True)` trong `config.py`. Ngoài ra cấu hình `OPENAI_BASE_URL` (cho OpenAI SDK) và `OPENAI_API_BASE` (cho langchain-openai mà RAGAS 0.1 dùng) để trỏ tới endpoint tương thích OpenAI.

**2. Test M5 fail khi có key thật**
- Error: `AssertionError: assert 174 <= (65 * 2)` trong `test_summarize_shorter_than_original`.
- Root cause: prompt "Summarize in 2-3 sentences" khiến LLM viết dài hơn câu gốc và tự thêm ý không có trong văn bản (một dạng hallucination ngay ở bước enrichment).
- Fix: viết lại prompt thành "ngắn hơn bản gốc, giữ nguyên số liệu, không thêm thông tin", kèm chốt chặn `summary if len(summary) <= len(text) else text`.

**3. Pytest bị treo / thiếu thư viện**
- `No module named pytest`, `ModuleNotFoundError: No module named 'ragas'` → cài lại `requirements.txt` vào `.venv`.
- Test treo khi `SentenceTransformer` tải model từ Hugging Face → tải trước 3 model (qua `HF_ENDPOINT` mirror), sau đó chạy với `HF_HUB_OFFLINE=1`.

**4. Điểm RAGAS có thể bị NaN**
- RAGAS trả NaN cho một số câu (ví dụ judge parse lỗi), làm `sorted()`/`min()` trong `failure_analysis` cho kết quả sai. Fix: đổi NaN thành 0.0 trước khi phân tích.

**5. Answer Relevancy = 0 cho câu trả lời đúng**
- Triệu chứng: 13/20 câu có AR = 0, kể cả câu trả lời đúng hoàn toàn ("Phụ cấp ăn trưa 1.000.000 VNĐ/tháng").
- Error trong log: `Exception raised in Job[9]: BadRequestError(Error code: 400 - ... 'Invalid request, please check your parameters.')`, lặp lại ở các job 13, 17, 21, ... Các job lỗi cách nhau đúng 4, tức là luôn cùng một metric trong 4 metric.
- Root cause: `answer_relevancy` mặc định `strictness=3`, tức gọi API với `n=3`, nhưng proxy không hỗ trợ `n>1`. RAGAS ghi lần lỗi là NaN, và chính bước đổi NaN thành 0 tôi thêm vào M4 đã che mất lỗi này.
- Fix: `answer_relevancy.strictness = 1`, chạy lại cả baseline lẫn production → 0 exception. Bài học: một metric bằng 0 cần được nghi là lỗi đo trước khi kết luận pipeline kém.

**Kiến thức còn thiếu & cách bổ sung:** cách RAGAS tách claim để tính faithfulness (đọc source `ragas.metrics._faithfulness`), và đánh đổi giữa latency và độ chính xác của cross-encoder (benchmark bằng `benchmark_reranker()`).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý hỏi đáp chính sách nội bộ (HR / IT policy chatbot)

#### 1. Hiện trạng
- **Pipeline hiện tại:** chunk cố định 500 ký tự → dense search (1 embedding model) → top-5 → LLM.
- **Vấn đề:** trả lời theo chính sách cũ khi có nhiều phiên bản; tài liệu dạng bảng bị cắt vụn; chưa có bộ đo chất lượng định lượng.

#### 2. Kế hoạch cải tiến
1. **Chunking:** hierarchical (child 256 / parent 2048) làm mặc định; structure-aware cho tài liệu có bảng hoặc header. Lý do: lab cho recall 0.933, còn bảng bị cắt là nguyên nhân của failure #2.
2. **Search:** hybrid BM25 (underthesea) + bge-m3 + RRF, vì người dùng hay hỏi bằng mã/từ khóa chính xác (tên form, mã cấp bậc P3-P4).
3. **Reranking:** có dùng, bge-reranker-v2-m3 nhưng chỉ rerank top-10 và chạy trên GPU/FP16, mục tiêu < 500 ms. Nếu phải chạy CPU thì dùng Flashrank.
4. **Evaluation:** RAGAS 4 metrics trên golden set 50 câu (đủ 6 loại: lookup, version, negation, multi-hop, numeric, ambiguous), chạy trong CI mỗi khi đổi prompt/chunking; report lưu kèm answer + contexts.
5. **Enrichment:** combined single-call (contextual prepend + HyQA) cộng với metadata `effective_date` / `version` để filter tài liệu superseded.

#### 3. Timeline triển khai
- **Tuần 1:** Xây golden set 50 câu và chạy RAGAS cho pipeline hiện tại để lấy baseline.
- **Tuần 2:** Chuyển sang hierarchical + structure-aware chunking và metadata phiên bản, đo lại.
- **Tuần 3:** Hybrid search + RRF, thêm reranker, tối ưu latency (top-10, GPU).
- **Tuần 4:** Enrichment combined mode, query decomposition cho câu multi-hop, đưa RAGAS vào CI với ngưỡng ≥ 0.75 cho cả 4 metric.
