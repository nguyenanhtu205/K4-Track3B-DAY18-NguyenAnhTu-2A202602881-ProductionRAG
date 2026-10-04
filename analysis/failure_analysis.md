# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Anh Tú  
**Khóa:** K4 - Track 3B

## Lưu ý về lần chạy đánh giá

Pipeline hoàn tất end-to-end trong 118 giây: 100 child chunks, enrichment fallback, BM25 + Qdrant và Cross-Encoder đều chạy. OpenRouter free model `inclusionai/ling-3.1-flash` trả lỗi `429` (shared provider pool rate-limited), nên `RAGAS_ENABLE_LLM_EVALUATION=false` được bật để tránh pipeline treo.

Các giá trị `0.0` là **không đo được**, không phải RAGAS score thật. LLM cấu hình qua **OpenRouter free**, không dùng OpenAI; embedding dùng BGE-M3 local. Năm case dưới đây hòa điểm placeholder nên được liệt kê theo thứ tự test set, không phải xếp hạng RAGAS có ý nghĩa.

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | Không đo được | Không đo được | N/A |
| Answer Relevancy | Không đo được | Không đo được | N/A |
| Context Precision | Không đo được | Không đo được | N/A |
| Context Recall | Không đo được | Không đo được | N/A |

## Bottom-5 Failures / Priority Cases

### #1 — Nghỉ khi kết hôn

- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Expected:** 3 ngày làm việc có lương, không trừ phép năm.
- **Got:** Không có câu trả lời LLM để chấm; pipeline dùng retrieval fallback sau lỗi `429` của OpenRouter free.
- **Worst metric:** Không đo được (placeholder `0.0` cho cả 4 metric).
- **Error Tree:** Output chưa xác minh → context chưa được RAGAS chấm → query rõ ràng → root cause là provider LLM free rate-limit, không phải kết luận hallucination.
- **Root cause:** Generation/evaluation phụ thuộc provider free đang unavailable.
- **Suggested fix:** Khi rate-limit được gỡ, đặt `OPENROUTER_ENABLE_LLM_CALLS=true` và `RAGAS_ENABLE_LLM_EVALUATION=true`; kiểm tra top-3 có ưu tiên `nghi_phep_dac_biet.md`.

### #2 — Hạn mức PVI

- **Question:** Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Expected:** 200.000.000 VNĐ/năm, gồm nội trú, ngoại trú và nha khoa.
- **Got:** Không có câu trả lời LLM có thể đánh giá tự động do fallback.
- **Worst metric:** Không đo được (placeholder `0.0`).
- **Error Tree:** Output chưa xác minh → context cần kiểm tra lại chứa `bao_hiem_suc_khoe.md` → query rõ ràng → root cause ngoài retrieval: OpenRouter free rate-limit.
- **Root cause:** Không có LLM judge/generation khả dụng trong lần run.
- **Suggested fix:** Rerun RAGAS khi provider hoạt động; nếu context precision thấp, boost metadata nguồn chính sách bảo hiểm.

### #3 — Phụ cấp ăn trưa

- **Question:** Phụ cấp ăn trưa hàng tháng là bao nhiêu?
- **Expected:** 1.000.000 VNĐ/tháng, chi trả cùng kỳ lương.
- **Got:** Không có answer synthesis để so sánh với expected; chỉ có retrieval fallback.
- **Worst metric:** Không đo được (placeholder `0.0`).
- **Error Tree:** Output chưa xác minh → context cần chứa `phu_cap.md` → query không mơ hồ → root cause là availability của LLM free.
- **Root cause:** Free provider không phục vụ request tại thời điểm chạy.
- **Suggested fix:** Khi rerun, kiểm tra top-3 sau rerank; nếu thiếu policy phụ cấp, thêm metadata category `hr`/`benefit`.

### #4 — Phiên bản phép năm 2024

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Chính sách hiện hành 2024 là 15 ngày; bản 2023 là 12 ngày và đã bị thay thế.
- **Got:** Không có LLM output được đánh giá trong lần run.
- **Worst metric:** Không đo được (placeholder `0.0`).
- **Error Tree:** Output chưa xác minh → context có thể chứa cả v2023 và v2024 → query cần ưu tiên văn bản hiện hành → root cause tiềm ẩn là version conflict, nhưng chưa thể kết luận khi thiếu RAGAS.
- **Root cause:** Evaluation outage che khuất lỗi versioning nếu có.
- **Suggested fix:** Boost metadata `effective_date`/`version`, giảm điểm văn bản “đã thay thế”, và thêm assertion câu trả lời nêu v2024.

### #5 — Thâm niên cộng phép

- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo v2024: từ 3 năm, cộng 1 ngày cho mỗi 3 năm; v2023 dùng mốc 5 năm.
- **Got:** Không có LLM output được chấm trong lần run.
- **Worst metric:** Không đo được (placeholder `0.0`).
- **Error Tree:** Output chưa xác minh → context có thể mâu thuẫn v2023/v2024 → query nên ghi “theo chính sách hiện hành” → root cause hiện tại là evaluation unavailable.
- **Root cause:** Không có score/context judgement thật do OpenRouter free rate-limit.
- **Suggested fix:** Thêm query rewrite cho các câu có version, filter/boost policy hiện hành và đánh giá lại bằng RAGAS khi provider ổn định.

## Case Study (cho presentation)

**Question chọn phân tích:** Nhân viên được nghỉ bao nhiêu ngày phép năm?

**Error Tree walkthrough:**

1. Output đúng? → Chưa thể kết luận: generation fallback, không có LLM answer được RAGAS chấm.
2. Context đúng? → Rủi ro là đồng thời retrieve `nghi_phep_nam_v2023.md` và `nghi_phep_nam_v2024.md`.
3. Query rewrite OK? → Nên thêm “theo chính sách hiện hành/v2024”.
4. Fix ở bước: → M2 boost version metadata, M3 ưu tiên văn bản hiện hành, M4 rerun khi OpenRouter available.

**Nếu có thêm 1 giờ, sẽ optimize:** OCR ba PDF; thêm metadata `version`, `effective_date`, `superseded`; rerun RAGAS với provider free/quota khả dụng để thay placeholder bằng metric thật.
