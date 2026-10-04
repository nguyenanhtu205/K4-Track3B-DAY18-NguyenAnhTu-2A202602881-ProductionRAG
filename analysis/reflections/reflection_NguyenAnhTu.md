# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Anh Tú  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Tách câu bằng regex, dùng `all-MiniLM-L6-v2` và cosine similarity threshold 0.85; model nạp local cache. |
| Hierarchical / structure-aware chunking | M1 | `chunk_hierarchical()`, `chunk_structure_aware()` | Parent 2048 ký tự cho context, child 256 ký tự cho retrieval; header Markdown được giữ trong `section`. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | Chuẩn hóa `_` về khoảng trắng giúp BM25 bắt “nghỉ phép”; RRF dùng `1/(60 + rank + 1)`. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | BGE reranker trả top-3. Benchmark warm-run với 3 documents: trung bình 181.18ms trên CPU, cao hơn mục tiêu tham khảo 150ms. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Có đủ 4 metric và Diagnostic Tree. LLM dùng OpenRouter free, embeddings BGE-M3 local; run cuối chưa có metric thật vì provider free rate-limit. |
| Contextual embeddings | M5 | `_enrich_single_call()`, `contextual_prepend()` | Combined mode sinh summary, HyQA, context, metadata trong một call; fallback deterministic giữ pipeline hoạt động khi LLM unavailable. |

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải:**
  - `C:\Program Files\Python314\python.exe: No module named pytest`
  - `Error code: 429 - ... inclusionai/ling-3.1-flash is temporarily rate-limited upstream.`
  - `⚠️ Bỏ qua BCTC.pdf: PDF scan ảnh, không có text layer (cần OCR).`
- **Debug & giải quyết:** Dùng `.venv\Scripts\python.exe -m pytest` với Python 3.11; cấu hình OpenRouter-compatible endpoint, timeout 15 giây, không retry, circuit breaker và fallback. Khi free pool rate-limit, tắt RAGAS LLM trong `.env` để không treo 80 jobs và ghi rõ score 0 là không đo được. PDF scan được bỏ qua đúng cách; hướng tiếp theo là OCR.
- **Kiến thức cần bổ sung:** Version-aware retrieval cho chính sách v2023/v2024; benchmark reranker theo batch/GPU/ONNX để hạ latency.

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống RAG tra cứu quy chế nội bộ tiếng Việt

#### 1. Hiện trạng

- **Pipeline hiện tại:** Markdown/PDF text layer → hierarchical chunks → contextual enrichment → BM25 + BGE-M3/Qdrant + RRF → BGE Cross-Encoder top-3 → LLM synthesis → RAGAS.
- **Vấn đề / Bottlenecks:** PDF scan chưa OCR; policy cũ/mới conflict; reranker CPU khoảng 181ms/3 docs; OpenRouter free rate-limit nên RAGAS end-to-end chưa đáng tin cậy.

#### 2. Kế hoạch cải tiến

1. **Chunking:** Giữ Hierarchical mặc định và Structure-Aware cho Markdown để bảng/list không mất header.
2. **Search:** Hybrid BM25 + Dense + RRF; thêm metadata `version`, `effective_date`, `superseded` để boost văn bản hiện hành.
3. **Reranking:** Giữ BGE Cross-Encoder cho chất lượng; thử Flashrank/ONNX hoặc GPU nếu SLA dưới 150ms là bắt buộc.
4. **Evaluation:** Bật lại RAGAS khi provider ổn định; thêm regression tests cho version conflict, negation, multi-hop và số liệu.
5. **Enrichment:** Dùng combined contextual prepend + HyQA khi LLM sẵn sàng; giữ fallback deterministic cho availability.

#### 3. Timeline triển khai

- **Tuần 1:** OCR ba PDF, bổ sung metadata version/effective date, viết regression tests cho policy bị thay thế.
- **Tuần 2:** Benchmark search/reranker, tối ưu batch/GPU hoặc Flashrank; rerun RAGAS với quota/provider khả dụng và cập nhật failure analysis bằng metric thật.
