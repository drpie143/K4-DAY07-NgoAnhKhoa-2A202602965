# Tổng hợp benchmark chunking

Ngày chạy: 20/09/2026

Corpus: 8 tài liệu chính sách Shopee

Embedding: OpenAI `text-embedding-3-small`

Agent: OpenAI `gpt-4o-mini`

| Strategy | Chunks | Trung bình | Max | Điểm | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| FixedSizeChunker (400, overlap 50) | 374 | 396,9 | 400 | 6/10 | 40% | 80% | 0,5667 |
| SentenceChunker (3 câu/chunk) | 337 | 383,8 | 1.748 | **9/10** | 80% | **100%** | **0,9000** |
| RecursiveChunker (400) | 416 | 308,7 | 400 | 7/10 | 60% | 80% | 0,6667 |
| HeadingChunker (400) | 645 | 327,4 | 400 | 7/10 | **80%** | 80% | 0,8000 |

## Điểm từng câu

| Strategy | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| Fixed-size | 2 | 1 | 2 | 1 | 0 |
| Sentence | 2 | 2 | 1 | 2 | 2 |
| Recursive | 2 | 2 | 1 | 2 | 0 |
| Heading | 2 | 2 | 0 | 2 | 1 |

SentenceChunker thắng trên bộ benchmark này. HeadingChunker của Ngô Anh Khoa giữ ngữ cảnh cấu trúc tốt và có 4/5 gold chunk ở Top-1, nhưng câu 3 bị các chunk cùng heading cạnh tranh và câu 5 Agent thiếu điều kiện tem bảo hành.

Kết quả chi tiết nằm trong:

- `ket_qua_benchmark_fixed_size.txt`
- `ket_qua_benchmark_by_sentences.txt`
- `ket_qua_benchmark_recursive.txt`
- `ket_qua_benchmark_heading.txt`
