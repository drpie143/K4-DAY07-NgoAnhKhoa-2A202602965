from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, HeadingChunker, RecursiveChunker, SentenceChunker
from src.embeddings import OpenAIEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

DEFAULT_STRATEGY = "heading"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 50
DEFAULT_SENTENCES_PER_CHUNK = 3
CACHE_FILE = Path("data/ecommerce/.embedding_cache.json")


class CachedOpenAIEmbedder:
    """OpenAI embeddings with a model-aware disk cache."""

    def __init__(self) -> None:
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedder = OpenAIEmbedder(model_name=model_name)
        self.cache: dict[str, list[float]] = {}
        self._new_entries = 0
        if CACHE_FILE.exists():
            try:
                self.cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                self.cache = {}

    def _hash(self, text: str) -> str:
        payload = f"{self.embedder.model_name}\0{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def embed_many(self, texts: list[str], batch_size: int = 100) -> None:
        missing: dict[str, str] = {}
        for text in texts:
            digest = self._hash(text)
            if digest not in self.cache:
                missing[digest] = text
        pending = list(missing.items())
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            response = self.embedder.client.embeddings.create(
                model=self.embedder.model_name,
                input=[text for _, text in batch],
            )
            for (digest, _), item in zip(batch, response.data):
                self.cache[digest] = [float(value) for value in item.embedding]
            self._new_entries += len(batch)
            self.save()

    def __call__(self, text: str) -> list[float]:
        digest = self._hash(text)
        if digest not in self.cache:
            self.cache[digest] = self.embedder(text)
            self._new_entries += 1
            if self._new_entries >= 25:
                self.save()
        return self.cache[digest]

    def save(self) -> None:
        if not self._new_entries:
            return
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(self.cache), encoding="utf-8")
        self._new_entries = 0


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    metadata: dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, parts[2].strip()


def load_and_chunk_corpus(corpus_dir: Path, chunker: Any) -> list[Document]:
    documents: list[Document] = []
    for file_path in sorted(corpus_dir.glob("*.md")):
        metadata, body = parse_frontmatter(file_path.read_text(encoding="utf-8"))
        chunks = chunker.chunk(body)
        doc_id = metadata.get("doc_id", file_path.stem)
        for index, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "doc_id": doc_id,
                "source": str(file_path),
                "chunk_index": index,
                "total_chunks": len(chunks),
            }
            documents.append(Document(f"{doc_id}#{index}", chunk, chunk_metadata))
    return documents


def llm_answer_fn(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and not api_key.startswith("your-key"):
        try:
            from openai import OpenAI

            completion = OpenAI(api_key=api_key).chat.completions.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là trợ lý RAG chính sách Shopee. Chỉ dùng ngữ cảnh, "
                            "trả lời ngắn gọn và trích dẫn [1], [2]."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return completion.choices[0].message.content or ""
        except Exception:
            pass
    return f"[RAG Agent Answer] Phản hồi dựa trên ngữ cảnh: {prompt[:250]}..."


BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Người mua có thể gửi yêu cầu trả hàng hoàn tiền trong thời hạn bao lâu sau khi nhận hàng?",
        "gold_answer": "15 ngày; riêng thực phẩm tươi sống và đông lạnh là 24 giờ.",
        "gold_doc": "shopee-return-refund-buyer",
        "key_fact": "15 (mười lăm) ngày",
        "answer_facts": ["15 ngày", "24 giờ"],
        "filter": None,
    },
    {
        "id": 2,
        "query": "Thời hạn xử lý và phản hồi khiếu nại yêu cầu trả hàng hoàn tiền của Người Bán là bao lâu?",
        "gold_answer": "Người Bán cần phản hồi trong vòng 02 ngày lịch.",
        "gold_doc": "shopee-return-refund-seller",
        "key_fact": "02 ngày lịch",
        "answer_facts": ["02 ngày lịch"],
        "filter": {"audience": "seller"},
    },
    {
        "id": 3,
        "query": "Người bán có hành vi gian lận tạo đơn hàng ảo trên Shopee bị xử phạt bồi thường bao nhiêu tiền cho mỗi đơn hàng vi phạm?",
        "gold_answer": "Tối đa 10.000.000 VND cho mỗi đơn hàng vi phạm.",
        "gold_doc": "shopee-antifraud-seller-penalty",
        "key_fact": "10.000.000",
        "answer_facts": ["10.000.000"],
        "filter": {"audience": "seller"},
    },
    {
        "id": 4,
        "query": "Quy định về hạn sử dụng của hàng hóa khi Người Bán giao đi trên Shopee phải còn lại ít nhất bao nhiêu?",
        "gold_answer": "Ít nhất 30% thời hạn sử dụng và còn ít nhất 30 ngày.",
        "gold_doc": "shopee-seller-listing-rules",
        "key_fact": "30%",
        "answer_facts": ["30%", "30 ngày"],
        "filter": None,
    },
    {
        "id": 5,
        "query": "Sản phẩm mua trên Shopee được bảo hành miễn phí khi đáp ứng những điều kiện nào?",
        "gold_answer": "Lỗi kỹ thuật, còn hạn, có hóa đơn/mã đơn và tem bảo hành nguyên vẹn.",
        "gold_doc": "shopee-warranty-buyer",
        "key_fact": "lỗi kỹ thuật",
        "answer_facts": ["lỗi kỹ thuật", "thời hạn bảo hành", "mã đơn hàng", "tem bảo hành"],
        "filter": {"audience": "buyer"},
    },
]


CHUNKER_REGISTRY = {
    "fixed_size": (
        lambda: FixedSizeChunker(DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP),
        "FixedSizeChunker (size=400, overlap=50)",
    ),
    "by_sentences": (
        lambda: SentenceChunker(DEFAULT_SENTENCES_PER_CHUNK),
        "SentenceChunker (max_sentences=3)",
    ),
    "recursive": (
        lambda: RecursiveChunker(chunk_size=DEFAULT_CHUNK_SIZE),
        "RecursiveChunker (chunk_size=400)",
    ),
    "heading": (
        lambda: HeadingChunker(max_size=DEFAULT_CHUNK_SIZE),
        "HeadingChunker (max_size=400, hierarchical Markdown sections)",
    ),
}

CHUNKER_ALIASES = {
    "fixed": "fixed_size",
    "fixedsize": "fixed_size",
    "sentence": "by_sentences",
    "sentences": "by_sentences",
    "headingchunker": "heading",
    "section": "heading",
}


def get_chunker_by_name(name: str) -> tuple[Any, str]:
    normalized = CHUNKER_ALIASES.get(name.strip().lower(), name.strip().lower())
    if normalized not in CHUNKER_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Choose one of: {', '.join(CHUNKER_REGISTRY)}")
    factory, label = CHUNKER_REGISTRY[normalized]
    return factory(), label


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAG chính sách Shopee")
    parser.add_argument(
        "--strategy", default=DEFAULT_STRATEGY, choices=list(CHUNKER_REGISTRY)
    )
    args = parser.parse_args()
    chunker, display_name = get_chunker_by_name(args.strategy)
    corpus_dir = Path("data/ecommerce")

    try:
        embedder: Any = CachedOpenAIEmbedder()
        backend = "OpenAI text-embedding-3-small (có cache disk)"
    except Exception as error:
        embedder = _mock_embed
        backend = f"MockEmbedder fallback ({error})"

    docs = load_and_chunk_corpus(corpus_dir, chunker)
    avg_length = sum(len(doc.content) for doc in docs) / len(docs) if docs else 0.0
    max_length = max((len(doc.content) for doc in docs), default=0)
    if hasattr(embedder, "embed_many"):
        embedder.embed_many(
            [doc.content for doc in docs] + [item["query"] for item in BENCHMARK_QUERIES]
        )
    store = EmbeddingStore(f"bench_{args.strategy}", embedding_fn=embedder)
    store.add_documents(docs)
    if hasattr(embedder, "save"):
        embedder.save()
    agent = KnowledgeBaseAgent(store, llm_answer_fn)

    print("=" * 65)
    print("        BÁO CÁO BENCHMARK RETRIEVAL — K4-L3B TMĐT SHOPEE")
    print(f"        Chiến lược đang chạy: {display_name}")
    print("=" * 65)
    print(f"\n[*] Embedding Backend: {backend}")
    print(f"[*] Tổng số tài liệu: {len(list(corpus_dir.glob('*.md')))} file .md")
    print(f"[*] Tổng số chunks tạo ra: {len(docs)} chunks")

    lines = [
        f"=== KẾT QUẢ ĐO LƯỜNG BENCHMARK ({display_name}) ===",
        f"Tổng số tài liệu: 8 file .md | Tổng số chunks: {len(docs)}",
        f"Độ dài chunk: trung bình={avg_length:.1f} ký tự | lớn nhất={max_length} ký tự",
        "Mô hình embedding: OpenAI text-embedding-3-small\n",
    ]
    total_points = hit_at_1 = hit_at_3 = 0
    reciprocal_rank_sum = 0.0

    for item in BENCHMARK_QUERIES:
        question = item["query"]
        metadata_filter = item["filter"]
        header = f"--- [Câu hỏi {item['id']}] {question} ---"
        print(header)
        lines.extend([header, f"Áp dụng Metadata Filter: {metadata_filter or 'Không (None)'}"])
        if metadata_filter:
            results = store.search_with_filter(question, 3, metadata_filter)
            answer = agent.answer_with_filter(question, metadata_filter, 3)
        else:
            results = store.search(question, 3)
            answer = agent.answer(question, 3)

        gold_rank = None
        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            content = result["content"]
            if (
                metadata.get("doc_id") == item["gold_doc"]
                and item["key_fact"].lower() in content.lower()
                and gold_rank is None
            ):
                gold_rank = rank
            line = (
                f"  Top-{rank} [Score: {result['score']:.4f}] "
                f"({metadata.get('doc_id')}): {' '.join(content.split())[:110]}..."
            )
            print(line)
            lines.append(line)
        print(f"  => Agent Answer: {answer}\n")

        normalized_answer = " ".join(answer.lower().split())
        normalized_answer = " ".join(re.sub(r"\([^)]*\)", " ", normalized_answer).split())
        facts_ok = all(fact.lower() in normalized_answer for fact in item["answer_facts"])
        citation_ok = bool(re.search(r"\[\d+\]", answer))
        if gold_rank == 1:
            hit_at_1 += 1
        if gold_rank is not None:
            hit_at_3 += 1
            reciprocal_rank_sum += 1 / gold_rank
        if gold_rank == 1 and facts_ok and citation_ok:
            query_points = 2
            status = "2/2 - Chunk đúng Top-1, Agent đủ key facts và citation"
        elif gold_rank is not None:
            query_points = 1
            status = "1/2 - Có chunk đúng nhưng thứ hạng/Agent/citation chưa đầy đủ"
        else:
            query_points = 0
            status = "0/2 - Không có chunk chứa key fact trong Top-3"
        total_points += query_points
        lines.extend(
            [
                f"  => Gold Answer : {item['gold_answer']}",
                f"  => Agent Answer: {answer}",
                f"  => Gold rank: {gold_rank or 'không có trong Top-3'} | Agent facts: {'đủ' if facts_ok else 'thiếu'} | Citation: {'có' if citation_ok else 'không'}",
                f"  => Đánh giá: {status}\n",
            ]
        )

    ab_question = BENCHMARK_QUERIES[1]["query"]
    lines.extend(["=" * 65, "A/B TESTING METADATA FILTER (CÂU 2)", "=" * 65])
    for label, results in (
        ("KHÔNG LỌC", store.search(ab_question, 3)),
        ("CÓ LỌC audience='seller'", store.search_with_filter(ab_question, 3, {"audience": "seller"})),
    ):
        lines.append(f"\n[{label}]")
        for rank, result in enumerate(results, start=1):
            lines.append(
                f"  Top-{rank} [{result['metadata'].get('audience')} | {result['score']:.4f}]: "
                f"{' '.join(result['content'].split())[:110]}..."
            )

    count = len(BENCHMARK_QUERIES)
    lines.extend(
        [
            "\n=== TỔNG KẾT ĐÁNH GIÁ ===",
            f"Điểm Retrieval Quality: {total_points}/10",
            f"Hit@1: {hit_at_1}/{count} = {hit_at_1 / count:.2%}",
            f"Hit@3: {hit_at_3}/{count} = {hit_at_3 / count:.2%}",
            f"MRR: {reciprocal_rank_sum / count:.4f}",
        ]
    )
    Path("ket_qua_benchmark.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n[✓] Đã ghi kết quả vào ket_qua_benchmark.txt")


if __name__ == "__main__":
    main()
