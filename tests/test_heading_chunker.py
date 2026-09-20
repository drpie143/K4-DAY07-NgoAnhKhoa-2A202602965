from src.chunking import HeadingChunker


def test_empty_text_returns_empty_list():
    assert HeadingChunker().chunk("") == []


def test_keeps_short_markdown_sections():
    text = "# Chính sách\nMở đầu.\n\n## Điều 1\nNội dung điều một."
    chunks = HeadingChunker(max_size=100).chunk(text)
    assert chunks == [
        "# Chính sách\nMở đầu.",
        "# Chính sách\n## Điều 1\nNội dung điều một.",
    ]


def test_repeats_heading_for_oversized_section_children():
    text = "## Điều kiện\n" + ("Nội dung chính sách. " * 30)
    chunks = HeadingChunker(max_size=100).chunk(text)
    assert len(chunks) > 1
    assert all(chunk.startswith("## Điều kiện\n") for chunk in chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_long_heading_is_not_duplicated():
    text = "## " + ("Điều khoản rất dài " * 20)
    chunks = HeadingChunker(max_size=80).chunk(text)
    assert chunks
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert all(chunk.count("## ") <= 1 for chunk in chunks)


def test_child_chunks_keep_parent_heading_context():
    text = "# Chính sách bảo hành\nMở đầu.\n\n## Điều kiện\nSản phẩm bị lỗi kỹ thuật."
    chunks = HeadingChunker(max_size=100).chunk(text)
    child = next(chunk for chunk in chunks if "lỗi kỹ thuật" in chunk)
    assert child.startswith("# Chính sách bảo hành\n## Điều kiện\n")


def test_plain_text_falls_back_to_bounded_recursive_chunks():
    chunks = HeadingChunker(max_size=50).chunk("từ " * 100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
