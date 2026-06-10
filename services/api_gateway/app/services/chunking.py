def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return []

    paragraphs = cleaned.split("\n")
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) > chunk_size:
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                chunks.append(paragraph[start:end])
                start = max(end - overlap, end)
            current = ""
        else:
            current = paragraph

    if current:
        chunks.append(current)

    return chunks
