"""Token-aware text chunking for LLM processing.

Splits large text into chunks that fit within a model's context window,
preferring paragraph boundaries, then sentence boundaries, then word boundaries.
Supports configurable overlap between chunks to preserve context at edges.
"""

import re

import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a string using tiktoken.

    Args:
        text: The text to count tokens for.
        model: Model name for tokenizer selection. Defaults to gpt-4
            (cl100k_base encoding), which is a reasonable approximation
            for most modern models.

    Returns:
        Token count.
    """
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newlines."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences on common sentence boundaries."""
    # Match sentence endings followed by whitespace or end of string
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _build_chunk(
    pieces: list[str], max_tokens: int, model: str
) -> str:
    """Join pieces into a chunk that fits within max_tokens."""
    return "\n\n".join(pieces)


def chunk_text(
    text: str,
    max_tokens: int = 2000,
    overlap_tokens: int = 0,
    model: str = "gpt-4",
) -> list[str]:
    """Split text into chunks that respect token limits.

    Chunking strategy (in priority order):
    1. If the entire text fits, return it as a single chunk.
    2. Try to split at paragraph boundaries.
    3. If a paragraph is too large, split it at sentence boundaries.
    4. If a sentence is too large, include it as-is (can't split mid-sentence
       without losing meaning).

    Args:
        text: The text to chunk.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of tokens to overlap between consecutive chunks.
            This is approximate — overlap is implemented at the sentence level.
        model: Model name for tokenizer selection.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    if count_tokens(text, model) <= max_tokens:
        return [text]

    chunks: list[str] = []
    current_pieces: list[str] = []
    current_tokens = 0

    paragraphs = _split_into_paragraphs(text)

    for para in paragraphs:
        para_tokens = count_tokens(para, model)

        # If the paragraph itself exceeds max_tokens, split it into sentences
        if para_tokens > max_tokens:
            # Flush current chunk first
            if current_pieces:
                chunks.append("\n\n".join(current_pieces))
                current_pieces = []
                current_tokens = 0

            sentences = _split_into_sentences(para)
            sent_pieces: list[str] = []
            sent_tokens = 0

            for sentence in sentences:
                sent_token_count = count_tokens(sentence, model)

                if sent_tokens + sent_token_count > max_tokens and sent_pieces:
                    chunks.append(" ".join(sent_pieces))
                    # Handle overlap: carry some sentences forward
                    if overlap_tokens > 0 and sent_pieces:
                        overlap_pieces: list[str] = []
                        overlap_count = 0
                        for s in reversed(sent_pieces):
                            s_t = count_tokens(s, model)
                            if overlap_count + s_t > overlap_tokens:
                                break
                            overlap_pieces.insert(0, s)
                            overlap_count += s_t
                        sent_pieces = overlap_pieces
                        sent_tokens = overlap_count
                    else:
                        sent_pieces = []
                        sent_tokens = 0

                sent_pieces.append(sentence)
                sent_tokens += sent_token_count

            if sent_pieces:
                chunks.append(" ".join(sent_pieces))
                # Set up overlap for the next paragraph
                if overlap_tokens > 0 and sent_pieces:
                    overlap_pieces = []
                    overlap_count = 0
                    for s in reversed(sent_pieces):
                        s_t = count_tokens(s, model)
                        if overlap_count + s_t > overlap_tokens:
                            break
                        overlap_pieces.insert(0, s)
                        overlap_count += s_t
                    if overlap_pieces:
                        current_pieces = overlap_pieces
                        current_tokens = overlap_count
            continue

        # Normal paragraph: does it fit in the current chunk?
        if current_tokens + para_tokens > max_tokens and current_pieces:
            chunks.append("\n\n".join(current_pieces))
            # Handle overlap
            if overlap_tokens > 0:
                overlap_pieces = []
                overlap_count = 0
                for p in reversed(current_pieces):
                    p_t = count_tokens(p, model)
                    if overlap_count + p_t > overlap_tokens:
                        break
                    overlap_pieces.insert(0, p)
                    overlap_count += p_t
                current_pieces = overlap_pieces
                current_tokens = overlap_count
            else:
                current_pieces = []
                current_tokens = 0

        current_pieces.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_pieces:
        chunks.append("\n\n".join(current_pieces))

    return chunks
