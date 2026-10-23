"""Regression tests for issue #23683.

`MetadataAwareTextSplitter._get_metadata_str()` must return an empty string
when the splitter is constructed with `include_metadata=False`, so that
disabled metadata never consumes chunk budget (and can never raise
"Metadata length (...) is longer than chunk size (...)").
"""

import pytest

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.node_parser.interface import MetadataAwareTextSplitter
from llama_index.core.schema import BaseNode, MetadataMode

BIG_METADATA = {"context": "x" * 100}


def _token_splitter(include_metadata: bool = False) -> TokenTextSplitter:
    return TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=include_metadata
    )


def _sentence_splitter(include_metadata: bool = False) -> SentenceSplitter:
    # chunking_tokenizer_fn avoids nltk punkt data (offline test env)
    return SentenceSplitter(
        chunk_size=20,
        chunk_overlap=0,
        tokenizer=list,
        include_metadata=include_metadata,
        chunking_tokenizer_fn=lambda text: [text],
    )


def test_get_metadata_str_returns_empty_when_metadata_disabled() -> None:
    """_get_metadata_str must not format metadata when include_metadata=False."""
    doc = Document(text="hello", metadata=BIG_METADATA)
    assert _token_splitter(include_metadata=False)._get_metadata_str(doc) == ""
    assert _sentence_splitter(include_metadata=False)._get_metadata_str(doc) == ""


def test_get_metadata_str_unchanged_when_metadata_enabled() -> None:
    """Enabled-metadata behavior must be byte-identical to before the fix."""
    doc = Document(text="hello", metadata={"a": "1"})

    expected = doc.get_metadata_str(mode=MetadataMode.EMBED)
    if len(doc.get_metadata_str(mode=MetadataMode.LLM)) > len(expected):
        expected = doc.get_metadata_str(mode=MetadataMode.LLM)

    assert _token_splitter(include_metadata=True)._get_metadata_str(doc) == expected
    assert _sentence_splitter(include_metadata=True)._get_metadata_str(doc) == expected


def test_token_splitter_disabled_metadata_ignores_budget() -> None:
    """Exact repro from issue #23683: must not raise, nodes get empty metadata."""
    splitter = _token_splitter(include_metadata=False)
    nodes = splitter.get_nodes_from_documents(
        [Document(text="word " * 12, metadata=BIG_METADATA)]
    )
    assert len(nodes) > 0
    for node in nodes:
        assert node.metadata == {}
    # text splits within its own budget: all 12 words preserved, chunks <= chunk size
    chunks = [n.get_content(metadata_mode=MetadataMode.NONE) for n in nodes]
    assert " ".join(chunks).split() == ["word"] * 12
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_sentence_splitter_disabled_metadata_ignores_budget() -> None:
    """Same guarantee for SentenceSplitter."""
    splitter = _sentence_splitter(include_metadata=False)
    nodes = splitter.get_nodes_from_documents(
        [Document(text="word " * 12, metadata=BIG_METADATA)]
    )
    assert len(nodes) > 0
    for node in nodes:
        assert node.metadata == {}


@pytest.mark.asyncio
async def test_token_splitter_disabled_metadata_async() -> None:
    """Async entry point (aget_nodes_from_documents) must behave the same."""
    splitter = _token_splitter(include_metadata=False)
    nodes = await splitter.aget_nodes_from_documents(
        [Document(text="word " * 12, metadata=BIG_METADATA)]
    )
    assert len(nodes) > 0
    for node in nodes:
        assert node.metadata == {}


@pytest.mark.asyncio
async def test_sentence_splitter_disabled_metadata_async() -> None:
    """Async entry point for SentenceSplitter."""
    splitter = _sentence_splitter(include_metadata=False)
    nodes = await splitter.aget_nodes_from_documents(
        [Document(text="word " * 12, metadata=BIG_METADATA)]
    )
    assert len(nodes) > 0
    for node in nodes:
        assert node.metadata == {}


def test_token_splitter_enabled_metadata_still_rejects_oversized() -> None:
    """Oversized metadata must still raise when include_metadata=True (default)."""
    splitter = TokenTextSplitter(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=True
    )
    with pytest.raises(ValueError, match="Metadata length"):
        splitter.get_nodes_from_documents(
            [Document(text="word " * 12, metadata=BIG_METADATA)]
        )


def test_sentence_splitter_enabled_metadata_still_rejects_oversized() -> None:
    """Oversized metadata must still raise for SentenceSplitter when enabled."""
    splitter = SentenceSplitter(
        chunk_size=20,
        chunk_overlap=0,
        tokenizer=list,
        include_metadata=True,
        chunking_tokenizer_fn=lambda text: [text],
    )
    with pytest.raises(ValueError, match="Metadata length"):
        splitter.get_nodes_from_documents(
            [Document(text="word " * 12, metadata=BIG_METADATA)]
        )


def test_disabled_metadata_budget_matches_enabled_empty_metadata() -> None:
    """Splitting must be identical to enabled-metadata with empty doc metadata."""
    disabled: MetadataAwareTextSplitter = _token_splitter(include_metadata=False)
    enabled: MetadataAwareTextSplitter = _token_splitter(include_metadata=True)
    doc_big = Document(text="word " * 12, metadata=BIG_METADATA)
    doc_empty = Document(text="word " * 12, metadata={})
    nodes_disabled = disabled.get_nodes_from_documents([doc_big])
    nodes_enabled_empty = enabled.get_nodes_from_documents([doc_empty])
    assert [
        n.get_content(metadata_mode=MetadataMode.NONE) for n in nodes_disabled
    ] == [n.get_content(metadata_mode=MetadataMode.NONE) for n in nodes_enabled_empty]
