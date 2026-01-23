"""Tests for FTS5 query tokenization in sqlite_storage module."""

from bm25_index_tool.storage.sqlite_storage import tokenize_fts5_query


class TestTokenizeFts5Query:
    """Tests for the tokenize_fts5_query function."""

    def test_simple_term(self) -> None:
        """Simple terms should be lowercased."""
        assert tokenize_fts5_query("hello") == "hello"
        assert tokenize_fts5_query("HELLO") == "hello"

    def test_multiple_terms(self) -> None:
        """Multiple terms separated by spaces."""
        assert tokenize_fts5_query("hello world") == "hello world"

    def test_hyphenated_term(self) -> None:
        """Hyphenated terms should be split into separate tokens."""
        assert tokenize_fts5_query("vip-layer") == "vip layer"

    def test_dotted_domain(self) -> None:
        """Domain names should be split on dots."""
        assert tokenize_fts5_query("dc-ratingen.de") == "dc ratingen de"

    def test_complex_url_pattern(self) -> None:
        """Complex patterns like URLs should be fully tokenized."""
        result = tokenize_fts5_query("vip-layerprd701.dc-ratingen.de")
        assert result == "vip layerprd701 dc ratingen de"

    def test_colon_term(self) -> None:
        """Terms with colons should be split."""
        assert tokenize_fts5_query("path:value") == "path value"

    def test_mixed_special_chars(self) -> None:
        """Mix of special characters should all be split."""
        result = tokenize_fts5_query("vip-layer config:test")
        assert result == "vip layer config test"

    def test_empty_query(self) -> None:
        """Empty query returns empty string."""
        assert tokenize_fts5_query("") == ""

    def test_whitespace_only(self) -> None:
        """Whitespace only returns empty string."""
        assert tokenize_fts5_query("   ") == ""

    def test_numbers_preserved(self) -> None:
        """Numbers should be preserved in tokens."""
        assert tokenize_fts5_query("layer7") == "layer7"
        assert tokenize_fts5_query("prd701") == "prd701"

    def test_uppercase_lowercased(self) -> None:
        """Uppercase should be converted to lowercase."""
        assert tokenize_fts5_query("VIP-LAYER") == "vip layer"

    def test_underscores_split(self) -> None:
        """Underscores should also split tokens."""
        assert tokenize_fts5_query("config_value") == "config value"
