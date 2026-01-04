"""Output formatters for BM25 index tool.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

import json
from typing import Any

from bm25_index_tool.logging_config import get_logger

logger = get_logger(__name__)


def _format_content_with_lines(content: str, max_length: int | None = None) -> str:
    """Format content with line numbers and optional truncation.

    Args:
        content: Document content
        max_length: Maximum content length (characters)

    Returns:
        Formatted content with line numbers
    """
    lines = content.splitlines()

    # Apply truncation if needed
    if max_length is not None and len(content) > max_length:
        truncated_content = content[:max_length]
        lines = truncated_content.splitlines()
        truncation_msg = f"... (truncated, {len(content) - max_length} more characters)"
        lines.append(truncation_msg)

    # Add line numbers
    numbered_lines = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def format_simple(
    results: list[dict[str, Any]],
    include_content: bool = False,
    content_max_length: int | None = None,
) -> str:
    """Format results as a simple list.

    Args:
        results: List of result dictionaries
        include_content: Include full document content
        content_max_length: Maximum content length (None = unlimited)

    Returns:
        Formatted string
    """
    if not results:
        return "No results found."

    lines = []
    for idx, result in enumerate(results, 1):
        score = result["score"]
        name = result["name"]
        path = result["path"]
        fragments = result.get("fragments", [])
        content = result.get("content", "")

        lines.append(f"[{idx}] {name} (score: {score:.4f})")
        lines.append(f"    {path}")

        # Display fragments if present
        if fragments:
            lines.append("")
            for frag_idx, fragment in enumerate(fragments, 1):
                line_start = fragment["line_start"]
                line_end = fragment["line_end"]
                frag_lines = fragment["lines"]

                lines.append(f"    Fragment {frag_idx} (lines {line_start}-{line_end}):")
                lines.append("    " + "-" * 40)

                # Show fragment with indentation
                for line in frag_lines:
                    lines.append(f"    {line}")

                lines.append("")

        # Display content if requested
        if include_content and content:
            lines.append("")
            lines.append("    Content:")
            lines.append("    " + "=" * 60)
            display_content = _format_content_with_lines(content, content_max_length)
            for line in display_content.splitlines():
                lines.append(f"    {line}")
            lines.append("")

    return "\n".join(lines)


def format_json(
    results: list[dict[str, Any]],
    include_content: bool = False,
    content_max_length: int | None = None,
) -> str:
    """Format results as JSON.

    Args:
        results: List of result dictionaries
        include_content: Include full document content
        content_max_length: Maximum content length (None = unlimited)

    Returns:
        JSON string
    """
    # Build output with optional fragments and content
    output = []
    for result in results:
        item = {
            "path": result["path"],
            "name": result["name"],
            "score": result["score"],
        }

        # Include fragments if present
        if "fragments" in result:
            item["fragments"] = result["fragments"]

        # Include content if requested
        if include_content and "content" in result:
            content = result["content"]
            if content_max_length is not None and len(content) > content_max_length:
                content = content[:content_max_length]
                item["content_truncated"] = True
            else:
                item["content_truncated"] = False
            item["content"] = content

        output.append(item)

    return json.dumps({"count": len(output), "results": output}, indent=2)


def format_rich(
    results: list[dict[str, Any]],
    context_lines: int = 2,
    include_content: bool = False,
    content_max_length: int | None = None,
) -> str:
    """Format results with rich context snippets.

    Args:
        results: List of result dictionaries
        context_lines: Number of context lines around matches
        include_content: Include full document content
        content_max_length: Maximum content length (None = unlimited)

    Returns:
        Formatted string
    """
    if not results:
        return "No results found."

    output_lines = []

    for idx, result in enumerate(results, 1):
        score = result["score"]
        name = result["name"]
        path = result["path"]
        content = result.get("content", "")
        fragments = result.get("fragments", [])

        # Header
        output_lines.append(f"\n[{idx}] {name} (score: {score:.4f})")
        output_lines.append(f"    {path}\n")

        # Show fragments if present, otherwise show first 5 lines
        if fragments:
            for frag_idx, fragment in enumerate(fragments, 1):
                line_start = fragment["line_start"]
                line_end = fragment["line_end"]
                frag_lines = fragment["lines"]

                output_lines.append(f"    Fragment {frag_idx} (lines {line_start}-{line_end}):")
                output_lines.append("    " + "=" * 60)

                # Show fragment with indentation
                for line in frag_lines:
                    output_lines.append(f"    {line}")

                output_lines.append("")

        # Show full content if requested
        if include_content and content:
            output_lines.append("    Content:")
            output_lines.append("    " + "=" * 60)
            display_content = _format_content_with_lines(content, content_max_length)
            for line in display_content.splitlines():
                output_lines.append(f"    {line}")
            output_lines.append("")
        elif content and not fragments:
            # Fallback to showing first 5 lines if no fragments and no full content
            lines = content.splitlines()
            preview_lines = lines[: min(5, len(lines))]
            snippet = "\n".join(preview_lines)

            if len(lines) > 5:
                snippet += f"\n    ... ({len(lines) - 5} more lines)"

            output_lines.append("    " + snippet.replace("\n", "\n    "))

    return "\n".join(output_lines)
