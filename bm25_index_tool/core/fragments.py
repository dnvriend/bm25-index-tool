"""Fragment extraction for BM25 search results.

Note: This code was generated with assistance from AI coding tools
and has been reviewed and tested by a human.
"""

from typing import Any

from bm25_index_tool.logging_config import get_logger

logger = get_logger(__name__)


def extract_fragments(
    content: str,
    query_terms: list[str],
    context_lines: int = 3,
    max_fragments: int = 3,
) -> list[dict[str, Any]]:
    """Extract text fragments containing query terms with context.

    Args:
        content: Full document content
        query_terms: List of query terms to find (case-insensitive)
        context_lines: Number of context lines before/after match
        max_fragments: Maximum number of fragments to extract

    Returns:
        List of fragment dicts with structure:
        {
            "line_start": int,
            "line_end": int,
            "lines": list[str],
            "matched_line_numbers": list[int]
        }
    """
    if not content or not query_terms:
        return []

    lines = content.splitlines()
    if not lines:
        return []

    # Find all lines containing any query term (case-insensitive)
    matched_lines = set()
    query_terms_lower = [term.lower() for term in query_terms]

    for line_idx, line in enumerate(lines):
        line_lower = line.lower()
        for term in query_terms_lower:
            if term in line_lower:
                matched_lines.add(line_idx)
                break

    if not matched_lines:
        logger.debug("No matching lines found for query terms")
        return []

    # Sort matched lines
    sorted_matches = sorted(matched_lines)

    # Build fragments with context, merging overlapping ranges
    fragments = []
    current_fragment: dict[str, Any] | None = None

    for match_idx in sorted_matches:
        # Calculate range for this match
        start = max(0, match_idx - context_lines)
        end = min(len(lines) - 1, match_idx + context_lines)

        if current_fragment is None:
            # Start new fragment
            current_fragment = {
                "line_start": start + 1,  # 1-indexed for display
                "line_end": end + 1,
                "lines": lines[start : end + 1],
                "matched_line_numbers": [match_idx + 1],
            }
        else:
            # Check if this match overlaps with current fragment
            current_end = current_fragment["line_end"] - 1  # Convert back to 0-indexed

            if start <= current_end + 1:
                # Overlapping or adjacent - extend current fragment
                new_end = max(current_end, end)
                current_fragment["line_end"] = new_end + 1
                current_fragment["lines"] = lines[current_fragment["line_start"] - 1 : new_end + 1]
                current_fragment["matched_line_numbers"].append(match_idx + 1)
            else:
                # No overlap - save current and start new
                fragments.append(current_fragment)
                if len(fragments) >= max_fragments:
                    break

                current_fragment = {
                    "line_start": start + 1,
                    "line_end": end + 1,
                    "lines": lines[start : end + 1],
                    "matched_line_numbers": [match_idx + 1],
                }

    # Add the last fragment
    if current_fragment and len(fragments) < max_fragments:
        fragments.append(current_fragment)

    logger.debug("Extracted %d fragments from %d matched lines", len(fragments), len(matched_lines))

    return fragments
