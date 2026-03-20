"""Shared contract helpers for comparable method outputs."""

from __future__ import annotations

from typing import Any


CORE_METRIC_NAMES = ("exact_acc", "mean_shared_digits")


def sanitize_method_token(value: Any) -> str:
    """Normalize one method-id token into a stable ASCII-ish fragment."""
    text = str(value).strip().lower()
    for old, new in (
        (" ", "-"),
        ("/", "-"),
        ("=", ""),
        ("(", ""),
        (")", ""),
        (",", "-"),
        (".", "p"),
    ):
        text = text.replace(old, new)
    return text


def build_flat_method_id(*parts: Any) -> str:
    """Build one flat method identifier from non-empty parts."""
    tokens = [sanitize_method_token(part) for part in parts if str(part).strip()]
    return "_".join(token for token in tokens if token)


def build_transport_method_id(
    method_family: str,
    *,
    site_policy: str,
    resolution: int,
    geometry_metric: str,
    alpha: float | None = None,
    pca_components: int | None = None,
    pca_candidate_count: int | None = None,
) -> str:
    """Build a stable transport method id with optional PCA metadata."""
    parts: list[Any] = [method_family, site_policy, f"res{int(resolution)}", geometry_metric]
    if alpha is not None:
        parts.append(f"a{alpha:g}")
    if site_policy == "pca":
        if pca_components is not None:
            parts.append(f"pc{int(pca_components)}")
        if pca_candidate_count is not None:
            parts.append(f"keep{int(pca_candidate_count)}")
    return build_flat_method_id(*parts)


def build_das_method_id() -> str:
    """Build the current DAS baseline id."""
    return build_flat_method_id("das")


def annotate_result_records(
    records: list[dict[str, object]],
    *,
    method_id: str,
    canonical_variable_mapping: dict[str, str],
) -> list[dict[str, object]]:
    """Attach required contract metadata to per-variable result records."""
    annotated = []
    for record in records:
        local_variable = str(record["variable"])
        annotated.append(
            {
                **record,
                "method_id": method_id,
                "local_variable": local_variable,
                "canonical_variable": canonical_variable_mapping[local_variable],
            }
        )
    return annotated
