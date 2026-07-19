from __future__ import annotations

from fate_oia.engine.build_mosaic_visual_credibility import build_visual_credibility


def test_missing_factor_measurement_is_marked_unavailable_not_zeroed() -> None:
    result = build_visual_credibility(
        {
            "factor": {
                "counts": {"confirmed_positive": 8, "reliable_negative": 8},
                "bootstrap_positive_rate": {
                    "full_minus_prior_only": 0.8,
                    "query_shuffle_drop": 0.7,
                    "image_shuffle_drop": 0.7,
                    # grounding_minus_random is intentionally unavailable.
                    "stability": 0.9,
                },
            }
        },
        factor_names=("factor",),
        factor_roles=("observable",),
        source_kinds=("grounded",),
    )
    assert result["component_availability"]["grounding"] == [False]
    assert result["credibility"].item() > 0.0
