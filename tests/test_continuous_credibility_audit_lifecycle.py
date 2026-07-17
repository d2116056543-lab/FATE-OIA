import torch
from types import SimpleNamespace

from fate_oia.models.mosaic_continuous_credibility import ContinuousVisualCredibility
from fate_oia.models.mosaic_continuous_credibility import visual_credibility_from_measurements


def test_credibility_builder_uses_bootstrap_visual_audit_not_reason_targets() -> None:
    from fate_oia.engine.build_mosaic_visual_credibility import build_visual_credibility

    factor_stats = {
        "strong": {
            "counts": {"confirmed_positive": 32, "reliable_negative": 32, "unknown": 0},
            "bootstrap_positive_rate": {
                "full_minus_prior_only": 1.0,
                "query_shuffle_drop": 1.0,
                "image_shuffle_drop": 1.0,
                "grounding_minus_random": 1.0,
                "stability": 1.0,
            },
        },
        "weak": {
            "counts": {"confirmed_positive": 32, "reliable_negative": 32, "unknown": 0},
            "bootstrap_positive_rate": {
                "full_minus_prior_only": 0.0,
                "query_shuffle_drop": 0.0,
                "image_shuffle_drop": 0.0,
                "grounding_minus_random": 0.0,
                "stability": 0.0,
            },
        },
    }
    result = build_visual_credibility(
        factor_stats,
        factor_names=("strong", "weak"),
        factor_roles=("observable", "observable"),
        source_kinds=("grounded", "grounded"),
    )

    assert result["source_split"] == "audit_visual"
    assert result["credibility"][0] > result["credibility"][1]
    assert result["reason_labels_used"] is False


def test_training_forward_cannot_mutate_audit_credibility_ema() -> None:
    module = ContinuousVisualCredibility(factor_count=2, dim=4)
    module.update_from_audit(torch.tensor([0.7, 0.2]))
    before = module.ema_cV.detach().clone()

    _ = module(torch.randn(3, 2, 4), torch.rand(3, 2), torch.rand(3, 2))

    assert torch.equal(module.ema_cV, before)
    assert bool(module.ema_initialized)


def test_credibility_requires_all_visual_intervention_components() -> None:
    common = {
        "content_score": torch.ones(1),
        "prior_score": torch.zeros(1),
        "image_shuffle_score": torch.ones(1),
        "grounding_score": torch.ones(1),
        "stability_score": torch.ones(1),
        "n_eff": torch.full((1,), 128.0),
        "factor_role": "observable",
        "reliable_negative": torch.ones(1, dtype=torch.bool),
        "source_kind": "grounded",
    }
    full = visual_credibility_from_measurements(
        query_shuffle_score=torch.ones(1),
        **common,
    )["cV"]
    missing_query_effect = visual_credibility_from_measurements(
        query_shuffle_score=torch.zeros(1),
        **common,
    )["cV"]

    assert float(missing_query_effect.item()) < float(full.item()) * 0.10


def test_audit_visual_refresh_is_the_only_route_to_model_credibility() -> None:
    from fate_oia.engine.build_mosaic_visual_credibility import refresh_model_visual_credibility

    module = ContinuousVisualCredibility(factor_count=2, dim=4)
    model = SimpleNamespace(
        ontology={
            "factors": [
                {"name": "strong", "role": "observable", "source_kind": "grounded"},
                {"name": "weak", "role": "observable", "source_kind": "grounded"},
            ]
        },
        continuous_credibility=module,
    )
    audit = {
        "source_split": "audit_visual",
        "factor_stats": {
            name: {
                "counts": {"confirmed_positive": 32, "reliable_negative": 32, "unknown": 0},
                "bootstrap_positive_rate": {
                    "full_minus_prior_only": 1.0 if name == "strong" else 0.0,
                    "query_shuffle_drop": 1.0 if name == "strong" else 0.0,
                    "image_shuffle_drop": 1.0 if name == "strong" else 0.0,
                    "grounding_minus_random": 1.0 if name == "strong" else 0.0,
                    "stability": 1.0 if name == "strong" else 0.0,
                },
            }
            for name in ("strong", "weak")
        },
    }

    result = refresh_model_visual_credibility(model, audit)

    assert result["source_split"] == "audit_visual"
    assert bool(module.ema_initialized)
    assert torch.equal(module.ema_cV, result["credibility"])
