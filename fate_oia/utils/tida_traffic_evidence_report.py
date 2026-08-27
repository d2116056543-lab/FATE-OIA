from __future__ import annotations

import html
import json
from typing import Any, Iterable


def _number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key, 0.0)
    return float(value) if value is not None else 0.0


def _ci_is_positive(mapping: dict[str, Any], key: str) -> bool:
    interval = mapping.get(key)
    return bool(
        isinstance(interval, (list, tuple))
        and len(interval) == 2
        and float(interval[0]) > 0.0
    )


def _mechanism_summary(branch: dict[str, Any]) -> dict[str, Any]:
    utility = branch.get("utility_quality", {})
    return {
        "conditional_information_gain_bits": _number(
            branch, "conditional_information_gain_bits"
        ),
        "conditional_nll_improvement": _number(
            branch, "conditional_nll_improvement"
        ),
        "conditional_nll_improvement_ci95": branch.get(
            "conditional_nll_improvement_ci95", []
        ),
        "relative_brier_reduction": _number(branch, "relative_brier_reduction"),
        "selected_minus_random_deletion_gap": _number(
            branch, "selected_minus_random_deletion_gap"
        ),
        "selected_minus_random_deletion_gap_ci95": branch.get(
            "selected_minus_random_deletion_gap_ci95", []
        ),
        "target_effective_route_rate": _number(
            branch, "target_effective_route_rate"
        ),
        "net_corrected_labels": int(branch.get("net_corrected_labels", 0)),
        "helpfulness_auc": _number(utility, "helpfulness_auc"),
    }


def build_traffic_evidence_summary(
    *,
    image_metrics: dict[str, Any],
    video_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    effectiveness: dict[str, Any],
    corrector: dict[str, Any],
) -> dict[str, Any]:
    """Build a no-leakage scorecard without conflating margin and traffic gains."""
    correction = corrector.get("test", {})
    action_mechanism = _mechanism_summary(effectiveness.get("action", {}))
    reason_mechanism = _mechanism_summary(effectiveness.get("reason", {}))
    rank_rows = []
    for action, row in enumerate(correction.get("traffic_rank_metrics_by_action", [])):
        rank_rows.append({
            "action": action,
            "ap_increment": _number(row, "traffic_ap_increment"),
            "auc_increment": _number(row, "traffic_auc_increment"),
        })

    image_action = _number(image_metrics, "Act_mF1")
    video_action = _number(video_metrics, "Act_mF1")
    final_action = _number(final_metrics, "Act_mF1")
    margin_action = _number(correction, "margin_only_Act_mF1")
    traffic_increment = _number(correction, "traffic_incremental_mf1")
    no_leakage = not bool(corrector.get("test_labels_used_for_fit_or_selection", True))

    checks = {
        "test_leakage_free": no_leakage,
        "positive_image_to_final_gain": final_action > image_action,
        "positive_traffic_increment": traffic_increment > 0.0,
        "positive_action_information": action_mechanism[
            "conditional_information_gain_bits"
        ] > 0.0,
        "positive_reason_information": reason_mechanism[
            "conditional_information_gain_bits"
        ] > 0.0,
        "positive_deletion_ci": (
            _ci_is_positive(
                effectiveness.get("action", {}),
                "selected_minus_random_deletion_gap_ci95",
            )
            and _ci_is_positive(
                effectiveness.get("reason", {}),
                "selected_minus_random_deletion_gap_ci95",
            )
        ),
        "selective_correction_precision_above_chance": _number(
            correction, "traffic_correction_precision"
        ) > 0.5,
    }
    failed = [name for name, passed in checks.items() if not passed]
    claim = "causal_transport_supported" if not failed else "predictive_association_only"

    return {
        "schema_version": "tida_traffic_evidence_scorecard_v1",
        "task_metrics": {
            "image": image_metrics,
            "video": video_metrics,
            "final": final_metrics,
        },
        "task_gain": {
            "image_to_video_action_mf1": video_action - image_action,
            "video_to_final_action_mf1": final_action - video_action,
            "image_to_final_action_mf1": final_action - image_action,
            "margin_over_video_action_mf1": margin_action - video_action,
            "traffic_over_margin_action_mf1": traffic_increment,
            "image_to_video_reason_mf1": (
                _number(video_metrics, "Exp_mF1") - _number(image_metrics, "Exp_mF1")
            ),
        },
        "selective_correction": {
            "routes": corrector.get("deployment_routes", []),
            "correction_precision": _number(
                correction, "traffic_correction_precision"
            ),
            "errors_recovered": int(correction.get("traffic_errors_recovered", 0)),
            "correct_damaged": int(correction.get("traffic_correct_damaged", 0)),
            "net_corrected": int(correction.get("traffic_net_corrected", 0)),
        },
        "traffic_rank_gain": rank_rows,
        "mechanism": {
            "action": action_mechanism,
            "reason": reason_mechanism,
        },
        "interaction_risk_quartiles": effectiveness.get(
            "interaction_risk_quartiles", []
        ),
        "evidence_verdict": {
            "claim": claim,
            "test_leakage_free": no_leakage,
            "checks": checks,
            "failed_checks": failed,
            "claim_boundary": (
                "Traffic evidence is called causal only when train-only selection, "
                "positive proper-score information, positive selected-vs-random "
                "deletion confidence intervals, and positive deployment gain agree."
            ),
        },
    }


def _metric_card(label: str, value: float, note: str = "") -> str:
    return (
        '<article class="card"><span>' + html.escape(label) + '</span>'
        f'<strong>{value:.4f}</strong><small>{html.escape(note)}</small></article>'
    )


def _bar(label: str, value: float, maximum: float = 0.8) -> str:
    width = max(0.0, min(100.0, 100.0 * value / maximum))
    return (
        '<div class="bar-row"><span>' + html.escape(label) + '</span>'
        f'<div class="track"><i style="width:{width:.2f}%"></i></div>'
        f'<b>{value:.4f}</b></div>'
    )


def render_traffic_evidence_html(
    summary: dict[str, Any], *, case_images: Iterable[str] = ()
) -> str:
    task = summary["task_metrics"]
    gain = summary["task_gain"]
    mechanism = summary["mechanism"]
    correction = summary["selective_correction"]
    verdict = summary["evidence_verdict"]
    images = "".join(
        f'<figure><img src="{html.escape(path)}" alt="traffic correction case">'
        f'<figcaption>{html.escape(path)}</figcaption></figure>'
        for path in case_images
    )
    rank_rows = "".join(
        "<tr>"
        f"<td>action {row['action']}</td>"
        f"<td>{row['ap_increment']:+.5f}</td>"
        f"<td>{row['auc_increment']:+.5f}</td>"
        "</tr>"
        for row in summary["traffic_rank_gain"]
    )
    evidence_json = html.escape(json.dumps(verdict["checks"], ensure_ascii=False))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Traffic Evidence Scorecard</title>
<style>
:root{{--ink:#17312b;--cream:#f4efe2;--green:#2d6a4f;--gold:#d49a34;--red:#a33b32}}
body{{margin:0;background:linear-gradient(145deg,#f4efe2,#e6efe7);color:var(--ink);font-family:Georgia,serif}}
main{{max-width:1100px;margin:auto;padding:38px}} h1{{font-size:42px;margin:0 0 8px}}
.lede{{max-width:820px;line-height:1.6}} .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}
.card{{background:#fff9;border:1px solid #17312b22;padding:18px;border-radius:4px;box-shadow:0 7px 20px #17312b10}}
.card span,.card small{{display:block}} .card strong{{font-size:29px;color:var(--green)}}
section{{background:#fff8;padding:24px;margin:18px 0;border-left:5px solid var(--gold)}}
.bar-row{{display:grid;grid-template-columns:100px 1fr 80px;gap:12px;align-items:center;margin:12px 0}}
.track{{height:14px;background:#17312b12}} .track i{{display:block;height:100%;background:linear-gradient(90deg,var(--green),var(--gold))}}
table{{width:100%;border-collapse:collapse}} td,th{{padding:9px;border-bottom:1px solid #17312b22;text-align:left}}
.cases{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}} img{{max-width:100%}}
.verdict{{font-weight:bold;color:{'#2d6a4f' if verdict['claim']=='causal_transport_supported' else '#a33b32'}}}
@media(max-width:720px){{.cards{{grid-template-columns:1fr 1fr}}main{{padding:20px}}}}
</style></head><body><main>
<h1>Traffic Evidence Scorecard</h1>
<p class="lede">This report separates temporal representation gain, selective traffic correction, and mechanism evidence. It does not credit margin-only threshold effects to traffic flow.</p>
<div class="cards">
{_metric_card('Image Act_mF1', _number(task['image'],'Act_mF1'))}
{_metric_card('Video Act_mF1', _number(task['video'],'Act_mF1'), f"{gain['image_to_video_action_mf1']:+.4f} vs image")}
{_metric_card('Final Act_mF1', _number(task['final'],'Act_mF1'), f"{gain['image_to_final_action_mf1']:+.4f} vs image")}
{_metric_card('Traffic-only gain', gain['traffic_over_margin_action_mf1'], 'over margin-only')}
</div>
<section><h2>Overall task progression</h2>
{_bar('Image', _number(task['image'],'Act_mF1'))}
{_bar('Video', _number(task['video'],'Act_mF1'))}
{_bar('Final', _number(task['final'],'Act_mF1'))}
</section>
<section><h2>Conditional information and causal deletion</h2>
<p>Action conditional information: <b>{mechanism['action']['conditional_information_gain_bits']:.6f} bits</b>; reason: <b>{mechanism['reason']['conditional_information_gain_bits']:.6f} bits</b>.</p>
<p>Selected vs random deletion: action <b>{mechanism['action']['selected_minus_random_deletion_gap']:+.6f}</b>; reason <b>{mechanism['reason']['selected_minus_random_deletion_gap']:+.6f}</b>.</p>
<p>Traffic corrections recovered {correction['errors_recovered']} errors, damaged {correction['correct_damaged']}, net {correction['net_corrected']}, precision {correction['correction_precision']:.1%}.</p>
</section>
<section><h2>Target-specific ranking gain</h2><table><thead><tr><th>Target</th><th>AP increment</th><th>AUC increment</th></tr></thead><tbody>{rank_rows}</tbody></table></section>
<section><h2>Evidence verdict</h2><p class="verdict">{html.escape(verdict['claim'])}</p><code>{evidence_json}</code><p>{html.escape(verdict['claim_boundary'])}</p></section>
<section><h2>Traffic correction cases</h2><div class="cases">{images}</div></section>
</main></body></html>"""
