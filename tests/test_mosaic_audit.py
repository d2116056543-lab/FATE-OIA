from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from fate_oia.engine import audit_acpr_mosaic_ad as implementation_audit


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = REPO_ROOT / ".codex" / "skills" / "acpr-mosaic-ad-implementation-audit" / "SKILL.md"
EXPECTED_BODY_SHA256 = "912955363D8302ABE67CCF6A248B56473B311636021E734D2E1212D9241D8072"


def test_required_files_gate_returns_a_real_complete_result() -> None:
    result = implementation_audit._required_files_gate(REPO_ROOT)
    assert isinstance(result, dict)
    assert result["pass"] is True
    assert result["missing_files"] == []
    assert result["missing_tests"] == []


def test_user_approved_pre_full_pilot_uses_one_complete_seed() -> None:
    assert implementation_audit.PILOT_SEEDS == (20260710,)
    script = (REPO_ROOT / "scripts" / "FATE_OIA_acpr_mosaic_ad_v1_foreground.ps1").read_text(
        encoding="utf-8-sig"
    )
    assert "foreach ($seed in @(20260710))" in script
    assert "20260711" not in script
    assert "20260712" not in script
    assert '$env:TEMP = $runtimeTemp' in script
    assert '$env:TMP = $runtimeTemp' in script


def _split_skill(text: str) -> tuple[str, str]:
    assert text.startswith("---\n"), "skill must start with YAML frontmatter"
    frontmatter, separator, body = text[4:].partition("\n---\n")
    assert separator, "skill frontmatter must have a closing delimiter"
    return frontmatter, body


def _parse_skill(text: str) -> tuple[dict[str, object], str]:
    frontmatter_text, body = _split_skill(text)
    frontmatter = yaml.safe_load(frontmatter_text)
    assert isinstance(frontmatter, dict), "skill frontmatter must be a YAML mapping"
    assert frontmatter.get("name") == "acpr-mosaic-ad-implementation-audit", "invalid frontmatter name"
    assert isinstance(frontmatter.get("description"), str) and frontmatter["description"].strip(), (
        "skill frontmatter description must be non-empty"
    )
    return frontmatter, body


def test_skill_frontmatter_rejects_malformed_keys() -> None:
    malformed = "---\nxname: acpr-mosaic-ad-implementation-audit\nnodescription: invalid\n---\nbody"
    with pytest.raises(AssertionError, match="frontmatter name"):
        _parse_skill(malformed)


def test_skill_is_discoverable_and_preserves_supplied_body_hash() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    _, body = _parse_skill(text)
    assert hashlib.sha256(body.encode("utf-8")).hexdigest().upper() == EXPECTED_BODY_SHA256

    for hard_gate in (
        "## 9. Geometry-typed attention gate",
        "## 14. Action firewall gate",
        "## 16. Selective observation mathematical gate",
        "## 19. Action-anchored trust-update gate",
        "## 20. Calibration gate",
        "## 24. Pilot gate",
    ):
        assert hard_gate in body
