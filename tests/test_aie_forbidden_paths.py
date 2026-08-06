from pathlib import Path

from fate_oia.utils.aie_contracts import scan_forbidden


def test_aie_formal_sources_have_no_forbidden_paths():
    paths = [p for p in Path("fate_oia").rglob("aie_*.py") if p.name != "aie_contracts.py"]
    paths += list(Path("scripts").glob("*aie*.ps1")) + list(Path("configs").glob("*aie*.yaml"))
    assert scan_forbidden(paths) == {}

