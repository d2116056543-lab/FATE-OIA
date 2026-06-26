from pathlib import Path


def test_gem_supervisor_script_is_foreground_only():
    text = Path("scripts/FATE_OIA_acpr_gem_v1_foreground.ps1").read_text(encoding="utf-8")
    forbidden = ["Start-Process", "Start-Job", "nohup", "scheduled task", "hidden cmd"]
    for pat in forbidden:
        assert pat not in text
    assert "supervise_acpr_gem_foreground" in text


def test_gem_supervisor_uses_worker_arg_for_smoke_and_defaults_to_four():
    ps1 = Path("scripts/FATE_OIA_acpr_gem_v1_foreground.ps1").read_text(encoding="utf-8")
    py = Path("fate_oia/engine/supervise_acpr_gem_foreground.py").read_text(encoding="utf-8")
    assert "[int]$NumWorkers = 4" in ps1
    assert 'parser.add_argument("--num_workers", type=int, default=4)' in py
    assert '"--num_workers",\n        str(args.num_workers),' in py
    assert '"--num_workers",\n        "0",' not in py
