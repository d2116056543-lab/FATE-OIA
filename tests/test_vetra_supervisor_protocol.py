from pathlib import Path


def test_supervisor_orders_all_hard_gates_before_training():
    text=(Path(__file__).parents[1]/"fate_oia/engine/supervise_vetra_oia_probe.py").read_text()
    audit=text.index("audit_vetra_oia_probe"); replay=text.index("replay_vetra_source")
    profile=text.index("profile_vetra_oia"); train=text.index("train_vetra_oia_probe")
    assert audit < replay < profile < train
    assert "subprocess.run(args, check=True)" in text
