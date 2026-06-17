from fate_oia.utils.acpr_pmt_phase_schedule import pmt_phase_for_epoch


def test_pmt_phase_schedule():
    assert pmt_phase_for_epoch(0)["phase"] == "warmup"
    assert pmt_phase_for_epoch(3)["phase"] == "pmt"
    stable = pmt_phase_for_epoch(9)
    assert stable["phase"] == "stable"
    assert stable["pair_cap_ratio"] == 0.05
    assert stable["triadic_lr_multiplier"] == 0.5
