from fate_oia.utils.aie_cert_schedule import schedule_values


def test_schedule_is_update_based_continuous_and_resumable():
    a=schedule_values(10,100,{}); b=schedule_values(10,100,{})
    assert a==b and a['action_scale']>0.1 and a['reason_budget_max']>.1
    assert schedule_values(4,100,{})['cf_scale']==0


def test_action_scale_uses_the_configured_safety_cap():
    cfg = {"evidence": {"action_scale_start": 0.1, "action_scale_max": 0.25}}
    assert schedule_values(0, 100, cfg)["action_scale"] == 0.1
    assert schedule_values(100, 100, cfg)["action_scale"] == 0.25
