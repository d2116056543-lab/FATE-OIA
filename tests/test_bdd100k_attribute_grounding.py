from fate_oia.datasets.mosaic_icdor_grounding import parse_bdd100k_attributes


def test_bdd100k_attributes_are_preserved():
    row = {"attributes": {"trafficLightColor": "red", "occluded": 1, "truncated": 0}}
    parsed = parse_bdd100k_attributes(row)
    assert parsed["traffic_light_color"] == "red"
    assert parsed["occluded"] is True
    assert parsed["truncated"] is False

