from fate_oia.models.tida_terminal_query_reader import TIDATerminalQueryReader


def test_reader_uses_explicit_semantic_layer_order():
    reader = TIDATerminalQueryReader(dim=8, layer_ids=(3, 7, 11))
    assert reader.layer_id_to_index == {3: 0, 7: 1, 11: 2}
    assert reader.read_order == (11, 7, 3)
